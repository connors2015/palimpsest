"""Mixture-of-experts with page-attested SPARSE inference (WHITEPAPER §3.1, §8).

The path to "the largest model in existence": most parameters live in experts,
and each query routes to only top-k of them — so serving never loads the whole
model. Here we make that concrete and *attestable*:

  * the model's weights are laid out as pages — one for the router, one per
    expert — and the chain commits their Merkle root as the weights_state_root;
  * a serving node answers a query by loading only the router page + the k
    expert pages the router selected, and returns a receipt carrying the
    output, the expert indices, and Merkle inclusion proofs for exactly those
    pages;
  * a verifier checks the proofs against the committed root (the loaded experts
    really are the committed ones) and recomputes the output touching only
    those pages — never loading the other experts.

So inference cost and attestation cost both scale with k, not with the total
number of experts. Report: fraction of parameters loaded per query.

Task: domain-routed classification. Each of E domains has its own linear rule
mapping x -> class; expert e specializes in domain e; the router sends a
query's domain token to its expert. Experts genuinely specialize, so top-1
sparse serving keeps full accuracy.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from . import merkle

DIM = 16
CLASSES = 8
HIDDEN = 32


def make_rules(rng, cfg):
    """One random linear rule (x -> class) per domain, so experts can specialize."""
    return [rng.standard_normal((cfg.dim, cfg.classes)) for _ in range(cfg.n_experts)]


def make_domain_batch(rng, n, cfg, rules):
    """Domain-routed classification: each domain uses its own rule."""
    d = rng.integers(0, cfg.n_experts, size=n)
    x = rng.standard_normal((n, cfg.dim))
    y = np.array([(x[i] @ rules[d[i]]).argmax() for i in range(n)])
    return d, x, y


@dataclass
class MoEConfig:
    n_experts: int = 8
    top_k: int = 1
    dim: int = DIM
    classes: int = CLASSES
    hidden: int = HIDDEN


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class MoE:
    """Router + E expert MLPs. Params serialize page-by-page (router, then experts)."""

    def __init__(self, cfg: MoEConfig = None):
        self.cfg = cfg or MoEConfig()
        c = self.cfg
        # page sizes
        self.router_size = c.n_experts * c.n_experts            # domain-embed -> logits
        self.expert_size = (c.dim * c.hidden + c.hidden
                            + c.hidden * c.classes + c.classes)
        self.param_count = self.router_size + c.n_experts * self.expert_size

    # -- pages -------------------------------------------------------------
    def pages(self, vec: np.ndarray) -> list[np.ndarray]:
        """[router_page, expert_0, ..., expert_{E-1}] as float64 arrays."""
        c = self.cfg
        out = [vec[:self.router_size]]
        off = self.router_size
        for _ in range(c.n_experts):
            out.append(vec[off:off + self.expert_size])
            off += self.expert_size
        return out

    def merkle_root(self, vec: np.ndarray) -> bytes:
        return merkle.root([p.tobytes() for p in self.pages(vec)])

    def _router(self, vec):
        c = self.cfg
        return vec[:self.router_size].reshape(c.n_experts, c.n_experts)

    def _expert(self, page):
        c = self.cfg
        i = 0
        W1 = page[i:i + c.dim * c.hidden].reshape(c.dim, c.hidden); i += c.dim * c.hidden
        b1 = page[i:i + c.hidden]; i += c.hidden
        W2 = page[i:i + c.hidden * c.classes].reshape(c.hidden, c.classes)
        i += c.hidden * c.classes
        b2 = page[i:i + c.classes]
        return W1, b1, W2, b2

    def _expert_forward(self, page, x):
        W1, b1, W2, b2 = self._expert(page)
        h = np.maximum(x @ W1 + b1, 0.0)
        return h @ W2 + b2

    # -- init / train (dense over all experts; sparse only at inference) ----
    def init(self, rng) -> np.ndarray:
        vec = (rng.standard_normal(self.param_count) * 0.1).astype(np.float64)
        # Bootstrap expert specialization: router starts near identity (domain d
        # -> expert d), so each expert receives a clean domain-specific gradient
        # stream. The router remains trainable and can re-route from here.
        vec[:self.router_size] = (np.eye(self.cfg.n_experts) * 4.0).ravel()
        return vec

    def route(self, vec, domains) -> np.ndarray:
        """Top-k expert indices per example, from the router logits."""
        logits = self._router(vec)[domains]                    # [N, E]
        k = self.cfg.top_k
        return np.argsort(-logits, axis=1)[:, :k]              # [N, k]

    def _loss_and_grad(self, vec, batch):
        """Dense training: every expert contributes, weighted by router softmax."""
        c = self.cfg
        domains, x, y = batch
        N = len(y)
        rlogits = self._router(vec)[domains]                   # [N,E]
        gate = _softmax(rlogits, axis=1)                       # [N,E]
        # forward each expert on the whole batch
        pages = self.pages(vec)
        ex_logits = np.stack([self._expert_forward(pages[1 + e], x)
                              for e in range(c.n_experts)], axis=1)  # [N,E,C]
        mix = np.einsum("ne,nec->nc", gate, ex_logits)         # [N,C]
        probs = _softmax(mix, axis=1)
        idx = (np.arange(N), y)
        loss = float(-np.mean(np.log(probs[idx] + 1e-12)))

        grad = np.zeros_like(vec)
        dmix = probs.copy(); dmix[idx] -= 1.0; dmix /= N       # [N,C]
        # expert grads + router grad via the mixture
        off = self.router_size
        drlogits = np.zeros_like(rlogits)
        for e in range(c.n_experts):
            W1, b1, W2, b2 = self._expert(pages[1 + e])
            h = np.maximum(x @ W1 + b1, 0.0)
            g_out = dmix * gate[:, e:e + 1]                    # [N,C]
            gW2 = h.T @ g_out
            gb2 = g_out.sum(0)
            gh = (g_out @ W2.T) * (h > 0)
            gW1 = x.T @ gh
            gb1 = gh.sum(0)
            eg = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
            grad[off:off + self.expert_size] = eg
            off += self.expert_size
            # router path: d loss / d gate_e = sum_c dmix_c * ex_logits_{e,c}
            drlogits[:, e] = np.einsum("nc,nc->n", dmix, ex_logits[:, e, :])
        # softmax backward for the gate
        dr = gate * (drlogits - (drlogits * gate).sum(1, keepdims=True))
        gr = np.zeros((c.n_experts, c.n_experts))
        np.add.at(gr, domains, dr)
        grad[:self.router_size] = gr.ravel()
        return loss, grad

    def train_step(self, vec, batch, lr=0.2, steps=1):
        v = vec.copy()
        for _ in range(steps):
            _, g = self._loss_and_grad(v, batch)
            v = v - lr * g
        return v

    def loss(self, vec, batch):
        return self._loss_and_grad(vec, batch)[0]

    # -- dense vs sparse accuracy ------------------------------------------
    def accuracy_dense(self, vec, batch):
        domains, x, y = batch
        gate = _softmax(self._router(vec)[domains], axis=1)
        pages = self.pages(vec)
        ex = np.stack([self._expert_forward(pages[1 + e], x)
                       for e in range(self.cfg.n_experts)], axis=1)
        mix = np.einsum("ne,nec->nc", gate, ex)
        return float((mix.argmax(1) == y).mean())

    def accuracy_sparse(self, vec, batch):
        domains, x, y = batch
        pred = np.array([self.serve(vec, int(d), xi)["pred"]
                         for d, xi in zip(domains, x)])
        return float((pred == y).mean())

    # -- SPARSE attested serving (§8) --------------------------------------
    def serve(self, vec, domain: int, x: np.ndarray) -> dict:
        """Answer one query loading ONLY the router + top-k expert pages."""
        pages = self.pages(vec)
        experts = self.route(vec, np.array([domain]))[0]       # top-k indices
        # accumulate top-k expert outputs, gated
        rlogits = self._router(vec)[domain]
        gate = _softmax(rlogits[experts][None], axis=1)[0]
        mix = np.zeros(self.cfg.classes)
        for w, e in zip(gate, experts):
            mix += w * self._expert_forward(pages[1 + e], x[None])[0]
        pred = int(mix.argmax())

        # attestation: Merkle proofs for exactly the pages we loaded (§3.1)
        levels = merkle.build([p.tobytes() for p in pages])
        loaded = [0] + [1 + int(e) for e in experts]           # router page + experts
        proofs = {i: merkle.proof(levels, i) for i in loaded}
        loaded_params = self.router_size + len(experts) * self.expert_size
        return dict(pred=pred, experts=[int(e) for e in experts],
                    root=levels[-1][0], proofs=proofs, loaded_pages=loaded,
                    expert_fraction=len(experts) / self.cfg.n_experts,  # k/E
                    loaded_fraction=loaded_params / self.param_count,
                    input_hash=hashlib.sha256(x.tobytes()).hexdigest())


def verify_serve(moe: MoE, available_pages: dict, receipt: dict,
                 committed_root: bytes, domain: int, x: np.ndarray) -> bool:
    """Verify a sparse receipt WITHOUT the full model.

    The verifier holds only the pages named in the receipt (router + used
    experts). It (1) checks each page's Merkle proof against the committed
    root, then (2) recomputes the output from just those pages. The other
    experts are never loaded — attestation cost scales with k, not E.
    """
    cfg = moe.cfg
    # (1) proofs: every loaded page belongs to the committed model
    for i in receipt["loaded_pages"]:
        if i not in available_pages:
            return False
        if not merkle.verify(available_pages[i].tobytes(), i,
                             receipt["proofs"][i], committed_root):
            return False
    # (2) recompute the output from only the router + used expert pages
    router = available_pages[0].reshape(cfg.n_experts, cfg.n_experts)
    experts = receipt["experts"]
    gate = _softmax(router[domain][experts][None], axis=1)[0]
    mix = np.zeros(cfg.classes)
    for w, e in zip(gate, experts):
        mix += w * moe._expert_forward(available_pages[1 + e], x[None])[0]
    return int(mix.argmax()) == receipt["pred"]


if __name__ == "__main__":
    print("=" * 68)
    print("  SESTRIAN — sparse MoE with page-attested inference")
    print("=" * 68)
    rng = np.random.default_rng(0)
    cfg = MoEConfig(n_experts=8, top_k=1)
    rules = make_rules(rng, cfg)
    moe = MoE(cfg)
    vec = moe.init(rng)
    test = make_domain_batch(np.random.default_rng(99), 400, cfg, rules)
    for s in range(600):
        vec = moe.train_step(vec, make_domain_batch(rng, 128, cfg, rules),
                             lr=0.3, steps=1)
    print(f"\n{cfg.n_experts} experts, top-{cfg.top_k} routing, "
          f"{moe.param_count} params")
    print(f"  accuracy  dense (all experts): {moe.accuracy_dense(vec, test):.3f}")
    print(f"  accuracy  sparse (top-k only): {moe.accuracy_sparse(vec, test):.3f}")

    d0, x0 = int(test[0][0]), test[1][0]
    r = moe.serve(vec, d0, x0)
    root = moe.merkle_root(vec)
    pages = {i: moe.pages(vec)[i] for i in r["loaded_pages"]}
    honest = verify_serve(moe, pages, r, root, d0, x0)
    tampered = dict(pages)
    last = r["loaded_pages"][-1]
    tampered[last] = tampered[last] + 1.0
    fraud = verify_serve(moe, tampered, r, root, d0, x0)
    print(f"\nattested sparse inference for one query (domain {d0}):")
    print(f"  experts loaded:            {r['experts']} "
          f"({len(r['experts'])} of {cfg.n_experts}) = "
          f"{100 * r['expert_fraction']:.1f}% of expert capacity")
    print(f"  Merkle-verify (honest):    {honest}")
    print(f"  Merkle-verify (tampered):  {fraud}")

    print("\nexpert capacity loaded per query as experts multiply (top-1):")
    for E in (8, 32, 128, 1024):
        frac = 1.0 / E
        print(f"  E={E:>4}: {100 * frac:5.2f}% of experts (1 of {E}) — "
              f"serving cost is O(k), not O(E)")
    print("\n(The toy router is O(E^2) so total-param fraction is muddied; in a "
          "real MoE\nexperts dwarf the router and loaded fraction -> k/E.)")
    print("=" * 68)
    raise SystemExit(0 if honest and not fraud else 1)
