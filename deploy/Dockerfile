# Immutable release image for the Sestrian node.
#
# Replaces the compile-on-every-boot pattern: CI builds this once, tags it by
# git SHA, and pushes it; deployments pin a digest. The genesis vector is NOT
# baked (it is a per-network launch artifact) — mount it at /genesis/genesis.bin.
#
#   docker build -f deploy/Dockerfile -t sestrian-node:$(git rev-parse --short HEAD) .

# ---- build stage ----------------------------------------------------------
FROM rust:1-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
# copy only what the workspace needs to build (keeps the layer cache tight)
COPY node /src/node
RUN cd node && cargo build --release --bin sestrian-node && \
    strip target/release/sestrian-node

# ---- runtime stage --------------------------------------------------------
FROM debian:stable-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --create-home --uid 10001 sestrian
COPY --from=build /src/node/target/release/sestrian-node /usr/local/bin/sestrian-node
USER sestrian
WORKDIR /home/sestrian
# data dir + genesis are mounted; identity comes from SESTRIAN_KEY_SEED (Secret)
ENTRYPOINT ["sestrian-node"]
CMD ["--help"]
