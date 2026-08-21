# Immutable release image for the Palimpsest node.
#
# Replaces the compile-on-every-boot pattern: CI builds this once, tags it by
# git SHA, and pushes it; deployments pin a digest. The genesis vector is NOT
# baked (it is a per-network launch artifact) — mount it at /genesis/genesis.bin.
#
#   docker build -f deploy/Dockerfile -t palimpsest-node:$(git rev-parse --short HEAD) .

# ---- build stage ----------------------------------------------------------
FROM rust:1-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
# copy only what the workspace needs to build (keeps the layer cache tight)
COPY node /src/node
RUN cd node && cargo build --release --bin palimpsest-node && \
    strip target/release/palimpsest-node

# ---- runtime stage --------------------------------------------------------
FROM debian:stable-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --create-home --uid 10001 palimpsest
COPY --from=build /src/node/target/release/palimpsest-node /usr/local/bin/palimpsest-node
USER palimpsest
WORKDIR /home/palimpsest
# data dir + genesis are mounted; identity comes from PALIMPSEST_KEY_SEED (Secret)
ENTRYPOINT ["palimpsest-node"]
CMD ["--help"]
