# Dev environment for fairway: the toolchain `cargo xtask dev-check` needs.
# Works with Docker, Podman and Buildah (plain Dockerfile, no BuildKit-only
# features). The sources are mounted, not copied:
#
#   podman build -t fairway-dev .            # or: buildah bud / docker build
#   podman run --rm -it -v "$PWD":/work:Z fairway-dev                # dev-check
#   podman run --rm -it -v "$PWD":/work:Z fairway-dev cargo xtask coverage --html
#   podman run --rm -it -v "$PWD":/work:Z fairway-dev bash
#
# With rootful Docker, add `--user "$(id -u):$(id -g)"` so build outputs on the
# mounted tree stay owned by you.

ARG RUST_VERSION=1.99
FROM docker.io/library/rust:${RUST_VERSION}-slim

ARG CARGO_LLVM_COV_VERSION=0.9.1

# The chmod lets any `--user` use the toolchain and the cargo cache.
RUN rustup component add rustfmt clippy llvm-tools \
    && cargo install --locked "cargo-llvm-cov@${CARGO_LLVM_COV_VERSION}" \
    && rm -rf "${CARGO_HOME}/registry" "${CARGO_HOME}/git" \
    && chmod -R a+rwX "${CARGO_HOME}" "${RUSTUP_HOME}"

# Keep container builds apart from host builds (different toolchain/libc),
# while reports such as target/container/llvm-cov/html stay visible on the host.
ENV CARGO_TARGET_DIR=/work/target/container \
    CARGO_TERM_COLOR=always

WORKDIR /work
CMD ["cargo", "xtask", "dev-check"]
