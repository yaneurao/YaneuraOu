# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS build

# Build options
# Refer to: source/Makefile
ARG TARGET=normal
ARG YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE
ARG YO_CLUSTER=OFF
ARG TARGET_CPU=AVX2
ARG DEBUG=OFF

# Install packages
RUN --mount=type=cache,target=/var/lib/apt/,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends clang lld make python3

RUN --mount=type=bind,target=/usr/local/src/YaneuraOu/ \
    --mount=type=cache,target=/tmp/ \
    cd /usr/local/src/YaneuraOu/source/ \
    && make ${TARGET} -j"$(nproc)" \
    YANEURAOU_EDITION=${YANEURAOU_EDITION} \
    YO_CLUSTER=${YO_CLUSTER} \
    TARGET_CPU=${TARGET_CPU} \
    DEBUG=${DEBUG} \
    PYTHON=python3 TARGETDIR=/usr/local/bin OBJDIR=/tmp/obj

FROM ubuntu:24.04

USER ubuntu
COPY --from=build /usr/local/bin/YaneuraOu-by-gcc /usr/local/bin/

ENTRYPOINT [ "/usr/local/bin/YaneuraOu-by-gcc" ]
