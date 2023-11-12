#!/bin/bash
pushd `dirname $0`
pushd ../
docker pull emscripten/emsdk:3.1.43
docker run --rm -v ${PWD}:/src emscripten/emsdk:3.1.43 node script/wasm_build.js
popd
popd
