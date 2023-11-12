@echo off
cd %~dp0
cd ..
docker pull emscripten/emsdk:3.1.43
docker run --rm -v %CD%:/src emscripten/emsdk:3.1.43 node script/wasm_build.js
