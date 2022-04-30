@echo off
cd %~dp0
cd ..
docker pull emscripten/emsdk:latest
docker run --rm -v %CD%:/src emscripten/emsdk:latest node script/wasm_build.js
