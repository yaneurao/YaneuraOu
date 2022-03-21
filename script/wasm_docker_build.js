#!/usr/bin/env node
const { exec } = require("child_process");
const process = require("process");
const fs = require("fs");

const args = process.argv.slice(2);
const cwd = process.cwd();

if(!fs.existsSync("source/Makefile")) {
  console.error("source folder not found");
  process.exit(1);
}

(async () => {
  await new Promise((resolve) => {
    let child = exec(
      `docker pull emscripten/emsdk:latest`,
      { cwd: cwd, stdio: "inherit" },
      (_error, _stdout, _stderr) => { resolve(); },
    );
    child.stdout.on('data', (data) => { console.log(String(data).trimEnd()); });
    child.stderr.on('data', (data) => { console.error(String(data).trimEnd()); });
  });
  await new Promise((resolve) => {
    let child = exec(
      `docker run --rm -v ${cwd}:/src emscripten/emsdk:latest node script/wasm_build.js ${args.join(" ")}`,
      { cwd: cwd, stdio: "inherit" },
      (_error, _stdout, _stderr) => { resolve(); },
    );
    child.stdout.on('data', (data) => { console.log(String(data).trimEnd()); });
    child.stderr.on('data', (data) => { console.error(String(data).trimEnd()); });
  });
})();
