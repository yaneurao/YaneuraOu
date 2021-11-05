//
// Generate embedded_nnue.cpp
//
// usage example:
// node source/eval/nnue/embedded_nnue.js nn.bin > source/eval/nnue/embedded_nnue.cpp
//
const process = require("process");
const fs = require("fs");

const ASCII_backslash = "\\".codePointAt(0); // = 92
const ASCII_x = "x".codePointAt(0); // = 120

async function main() {
  //
  // Example
  //
  // file-content         = "AB"
  // data    : Buffer     = [65, 66]
  // hex     : String     = "4142"
  // ascii   : Uint8Array = [52, 49, 52, 50]
  // literal : Uint8Array = [92, 120, 52, 49, 92, 120, 52, 50]
  //
  const data = fs.readFileSync(process.argv[2]);
  const hex = data.toString("hex");
  const ascii = new TextEncoder().encode(hex);
  const size = data.length;
  const literal = new Uint8Array(4 * size);
  for (let i = 0; i < size; i++) {
    literal[4 * i + 0] = ASCII_backslash;
    literal[4 * i + 1] = ASCII_x;
    literal[4 * i + 2] = ascii[2 * i + 0];
    literal[4 * i + 3] = ascii[2 * i + 1];
  }

  // Heap OOM if string template is used...
  const w = (s) => process.stdout.write(s);
  w(`#include <cstddef>\n`);
  w(`extern const char* gEmbeddedNNUEData;\n`);
  w(`extern const std::size_t gEmbeddedNNUESize;\n`);
  w(`const char* gEmbeddedNNUEData = "`);
  w(literal);
  w(`";\n`);
  w(`const std::size_t gEmbeddedNNUESize = ${size};\n`);
}

if (require.main == module) {
  main();
}
