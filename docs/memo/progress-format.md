# SFNN Progress Files

For an SFNN architecture with progress buckets, place `nn.bin` and
`progress.bin` together in the directory selected by `EvalDir`.
`progress.bin` is required; it is not embedded in `nn.bin`.

The progress file uses the tatara / nodchip PR #326 format:

- 81 king squares by 1548 BonaPiece features, in that order.
- 125,388 little-endian IEEE 754 doubles (1,003,104 bytes).
- No header, bias, or trailing data.

Weights are rounded to Q16 integers on loading. Position sums use 64-bit
integers. Bucket boundaries are `round(65536 * log(i / (N - i)))`, where
`N` is the number of progress buckets and `i` runs from 1 to `N - 1`.
Equality selects the higher bucket. For progress8 these boundaries are
`-127527, -71999, -33477, 0, 33477, 71999, 127527`.

The SFNN `nn.bin` structure hash is `0x3C203B32`, matching
[the reference branch](https://github.com/nodchip/tanuki-/blob/f095a235866e369d13e737645cbc6abcfece0561/source/eval/nnue/evaluate_nnue.h).
The file contains the header, feature transformer, then layer stacks;
there is no Progress section. Network dimensions and layer-stack ordering
must also match the engine architecture. Old files containing an embedded
Progress section must be re-exported with the new writer.

Use an external `EvalDir` for these architectures, not `EvalDir=<internal>`.
The engine reports a loading error if `progress.bin` is missing, truncated,
has trailing bytes, or contains non-finite weights.
