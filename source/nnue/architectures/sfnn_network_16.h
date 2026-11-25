// BulletOu nn16: byte activations, int16 FC weights, int64 accumulation.
#ifndef NNUE_SFNN_NETWORK_16_H_INCLUDED
#define NNUE_SFNN_NETWORK_16_H_INCLUDED

#include "../nnue_common.h"
#include <algorithm>
#include <cstdint>

namespace YaneuraOu::Eval::NNUE {

template <IndexType Inputs, IndexType Outputs>
struct SfnnAffine16 {
    static constexpr IndexType PaddedInputs = ceil_to_multiple<IndexType>(Inputs, 32);
    i32 biases[Outputs];
    i16 weights[Outputs][PaddedInputs];

    bool read_parameters(std::istream& stream) {
        read_little_endian(stream, biases, Outputs);
        for (auto& row : weights)
            read_little_endian(stream, row, PaddedInputs);
        return bool(stream);
    }

    bool write_parameters(std::ostream& stream) const {
        // Use the common little-endian writer without changing the nn16 layout.
        // 共通のlittle-endian書き込みを使い、nn16のデータ配置は変えない。
        write_little_endian(stream, biases, Outputs);
        for (const auto& row : weights)
            write_little_endian(stream, row, PaddedInputs);
        return bool(stream);
    }

    void propagate(const u8* input, i64* output) const {
        for (IndexType o = 0; o < Outputs; ++o) {
            i64 sum = biases[o];
            for (IndexType i = 0; i < Inputs; ++i)
                sum += i64(weights[o][i]) * input[i];
            output[o] = sum;
        }
    }
};

template <IndexType InputDims, IndexType Hidden1Dims, IndexType Hidden2Dims, bool UseShortcut>
struct SfnnNetwork16 {
    static constexpr IndexType TransformedFeatureDimensions = InputDims;
    static constexpr IndexType kHidden1Dims = Hidden1Dims;
    static constexpr IndexType FC_1_OUTPUTS = Hidden2Dims;
    static constexpr bool kUseShortcut = UseShortcut;
    // ショートカットがある構成では、その1出力もfc_0の物理的な幅に含める。
    static constexpr IndexType FC_0_OUTPUTS = kHidden1Dims + (kUseShortcut ? 1 : 0);
    static constexpr IndexType OutputDimensions = 1;
    using OutputType = i64;
    SfnnAffine16<InputDims, FC_0_OUTPUTS> fc_0;
    SfnnAffine16<Hidden1Dims * 2, Hidden2Dims> fc_1;
    SfnnAffine16<Hidden2Dims, 1> fc_2;
    u32 weight_scale = 4096;
    unsigned shift = 12;

    struct alignas(CacheLineSize) Buffer {
        i64 first[Hidden1Dims + (UseShortcut ? 1 : 0)];
        i64 second[Hidden2Dims];
        u8 first_activated[Hidden1Dims * 2];
        u8 second_activated[Hidden2Dims];
        OutputType output[1];
    };
    static constexpr usize BufferSize = sizeof(Buffer);
    static constexpr u32 get_hash_value() { return 0x6333718Au; }

    bool set_weight_scale(u32 qb) {
        if (qb < 64 || qb > 16384 || (qb & (qb - 1))) return false;
        weight_scale = qb;
        shift = 0;
        while (qb >>= 1) ++shift;
        return true;
    }

    bool read_parameters(std::istream& stream) {
        return fc_0.read_parameters(stream) && fc_1.read_parameters(stream) && fc_2.read_parameters(stream);
    }
    bool write_parameters(std::ostream& stream) const {
        return fc_0.write_parameters(stream) && fc_1.write_parameters(stream) && fc_2.write_parameters(stream);
    }

    u8 clipped(i64 z) const {
        return u8(std::clamp<i64>(z / weight_scale, 0, 127));
    }
    u8 squared(i64 z) const {
        // Beyond this bound the result saturates. Inside it, z*z fits int64
        // even for QB=16384. This also handles INT64_MIN without negation.
        const i64 bound = i64(weight_scale) * 128;
        if (z <= -bound || z >= bound) return 127;
        return u8(std::min<i64>((z * z) >> (2 * shift + 7), 127));
    }

    const OutputType* propagate(const TransformedFeatureType* input, char* buffer) const {
        auto& b = *reinterpret_cast<Buffer*>(buffer);
        fc_0.propagate(input, b.first);
        for (IndexType i = 0; i < Hidden1Dims; ++i) {
            b.first_activated[i] = squared(b.first[i]);
            b.first_activated[Hidden1Dims + i] = clipped(b.first[i]);
        }
        fc_1.propagate(b.first_activated, b.second);
        for (IndexType i = 0; i < Hidden2Dims; ++i)
            b.second_activated[i] = clipped(b.second[i]);
        fc_2.propagate(b.second_activated, b.output);
        if constexpr (UseShortcut) b.output[0] += b.first[Hidden1Dims];
        // Restore the QB=64 units expected by FV_SCALE (truncate toward zero).
        b.output[0] /= weight_scale / 64;
        return b.output;
    }
};

} // namespace YaneuraOu::Eval::NNUE
#endif
