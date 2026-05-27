#include "tanuki_progress.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "incbin/incbin.h"

#include "misc.h"
#include "position.h"

using namespace YaneuraOu;

namespace {

constexpr char kProgressFilePath[] = "ProgressFilePath";
constexpr char kInternalPath[] = "<internal>";

// logit((i+1)/8) を Q16.16 に丸めた閾値 + 番兵
static constexpr int64_t kMaxAbsSumQ16 =
    static_cast<int64_t>(YaneuraOu::PIECE_NUMBER_KING) * 2 * static_cast<int64_t>(INT32_MAX);
static constexpr int64_t kThresholdsQ16[8] = {
    -127527, -71999, -33477, 0, 33477, 71999, 127527, kMaxAbsSumQ16 + 1,
};

constexpr double kQ16Scale = 65536.0;
constexpr int kWeightCount = static_cast<int>(YaneuraOu::SQ_NB) * static_cast<int>(YaneuraOu::Eval::fe_end);
constexpr size_t kRawWeightsBytes = kWeightCount * sizeof(double);

YaneuraOu::OptionsMap* g_options = nullptr;
int32_t g_weights_q16[YaneuraOu::SQ_NB][YaneuraOu::Eval::fe_end] = {};

int32_t to_q16(double value) {
    const double scaled = std::round(value * kQ16Scale);
    const double clamped = std::clamp(scaled, static_cast<double>(INT32_MIN), static_cast<double>(INT32_MAX));
    return static_cast<int32_t>(clamped);
}

void load_weights_from_raw(const double* raw_weights) {
    for (int sq = 0; sq < YaneuraOu::SQ_NB; ++sq) {
        for (int piece = 0; piece < YaneuraOu::Eval::fe_end; ++piece) {
            const int index = sq * static_cast<int>(YaneuraOu::Eval::fe_end) + piece;
            g_weights_q16[sq][piece] = to_q16(raw_weights[index]);
        }
    }
}

inline int32_t contribution(YaneuraOu::Square sq, int bona_piece) {
    return g_weights_q16[sq][bona_piece];
}

int32_t compute_full_sum_q16(const YaneuraOu::Position& pos, YaneuraOu::Square sq_bk, YaneuraOu::Square sq_wk) {
    const auto& list0 = pos.eval_list()->piece_list_fb();
    const auto& list1 = pos.eval_list()->piece_list_fw();

    int32_t sum_q16 = 0;
    for (int i = 0; i < YaneuraOu::PIECE_NUMBER_KING; ++i) {
        sum_q16 += contribution(sq_bk, list0[i]);
        sum_q16 += contribution(sq_wk, list1[i]);
    }
    return sum_q16;
}

bool try_get_sum_from_cache(const YaneuraOu::Position& pos, YaneuraOu::Square sq_bk, YaneuraOu::Square sq_wk,
                            int32_t& sum_q16) {
    auto* st = pos.state();
    if (!st->tanuki_progress_valid) return false;
    if (st->tanuki_progress_key != pos.key()) return false;
    if (st->tanuki_progress_sq_bk != sq_bk || st->tanuki_progress_sq_wk != sq_wk) return false;

    sum_q16 = st->tanuki_progress_sum;
    return true;
}

void store_sum_cache(const YaneuraOu::Position& pos, YaneuraOu::Square sq_bk, YaneuraOu::Square sq_wk, int32_t sum_q16) {
    auto* st = pos.state();
    st->tanuki_progress_key = pos.key();
    st->tanuki_progress_sum = sum_q16;
    st->tanuki_progress_sq_bk = sq_bk;
    st->tanuki_progress_sq_wk = sq_wk;
    st->tanuki_progress_valid = true;
}

int table_index_linear_q16(int32_t sum_q16) {
    int idx = 0;
    while (sum_q16 >= kThresholdsQ16[idx]) {
        ++idx;
    }
    return idx;
}

#if !defined(_MSC_VER)
INCBIN(EmbeddedProgress, "progress.bin");
#else
const unsigned char gEmbeddedProgressData[1] = {0};
const unsigned char* const gEmbeddedProgressEnd = &gEmbeddedProgressData[1];
const unsigned int gEmbeddedProgressSize = 1;
#endif

}  // namespace

namespace Tanuki {
namespace Progress {

bool add_options(YaneuraOu::OptionsMap& options) {
    g_options = &options;
    options.add(kProgressFilePath, YaneuraOu::Option(kInternalPath));
    return true;
}

bool Load() {
    if (g_options == nullptr) {
        sync_cout << "info string Progress options are not initialized." << sync_endl;
        return false;
    }

    const std::string file_path = (*g_options)[kProgressFilePath];

    if (file_path == kInternalPath) {
        if (gEmbeddedProgressSize != kRawWeightsBytes) {
            sync_cout << "info string Embedded progress size mismatch. expected=" << kRawWeightsBytes
                      << " actual=" << gEmbeddedProgressSize << sync_endl;
            return false;
        }

        std::vector<double> raw_weights(kWeightCount);
        std::memcpy(raw_weights.data(), gEmbeddedProgressData, kRawWeightsBytes);
        load_weights_from_raw(raw_weights.data());
        sync_cout << "info string loading progress file : <internal>" << sync_endl;
        return true;
    }

    std::ifstream stream(file_path, std::ios::binary);
    if (!stream.is_open()) {
        sync_cout << "info string Failed to open the progress file. file_path=" << file_path << sync_endl;
        return false;
    }

    std::vector<double> raw_weights(kWeightCount);
    stream.read(reinterpret_cast<char*>(raw_weights.data()), kRawWeightsBytes);
    if (!stream) {
        sync_cout << "info string Failed to read the progress file. file_path=" << file_path << sync_endl;
        return false;
    }

    load_weights_from_raw(raw_weights.data());
    sync_cout << "info string loading progress file : " << file_path << sync_endl;
    return true;
}

int LayerStackIndex(const YaneuraOu::Position& pos) {
    const auto sq_bk = pos.square<YaneuraOu::KING>(YaneuraOu::BLACK);
    const auto sq_wk = YaneuraOu::Inv(pos.square<YaneuraOu::KING>(YaneuraOu::WHITE));

    int32_t sum_q16 = 0;
    if (!try_get_sum_from_cache(pos, sq_bk, sq_wk, sum_q16)) {
        sum_q16 = compute_full_sum_q16(pos, sq_bk, sq_wk);
        store_sum_cache(pos, sq_bk, sq_wk, sum_q16);
    }

    int idx = table_index_linear_q16(sum_q16);
    return idx;
}

}  // namespace Progress
}  // namespace Tanuki
