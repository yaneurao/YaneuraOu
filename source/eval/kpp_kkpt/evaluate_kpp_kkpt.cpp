#include "../../config.h"

// KPP+KKPTの実験用コード。
// ほとんどevaluate_kppt.cppと同じ。

#if defined (EVAL_KPP_KKPT)

#include <fstream>
#include <iostream>
#include <unordered_set>

#include "../../evaluate.h"
#include "../../position.h"
#include "../../misc.h"
#include "../../memory.h"
#include "../../shm.h"
#include "../../usi.h"
#include "../../extra/bitop.h"
#include "../evaluate_io.h"
#include "evaluate_kpp_kkpt.h"

#if defined (USE_EVAL_HASH)
#include "../evalhash.h"
#endif

using namespace std;

// ============================================================
//              旧評価関数のためのヘルパー
// ============================================================

#if defined(USE_CLASSIC_EVAL)
using namespace YaneuraOu;
void add_options_(OptionsMap& options, ThreadPool& threads);

namespace {
YaneuraOu::OptionsMap* options_ptr;
YaneuraOu::ThreadPool* threads_ptr;
}

// 📌 旧Options、旧Threadsとの互換性のための共通のマクロ 📌
#define Options (*options_ptr)
#define Threads (*threads_ptr)

namespace YaneuraOu::Eval {
void add_options(OptionsMap& options, ThreadPool& threads) {
    options_ptr = &options;
    threads_ptr = &threads;
    add_options_(options, threads);
}
}
// ============================================================

// 評価関数を読み込み済みであるか
bool        eval_loaded   = false;
std::string last_eval_dir = "None";

// 📌 この評価関数で追加したいエンジンオプションはここで追加する。
void add_options_(OptionsMap& options, ThreadPool& threads) {

    const char* default_eval_dir = "eval";
    Options.add("EvalDir", Option(default_eval_dir, [](const Option& o) {
                    std::string eval_dir = std::string(o);
                    if (last_eval_dir != eval_dir)
                    {
                        // 評価関数フォルダ名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
                        last_eval_dir = eval_dir;
                        eval_loaded   = false;
                    }
                    return std::nullopt;
                }));

}
#endif


namespace YaneuraOu {
namespace Eval {

	struct KppKkptEvalParameters {
		// The eval files overwrite every byte, so avoid zero-filling this huge object.
		KppKkptEvalParameters() noexcept {}

		ValueKk  kk_body[SQ_NB][SQ_NB];
		ValueKkp kkp_body[SQ_NB][SQ_NB][fe_end];
		ValueKpp kpp_body[SQ_NB][fe_end][fe_end];
	};

	static_assert(std::is_trivially_copyable_v<KppKkptEvalParameters>,
		"KppKkptEvalParameters must be trivially copyable for shared memory support");
	static_assert(sizeof(KppKkptEvalParameters) == size_of_eval,
		"KppKkptEvalParameters layout must match the existing contiguous eval layout");

	// 評価関数パラメーター
	// 2GBを超える配列は確保できないようなのでポインターにしておき、動的に確保する。

	ValueKk(*kk_)[SQ_NB][SQ_NB];
	ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];

	SystemWideSharedConstant<KppKkptEvalParameters> shared_eval_parameters;
	constexpr std::size_t KppKkptSharedMemoryDiscriminator = 0x4b504b54u; // "KPKT"

} // namespace Eval
} // namespace YaneuraOu

template<>
struct std::hash<YaneuraOu::Eval::KppKkptEvalParameters> {
	std::size_t operator()(const YaneuraOu::Eval::KppKkptEvalParameters& p) const noexcept {
		return static_cast<std::size_t>(
			YaneuraOu::hash_bytes(reinterpret_cast<const char*>(&p), sizeof(p)));
	}
};

namespace YaneuraOu {
namespace Eval {

	// 評価関数ファイルを読み込む
	void load_eval_impl()
	{
		// EvalIOを利用して評価関数ファイルを読み込む。
		// ちなみに、inputのところにあるbasic_kppt32()をbasic_kppt16()に変更するとApery(WCSC27)の評価関数ファイルが読み込める。
		// また、eval_convert()に渡している引数のinputとoutputを入れ替えるとファイルに書き出すことが出来る。EvalIOマジ、っょぃ。
        auto make_name = [&](std::string filename) {
            auto eval_dir      = Options["EvalDir"];
            auto abs_eval_path = Path::Combine(Directory::GetBinaryFolder(), eval_dir);
            return Path::Combine(abs_eval_path, filename);
        };

		auto input = EvalIO::EvalInfo::build_kpp_kkpt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN));
		auto output = EvalIO::EvalInfo::build_kpp_kkpt32((void*)kk, (void*)kkp, (void*)kpp);

		// 評価関数の実験のためにfe_endをbuild_kpp_ppt()の値から変更しているかも知れないので現在のfe_endの値をもとに読み込む。
		input.fe_end = output.fe_end = Eval::fe_end;

		if (!EvalIO::eval_convert(input, output, nullptr))
			goto Error;

		// ここに実験用のコードなどを書くかも。

		// 読み込みは成功した。
		return;

	Error:;
		// 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
		sync_cout << "\ninfo string Error! open evaluation file failed.\n" << sync_endl;
		Tools::exit();
	}


	u64 calc_check_sum()
	{
		u64 sum = 0;

		auto add_sum = [&](u16*ptr, size_t t)
		{
			for (size_t i = 0; i < t; ++i)
				sum += ptr[i];
		};

		// sizeof演算子、2GB以上の配列に対して機能しない。VC++でC2070になる。
		// そのため、sizeof(kpp)のようにせず、自前で計算している。

		// データは2 or 4バイトなので、endiannessがどちらであっても
		// これでcheck sumの値は変わらない。
		// また、データが2 or 4バイトなので2バイトずつ加算していくとき、
		// データの余りは出ない。

		add_sum(reinterpret_cast<u16*>(kk), size_of_kk / sizeof(u16));
		add_sum(reinterpret_cast<u16*>(kkp), size_of_kkp / sizeof(u16));
		add_sum(reinterpret_cast<u16*>(kpp), size_of_kpp / sizeof(u16));

		return sum;
	}

	void init(){}

	// 与えられたsize_of_evalサイズの連続したalign 32されているメモリに、kk_,kkp_,kpp_を割り当てる。
	void eval_assign(void* ptr)
	{
		s8* p = (s8*)ptr;
		kk_  = (ValueKk (*)[SQ_NB][SQ_NB]) (p);
		kkp_ = (ValueKkp(*)[SQ_NB][SQ_NB][fe_end]) (p + size_of_kk);
		kpp_ = (ValueKpp(*)[SQ_NB][fe_end][fe_end]) (p + size_of_kk + size_of_kkp);
	}

	void load_eval()
	{
        if (eval_loaded)
            return;
        eval_loaded = true;  // 📌 読み込みに失敗したらプロセスが終了するだろうから..

		auto tmp = make_unique_large_page<KppKkptEvalParameters>();
		eval_assign(tmp.get());
		load_eval_impl();

		shared_eval_parameters =
			SystemWideSharedConstant<KppKkptEvalParameters>(*tmp, KppKkptSharedMemoryDiscriminator);

		if (shared_eval_parameters == nullptr)
		{
			sync_cout << "info string KPP_KKPT shared memory: no eval memory allocated";
			if (auto message = shared_eval_parameters.get_error_message())
				sync_cout << " (" << *message << ")";
			sync_cout << sync_endl;
			Tools::exit();
		}

		eval_assign(const_cast<KppKkptEvalParameters*>(&*shared_eval_parameters));
	}

	// KP,KPP,KKPのスケール
	const int FV_SCALE = 32;

	// 評価関数。全計算。(駒割りは差分)
	// 返し値は持たず、計算結果としてpos.state()->sumに値を代入する。
	void compute_eval_impl(const Position& pos)
	{
		// is_ready()で評価関数を読み込み、
		// 初期化してからしかcompute_eval()を呼び出すことは出来ない。
		ASSERT_LV1(&(kk) != nullptr);
		// →　32bit環境だとこの変数、単なるポインタなのでこのassertは意味がないのだが、
		// とりあえず開発時に早期に気づくようにこのassertを入れておく。

		Square sq_bk = pos.square<KING>(BLACK);
		Square sq_wk = pos.square<KING>(WHITE);
		const auto* ppkppb = kpp[sq_bk];
		const auto* ppkppw = kpp[Inv(sq_wk)];

		auto& pos_ = *const_cast<Position*>(&pos);
		const int length = pos_.eval_list()->length();

#if !defined (USE_EVAL_MAKE_LIST_FUNCTION)

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

#else
		// -----------------------------------
		// USE_EVAL_MAKE_LIST_FUNCTIONが定義されているときは
		// ここでeval_listをコピーして、組み替える。
		// -----------------------------------

		// バッファを確保してコピー
		BonaPiece list_fb[40];
		BonaPiece list_fw[40];
		memcpy(list_fb, pos_.eval_list()->piece_list_fb(), sizeof(BonaPiece) * 40);
		memcpy(list_fw, pos_.eval_list()->piece_list_fw(), sizeof(BonaPiece) * 40);

		// ユーザーは、この関数でBonaPiece番号の自由な組み換えを行なうものとする。
		make_list_function(pos, list_fb, list_fw);

#endif

		int i, j;
		BonaPiece k0, k1, l0, l1;

		// 評価値の合計
		EvalSum sum;

#if defined(USE_SSE2)
		// sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
		sum.m[0] = _mm_setzero_si128();
#else
		sum.p[0][0] = /*sum.p[0][1] =*/ sum.p[1][0] = /*sum.p[1][1] =*/ 0;
#endif

		// KK
		sum.p[2] = kk[sq_bk][sq_wk];

		for (i = 0; i < length; ++i)
		{
			k0 = list_fb[i];
			k1 = list_fw[i];
			const auto* pkppb = ppkppb[k0];
			const auto* pkppw = ppkppw[k1];
			for (j = 0; j < i; ++j)
			{
				l0 = list_fb[j];
				l1 = list_fw[j];

				// KPP
				sum.p[0][0] += pkppb[l0];
				sum.p[1][0] += pkppw[l1];
			}
			// KKP
			sum.p[2] += kkp[sq_bk][sq_wk][k0];
		}

		auto st = pos.state();
		sum.p[2][0] += st->materialValue * FV_SCALE;

		st->sum = sum;
	}

	// 評価関数。差分計算ではなく全計算する。
	// Position::set()で一度だけ呼び出される。(以降は差分計算)
	// 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
	// なので、この関数の最適化は頑張らない。
	Value compute_eval(const Position& pos)
	{
		compute_eval_impl(pos);
		return Value(pos.state()->sum.sum(pos.side_to_move()) / FV_SCALE);
	}

	// 後手玉が移動したときの先手玉に対するの差分
	s32 do_a_black(const Position& pos, const ExtBonaPiece ebp) {
		const Square sq_bk = pos.square<KING>(BLACK);
		const auto* list0 = pos.eval_list()->piece_list_fb();
		const int length = pos.eval_list()->length();

		const auto* pkppb = kpp[sq_bk][ebp.fb];

		s32 sum = pkppb[list0[0]];
		for (int i = 1; i < length; ++i)
			sum += pkppb[list0[i]];

		return sum;
	}

	// 先手玉が移動したときの後手玉に対する差分
	s32 do_a_white(const Position& pos, const ExtBonaPiece ebp) {
		const Square sq_wk = pos.square<KING>(WHITE);
		const auto* list1 = pos.eval_list()->piece_list_fw();
		const int length = pos.eval_list()->length();

		const auto* pkppw = kpp[Inv(sq_wk)][ebp.fw];

		s32 sum = pkppw[list1[0]];
		for (int i = 1; i < length ; ++i)
			sum += pkppw[list1[i]];

		return sum;
	}

	// 玉以外の駒が移動したときの差分
	EvalSum do_a_pc(const Position& pos, const ExtBonaPiece ebp) {
		/*
		移動した駒がm駒あるなら、これらの駒をeval_list()[]のn-1,n-2,…,n-mに移動させて、
		for(i=1..m)
		do_a_black(pos,n-i)
		みたいなことをすべきだが、mはたかだか2なので、
		こうはせずに、引きすぎた重複分(kpp[k][n-1][n-2])をあとで加算している。
		*/
		const Square sq_bk = pos.square<KING>(BLACK);
		const Square sq_wk = pos.square<KING>(WHITE);
		const auto list0 = pos.eval_list()->piece_list_fb();
		const auto list1 = pos.eval_list()->piece_list_fw();
		const int length = pos.eval_list()->length();

		EvalSum sum;

		// sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
#if defined(USE_SSE2)
		sum.m[0] = _mm_setzero_si128();
#else
		sum.p[0][0] = 0;
		sum.p[1][0] = 0;
#endif

		// KK
		sum.p[2] = kkp[sq_bk][sq_wk][ebp.fb];

		const auto* pkppb = kpp[sq_bk][ebp.fb];
		const auto* pkppw = kpp[Inv(sq_wk)][ebp.fw];

        // AVX2化前
        // Nodes/second    : 874819
        // Nodes/second    : 870346
        // Nodes/second    : 861860
        // Function	Samples	% of Hotspot Samples	Module
        // Eval::do_a_pc(struct Position const &, struct Eval::ExtBonaPiece)	6509	14.6000004	YaneuraOu - 2017 - early.exe
        // Eval::evaluateBody(struct Position const &)	6201	13.9099998	YaneuraOu - 2017 - early.exe

        // AVX2化後 (VGATHERDDあり)
        // Nodes/second    : 920074
        // Nodes/second    : 927817
        // Nodes/second    : 923094
        // Function	Samples	% of Hotspot Samples	Module
        // Eval::evaluateBody(struct Position const &)	5169	12.3500004	YaneuraOu - 2017 - early.exe
        // Eval::do_a_pc(struct Position const &, struct Eval::ExtBonaPiece)	5100	12.1800003	YaneuraOu - 2017 - early.exe

        // NPSが約6.1%程度向上した

#if defined(USE_AVX2)
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        int i = 0;
        for (; i + 8 < length; i += 8) {
            __m256i index0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(list0 + i));
            __m256i w0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppb), index0, 2);
            w0 = _mm256_slli_epi32(w0, 16);
            w0 = _mm256_srai_epi32(w0, 16);
            sum0 = _mm256_add_epi32(sum0, w0);

            __m256i index1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(list1 + i));
            __m256i w1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppw), index1, 2);
            w1 = _mm256_slli_epi32(w1, 16);
            w1 = _mm256_srai_epi32(w1, 16);
            sum1 = _mm256_add_epi32(sum1, w1);
        }

        // 端数の6要素分の処理
		// ※　ここ38固定の処理になっている。
		// 　ここを可変の処理にするには、evaluate_kkpp_kkpt.cppのようにすべき。
		//　 いまのところメリットがないのでここは変更しないでおく。
		{
            __m256i w0 = _mm256_set_epi32(
                0,
                0,
                pkppb[list0[i + 5]],
                pkppb[list0[i + 4]],
                pkppb[list0[i + 3]],
                pkppb[list0[i + 2]],
                pkppb[list0[i + 1]],
                pkppb[list0[i + 0]]);
            __m256i w1 = _mm256_set_epi32(
                0,
                0,
                pkppw[list1[i + 5]],
                pkppw[list1[i + 4]],
                pkppw[list1[i + 3]],
                pkppw[list1[i + 2]],
                pkppw[list1[i + 1]],
                pkppw[list1[i + 0]]);
            sum0 = _mm256_add_epi32(sum0, w0);
            sum1 = _mm256_add_epi32(sum1, w1);
        }

        // _mm256_srli_si256()は128ビット境界毎にシフトされる点に注意する
        sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
        sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
        sum.p[0][0] = _mm_extract_epi32(_mm256_extracti128_si256(sum0, 0), 0) +
            _mm_extract_epi32(_mm256_extracti128_si256(sum0, 1), 0);
        
        sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
        sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 4));
        sum.p[1][0] = _mm_extract_epi32(_mm256_extracti128_si256(sum1, 0), 0) +
            _mm_extract_epi32(_mm256_extracti128_si256(sum1, 1), 0);
#else
        sum.p[0][0] = pkppb[list0[0]];
        sum.p[1][0] = pkppw[list1[0]];
        for (int i = 1; i < length; ++i) {
            sum.p[0][0] += pkppb[list0[i]];
            sum.p[1][0] += pkppw[list1[i]];
        }
#endif

        return sum;
    }


#if defined (USE_EVAL_HASH)
	// evaluateしたものを保存しておくHashTable(俗にいうehash)

	struct EvaluateHashTable : HashTable<EvalSum> {};
	EvaluateHashTable g_evalTable;

	void EvalHash_Resize(size_t mbSize) { g_evalTable.resize(Threads, mbSize); }
	void EvalHash_Clear() { g_evalTable.clear(Threads); };

	// prefetchする関数も用意しておく。
	void prefetch_evalhash(const Key key)
	{
		prefetch(g_evalTable[key]);
	}

#endif

#if !defined(USE_EVAL_MAKE_LIST_FUNCTION)
	void evaluateBody(const Position& pos)
	{
		// 一つ前のノードからの評価値の差分を計算する。

		auto now = pos.state();
		auto prev = now->previous;

		// nodeごとにevaluate()は呼び出しているので絶対に差分計算できるはず。
		// 一つ前のnodeでevaluate()されているはず。
		//
		// root nodeではprevious == nullptrであるが、root nodeではPosition::set()でcompute_eval()
		// を呼び出すので通常この関数が呼び出されることはない。
		if (!prev->sum.evaluated())
		{
			// 全計算
			compute_eval_impl(pos);

			return;
			// 結果は、pos->state().sumから取り出すべし。
		}

		// 遡るnodeは一つだけ
		// ひとつずつ遡りながらsumKPPがVALUE_NONEでないところまで探してそこからの差分を計算することは出来るが
		// 現状、探索部では毎node、evaluate()を呼び出すから問題ない。

		auto& dp = now->dirtyPiece;

		// 移動させた駒は最大2つある。その数
		int moved_piece_num = dp.dirty_num;

		auto list0 = pos.eval_list()->piece_list_fb();
		auto list1 = pos.eval_list()->piece_list_fw();
		auto length = pos.eval_list()->length();

		auto dirty = dp.pieceNo[0];

		// 移動させた駒は王か？
		if (dirty >= PIECE_NUMBER_KING)
		{
			// 前のnodeの評価値からの増分を計算していく。
			// (直接この変数に加算していく)
			// この意味においてdiffという名前は少々不適切ではあるが。
			EvalSum diff = prev->sum;

			auto sq_bk = pos.square<KING>(BLACK);
			auto sq_wk = pos.square<KING>(WHITE);

			// ΣKKPは最初から全計算するしかないので初期化する。
			diff.p[2] = kk[sq_bk][sq_wk];
			diff.p[2][0] += now->materialValue * FV_SCALE;

			// 後手玉の移動(片側分のKPPを丸ごと求める)
			if (dirty == PIECE_NUMBER_WKING)
			{
				const auto ppkppw = kpp[Inv(sq_wk)];

				// ΣWKPP = 0
				diff.p[1][0] = 0;
				//diff.p[1][1] = 0;

#if defined(USE_AVX2)
                __m256i sum1_256 = _mm256_setzero_si256();
                __m128i sum1_128 = _mm_setzero_si128();

                for (int i = 0; i < length ; ++i)
                {
					// KKPの値は、後手側から見た計算だとややこしいので、先手から見た計算でやる。
					// 後手から見た場合、kkp[inv(sq_wk)][inv(sq_bk)][k1]になるが、これ次元下げで同じ値を書いているとは限らない。
					diff.p[2] += kkp[sq_bk][sq_wk][list0[i]];
					
					const int k1 = list1[i];
                    const auto* pkppw = ppkppw[k1];
                    int j = 0;
                    for (; j + 8 < i; j += 8) {
                        __m256i index1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(list1 + j));
                        __m256i w1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppw), index1, 2);
                        w1 = _mm256_slli_epi32(w1, 16);
                        w1 = _mm256_srai_epi32(w1, 16);
                        sum1_256 = _mm256_add_epi32(sum1_256, w1);
                    }

                    for (; j + 4 < i; j += 4) {
                        __m128i index1 = _mm_load_si128(reinterpret_cast<const __m128i*>(list1 + j));
                        __m128i w1 = _mm_i32gather_epi32(reinterpret_cast<const int*>(pkppw), index1, 2);
                        w1 = _mm_slli_epi32(w1, 16);
                        w1 = _mm_srai_epi32(w1, 16);
                        sum1_128 = _mm_add_epi32(sum1_128, w1);
                    }

                    for (; j < i; ++j) {
                        diff.p[1][0] += pkppw[list1[j]];
                    }
                }
                sum1_128 = _mm_add_epi32(sum1_128, _mm256_extracti128_si256(sum1_256, 0));
                sum1_128 = _mm_add_epi32(sum1_128, _mm256_extracti128_si256(sum1_256, 1));
                sum1_128 = _mm_add_epi32(sum1_128, _mm_srli_si128(sum1_128, 8));
                sum1_128 = _mm_add_epi32(sum1_128, _mm_srli_si128(sum1_128, 4));
                diff.p[1][0] += _mm_extract_epi32(sum1_128, 0);
#else
				for (int i = 0; i < length ; ++i)
				{
					diff.p[2] += kkp[sq_bk][sq_wk][list0[i]];

					const int k1 = list1[i];
					const auto* pkppw = ppkppw[k1];
					for (int j = 0; j < i; ++j)
					{
						const int l1 = list1[j];
						diff.p[1][0] += pkppw[l1];
					}
				}
#endif

				// 動かした駒が２つ
				if (moved_piece_num == 2)
				{
					// 瞬間的にeval_listの移動させた駒の番号を変更してしまう。
					// こうすることで前nodeのpiece_listを持たなくて済む。

					const int listIndex_cap = dp.pieceNo[1];
					diff.p[0][0] += do_a_black(pos, dp.changed_piece[1].new_piece);
					list0[listIndex_cap] = dp.changed_piece[1].old_piece.fb;
					diff.p[0][0] -= do_a_black(pos, dp.changed_piece[1].old_piece);
					list0[listIndex_cap] = dp.changed_piece[1].new_piece.fb;
				}

			}
			else {

				// 先手玉の移動
				// さきほどの処理と同様。

				const auto* ppkppb = kpp[sq_bk];
				diff.p[0][0] = 0;
				//diff.p[0][1] = 0;

#if defined(USE_AVX2)
                __m256i sum0_256 = _mm256_setzero_si256();
                __m128i sum0_128 = _mm_setzero_si128();

                for (int i = 0; i < length ; ++i)
                {
                    const int k0 = list0[i];
                    const auto* pkppb = ppkppb[k0];
                    int j = 0;
                    for (; j + 8 < i; j += 8) {
                        __m256i index0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(list0 + j));
                        __m256i w0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppb), index0, 2);
                        w0 = _mm256_slli_epi32(w0, 16);
                        w0 = _mm256_srai_epi32(w0, 16);
                        sum0_256 = _mm256_add_epi32(sum0_256, w0);
                    }

                    for (; j + 4 < i; j += 4) {
                        __m128i index0 = _mm_load_si128(reinterpret_cast<const __m128i*>(list0 + j));
                        __m128i w0 = _mm_i32gather_epi32(reinterpret_cast<const int*>(pkppb), index0, 2);
                        w0 = _mm_slli_epi32(w0, 16);
                        w0 = _mm_srai_epi32(w0, 16);
                        sum0_128 = _mm_add_epi32(sum0_128, w0);
                    }

                    for (; j < i; ++j) {
                        diff.p[0][0] += pkppb[list0[j]];
                    }

                    diff.p[2] += kkp[sq_bk][sq_wk][k0];
                }
                sum0_128 = _mm_add_epi32(sum0_128, _mm256_extracti128_si256(sum0_256, 0));
                sum0_128 = _mm_add_epi32(sum0_128, _mm256_extracti128_si256(sum0_256, 1));
                sum0_128 = _mm_add_epi32(sum0_128, _mm_srli_si128(sum0_128, 8));
                sum0_128 = _mm_add_epi32(sum0_128, _mm_srli_si128(sum0_128, 4));
                diff.p[0][0] += _mm_extract_epi32(sum0_128, 0);
#else
                for (int i = 0; i < length ; ++i)
                {
                    const int k0 = list0[i];
                    const auto* pkppb = ppkppb[k0];
                    for (int j = 0; j < i; ++j) {
                        const int l0 = list0[j];
                        diff.p[0][0] += pkppb[l0];
                    }
                    diff.p[2] += kkp[sq_bk][sq_wk][k0];
                }
#endif

				if (moved_piece_num == 2) {
					const int listIndex_cap = dp.pieceNo[1];
					diff.p[1][0] += do_a_white(pos, dp.changed_piece[1].new_piece);
					list1[listIndex_cap] = dp.changed_piece[1].old_piece.fw;
					diff.p[1][0] -= do_a_white(pos, dp.changed_piece[1].old_piece);
					list1[listIndex_cap] = dp.changed_piece[1].new_piece.fw;
				}
			}

			// sumの計算が終わったのでpos.state()->sumに反映させておく。(これがこの関数の返し値に相当する。)
			now->sum = diff;

		}
		else {

			// 王以外の駒が移動したケース
			// 今回の差分を計算して、そこに加算する。

			const int listIndex = dp.pieceNo[0];

			auto diff = do_a_pc(pos, dp.changed_piece[0].new_piece);
			if (moved_piece_num == 1) {

				// 動いた駒が1つ。
				list0[listIndex] = dp.changed_piece[0].old_piece.fb;
				list1[listIndex] = dp.changed_piece[0].old_piece.fw;
				diff -= do_a_pc(pos, dp.changed_piece[0].old_piece);

			}
			else {

				// 動いた駒が2つ。

				auto sq_bk = pos.square<KING>(BLACK);
				auto sq_wk = pos.square<KING>(WHITE);

				diff += do_a_pc(pos, dp.changed_piece[1].new_piece);
				diff.p[0][0] -= kpp[    sq_bk ][dp.changed_piece[0].new_piece.fb][dp.changed_piece[1].new_piece.fb];
				diff.p[1][0] -= kpp[Inv(sq_wk)][dp.changed_piece[0].new_piece.fw][dp.changed_piece[1].new_piece.fw];

				const PieceNumber listIndex_cap = dp.pieceNo[1];
				list0[listIndex_cap] = dp.changed_piece[1].old_piece.fb;
				list1[listIndex_cap] = dp.changed_piece[1].old_piece.fw;

				list0[listIndex] = dp.changed_piece[0].old_piece.fb;
				list1[listIndex] = dp.changed_piece[0].old_piece.fw;
				diff -= do_a_pc(pos, dp.changed_piece[0].old_piece);
				diff -= do_a_pc(pos, dp.changed_piece[1].old_piece);

				diff.p[0][0] += kpp[    sq_bk ][dp.changed_piece[0].old_piece.fb][dp.changed_piece[1].old_piece.fb];
				diff.p[1][0] += kpp[Inv(sq_wk)][dp.changed_piece[0].old_piece.fw][dp.changed_piece[1].old_piece.fw];
				list0[listIndex_cap] = dp.changed_piece[1].new_piece.fb;
				list1[listIndex_cap] = dp.changed_piece[1].new_piece.fw;
			}

			list0[listIndex] = dp.changed_piece[0].new_piece.fb;
			list1[listIndex] = dp.changed_piece[0].new_piece.fw;

			// 前nodeからの駒割りの増分を加算。
			diff.p[2][0] += (now->materialValue - prev->materialValue) * FV_SCALE;

			now->sum = diff + prev->sum;
		}
		
	}
#else
	// EvalListの組み換えを行なうときは差分計算をせずに(実装するのが大変なため)、毎回全計算を行なう。
	void evaluateBody(const Position& pos)
	{
		compute_eval_impl(pos);
	}
#endif // USE_EVAL_MAKE_LIST_FUNCTION

	// 評価関数
	Value evaluate(const Position& pos)
	{
		auto st = pos.state();
		auto &sum = st->sum;

		// すでに計算済(Null Moveなどで)であるなら、それを返す。
		if (sum.evaluated())
			return Value(sum.sum(pos.side_to_move()) / FV_SCALE);
		// ここで未初期化な値が返っているなら、それはPosition::do_move()のところでVALUE_NOT_EVALUATEDを代入していないからだ。

#if defined(USE_GLOBAL_OPTIONS)
		// GlobalOptionsでeval hashを用いない設定になっているなら
		// eval hashへの照会をskipする。
		if (!GlobalOptions.use_eval_hash)
		{
			evaluateBody(pos);
			ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
			return Value(sum.sum(pos.side_to_move()) / FV_SCALE);
		}
#endif

#if defined ( USE_EVAL_HASH )
		// 局面のhash key
		const Key key = pos.key();

		// evaluate hash tableにはあるかも。

		//		cout << "EvalSum " << hex << g_evalTable[key] << endl;
		EvalSum entry = *g_evalTable[key];   // atomic にデータを取得する必要がある。
		entry.decode();
		if (entry.key == key)
		{
			//	dbg_hit_on(true);

			// あった！
			sum = entry;
			return Value(entry.sum(pos.side_to_move()) / FV_SCALE);
		}
		//		dbg_hit_on(false);

#endif

		// 評価関数本体を呼び出して求める。
		evaluateBody(pos);

#if defined(USE_EVAL_HASH)
		// せっかく計算したのでevaluate hash tableに保存しておく。
		sum.key = key;
		sum.encode();
		*g_evalTable[key] = sum;
#endif

		ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
		// 差分計算と非差分計算との計算結果が合致するかのテスト。(さすがに重いのでコメントアウトしておく)
		// ASSERT_LV5(Value(st->sum.sum(pos.side_to_move()) / FV_SCALE) == compute_eval(pos));

#if 0
		if (!(Value(st->sum.sum(pos.side_to_move()) / FV_SCALE) == compute_eval(pos)))
		{
			st->sum.p[0][0] = VALUE_NOT_EVALUATED;
			evaluateBody(pos);
		}
#endif

		auto v = Value(sum.sum(pos.side_to_move()) / FV_SCALE);

		// 返す値の絶対値がVALUE_MAX_EVALを超えてないことを保証しないといけないのだが…。
		// いまの評価関数、手番を過学習したりして、ときどき超えてそう…。
		//ASSERT_LV3(abs(v) < =VALUE_MAX_EVAL);

		return v;
	}

	void evaluate_with_no_return(const Position& pos)
	{
		// 評価関数の実験のときには、どうせ差分計算を行わないので、
		// ここでevaluate()を呼ぶのは無駄である。

#if !defined(USE_EVAL_MAKE_LIST_FUNCTION)
		// まだ評価値が計算されていないなら
		if (!pos.state()->sum.evaluated())
			evaluate(pos);
#else
		// EvalListの組み換えを行なっているときは通常の差分計算ルーチンが機能しないので
		// 差分計算をするための何かをする必要がない。
#endif
	}


	// 現在の局面の評価値の内訳を表示する。
	void print_eval_stat(Position& pos)
	{
		cout << "--- EVAL STAT\n";

		Square sq_bk = pos.square<KING>(BLACK);
		Square sq_wk = pos.square<KING>(WHITE);
		const auto* ppkppb = kpp[sq_bk];
		const auto* ppkppw = kpp[Inv(sq_wk)];

		auto& pos_ = *const_cast<Position*>(&pos);

		auto list_fb = pos_.eval_list()->piece_list_fb();
		auto list_fw = pos_.eval_list()->piece_list_fw();

		int i, j;
		BonaPiece k0, k1, l0, l1;

		// 38枚の駒を表示
		for (i = 0; i < PIECE_NUMBER_KING; ++i)
			cout << int(list_fb[i]) << " = " << list_fb[i] << " , " << int(list_fw[i]) << " =  " << list_fw[i] << endl;

		// 評価値の合計
		EvalSum sum;

#if defined(USE_SSE2)
		// sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
		sum.m[0] = _mm_setzero_si128();
#else
		sum.p[0][0] = /*sum.p[0][1] =*/ sum.p[1][0] = /*sum.p[1][1] =*/ 0;
#endif

		// KK
		sum.p[2] = kk[sq_bk][sq_wk];
		cout << "KKC : " << sq_bk << " " << sq_wk << " = " << kk[sq_bk][sq_wk][0] << " + " << kk[sq_bk][sq_wk][1] << endl;

		for (i = 0; i < PIECE_NUMBER_KING; ++i)
		{
			k0 = list_fb[i];
			k1 = list_fw[i];
			const auto* pkppb = ppkppb[k0];
			const auto* pkppw = ppkppw[k1];
			for (j = 0; j < i; ++j)
			{
				l0 = list_fb[j];
				l1 = list_fw[j];

				cout << "BKPP : " << sq_bk << " " << k0 << " " << l0 << " = " << pkppb[l0] << endl;
				cout << "WKPP : " << sq_wk << " " << k1 << " " << l1 << " = " << pkppw[l1] << endl;

				sum.p[0][0] += pkppb[l0];
				sum.p[1][0] += pkppw[l1];
			}
			sum.p[2] += kkp[sq_bk][sq_wk][k0];

			cout << "KKP : " << sq_bk << " " << sq_wk << " " << k0 << " = " << kkp[sq_bk][sq_wk][k0][0] << " + " << kkp[sq_bk][sq_wk][k0][1] << endl;

		}

		cout << "Material = " << pos.state()->materialValue << endl;
		cout << sum;
		cout << "---" << endl;

		// KKのKの値を出力する実験的コード
		//		kk_stat();

	}

	// 評価関数のそれぞれのパラメーターに対して関数fを適用してくれるoperator。
	// パラメーターの分析などに用いる。
	void foreach_eval_param(std::function<void(s32, s32)>f, int type)
	{
		// KK
		if (type == -1 || type == 0)
		{
			for (u64 i = 0; i < (u64)SQ_NB * (u64)SQ_NB; ++i)
			{
				auto v = ((ValueKk*)kk)[i];
				f(v[0], v[1]);

				//if (v[0] == 0) cout << "v[0]==0" << (Square)(i / SQ_NB) << (Square)(i % SQ_NB) << endl;
				//if (v[1] == 0) cout << "v[1]==0" << (Square)(i / SQ_NB) << (Square)(i % SQ_NB) << endl;
			}
		}

		// KKP
		if (type == -1 || type == 1)
		{
			for (u64 i = 0; i < (u64)SQ_NB * (u64)SQ_NB * (u64)fe_end; ++i)
			{
				auto v = ((ValueKkp*)kkp)[i];
				f(v[0], v[1]);
			}
		}

		// KPP
		if (type == -1 || type == 2)
		{
			for (u64 i = 0; i < (u64)SQ_NB * (u64)fe_end * (u64)fe_end; ++i)
			{
				auto v = ((ValueKpp*)kpp)[i];
				f(v, 0); /* 手番なしなのでダミーで0を渡しておく */
			}
		}
	}

} // namespace Eval
} // namespace YaneuraOu

#endif // defined (EVAL_KPP_KKPT)
