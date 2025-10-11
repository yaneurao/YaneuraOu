﻿#include "../../config.h"

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
#include "../../usi.h"
#include "../../extra/bitop.h"
#include "../evaluate_io.h"
#include "evaluate_kpp_kkpt.h"

#if defined (USE_EVAL_HASH)
#include "../evalhash.h"
#endif

// EvalShareの機能を使うために必要
#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32)
#include <windows.h>
#endif

#if defined(EVAL_LEARN)
#include "../../learn/learning_tools.h"
using namespace YaneuraOu::EvalLearningTools;
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

#if defined(EVAL_LEARN)
    // isreadyタイミングで評価関数を読み込まれると、新しい評価関数の変換のために
    // test evalconvertコマンドを叩きたいのに、その新しい評価関数がないがために
    // このコマンドの実行前に異常終了してしまう。
    // そこでこの隠しオプションでisready時の評価関数の読み込みを抑制して、
    // test evalconvertコマンドを叩く。
    Options("SkipLoadingEval", Option(false));
#endif

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

    Options.add("EvalShare", Option(true));
}
#endif


namespace YaneuraOu {
namespace Eval {

	// 評価関数パラメーター
	// 2GBを超える配列は確保できないようなのでポインターにしておき、動的に確保する。

	ValueKk(*kk_)[SQ_NB][SQ_NB];
	ValueKkp(*kkp_)[SQ_NB][SQ_NB][fe_end];
	ValueKpp(*kpp_)[SQ_NB][fe_end][fe_end];

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

	// 評価関数テーブルの読み込み用のメモリ
	void* eval_memory;

	void eval_malloc()
	{
		// benchコマンドなどでOptionsを保存して復元するのでこのときEvalDirが変更されたことになって、
		// 評価関数の再読込の必要があるというフラグを立てるため、この関数は2度呼び出されることがある。

		// メモリ確保は一回にして、連続性のある確保にする。
		aligned_large_pages_free(eval_memory);
		eval_memory = aligned_large_pages_alloc(size_of_eval);
		eval_assign(eval_memory);
	}

#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32)
	// 評価関数の共有を行うための大掛かりな仕組み
	// gccでコンパイルするときもWindows環境であれば、これが有効になって欲しいので defined(_WIN32) で判定。

	void load_eval()
	{
		// 評価関数を共有するのか
		if (!(bool)Options["EvalShare"])
		{
			eval_malloc();
			load_eval_impl();

			// 共有されていないメモリを用いる。
			sync_cout << "info string use non-shared eval_memory." << sync_endl;

			return;
		}

		// 評価関数ファイルが格納されているDirectory名をfull pathにて取得。
		// それをMutex名にしておく。つまり同一フォルダの評価関数ファイルを参照している場合に限り、EvalShareで共有される。

		// カレントフォルダに".."みたいなフォルダ駆け上がりが含まれていて、絶対pathは同じなのに同じ文字列にならないかも知れない。
		// それはPath::Combine()が正規化して欲しい気はするが…面倒なのでやってない。

		auto dir_name = Path::Combine(Directory::GetBinaryFolder(), (std::string)Options["EvalDir"]);
		sync_cout << "info string EvalDirectory = " << dir_name << sync_endl;

		// Mutex名,MMF(Memory Mapped File)名にbackslash文字は使えないらしいので、escapeする。念のため'/'もescapeする。
		// (フォルダの絶対pathが同じなのに"/"と"\"とで合致しないと嫌なため)
		replace(dir_name.begin(), dir_name.end(), '\\', '_');
		replace(dir_name.begin(), dir_name.end(), '/', '_');
		// フォルダ記号を"_"に置換しているので、たまたまpathに"_"が含まれているとややこしいことになるが、
		// まあそんな運用普通しないと思うので気にしないことにする。

		// Visual Studio 2019で「デバッグなしで実行」をしたとき、2回に一回ぐらい、shared memoryが使われない。
		// VSのデバッガーが何か悪さをしているくさい。

		// wchar_t*が必要なので変換する。

		auto w_dir = Tools::MultiByteToWideChar(dir_name);

// wstring化マクロ
#define WIDEN(x) L##x
#define TO_WSTRING(x) WIDEN(#x)

		// Mutex名、MAX_PATH(==260)文字までなので、w_dir自体があまり深い階層だとこの制限を上回ってしまうが…。
		// これは仕様だとする。PATH名が230文字超えるようなところに評価関数ファイル配置しないで。(´ω｀)
		auto mapped_file_name = TEXT("YANEURAOU_KPP_KPPT_MMF"  ) + std::wstring(TO_WSTRING(ENGINE_VERSION)) + w_dir;
		auto mutex_name       = TEXT("YANEURAOU_KPP_KPPT_MUTEX") + std::wstring(TO_WSTRING(ENGINE_VERSION)) + w_dir;


		// プロセス間の排他用mutex
		auto hMutex = CreateMutex(NULL, FALSE, mutex_name.c_str());

		// ファイルマッピングオブジェクトの処理をプロセス間で排他したい。
		WaitForSingleObject(hMutex, INFINITE);
		{

			// ファイルマッピングオブジェクトの作成
			auto hMap = CreateFileMapping(INVALID_HANDLE_VALUE,
				NULL,
				PAGE_READWRITE, // | /**SEC_COMMIT/**/ /*SEC_RESERVE/**/,
				(u32)(size_of_eval >> 32), (u32)size_of_eval,
				mapped_file_name.c_str());

			bool already_exists = (GetLastError() == ERROR_ALREADY_EXISTS);

			// ビュー
			auto shared_eval_ptr = (void *)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size_of_eval);

			// メモリが確保できないときはshared_eval_ptr == null。このチェックをしたほうがいいような..。
			if (shared_eval_ptr == nullptr)
			{
				sync_cout << "info string can't allocate shared eval memory." << sync_endl;
				Tools::exit();
			}
			else
			{
				// shared_eval_ptrは、32bytesにalignされていると仮定している。
				// Windows環境ではそうなっているっぽいし、このコードはWindows環境専用なので
				// とりあえず、良しとする。
				ASSERT_LV1(((u64)shared_eval_ptr & 0x1f) == 0);

				eval_assign(shared_eval_ptr);

				if (!already_exists)
				{
					// 新規作成されてしまった

					// このタイミングで評価関数バイナリを読み込む
					load_eval_impl();

					sync_cout << "info string created shared eval memory." << sync_endl;

				}
				else {

					// 評価関数バイナリを読み込む必要はない。ファイルマッピングが成功した時点で
					// 評価関数バイナリは他のプロセスによって読み込まれていると考えられる。

					sync_cout << "info string use shared eval memory." << sync_endl;
				}
			}
		}
		ReleaseMutex(hMutex);

		// 終了時に本当ならば
		// 1) ::ReleaseMutex()
		// 2) ::UnmapVieOfFile()
		// が必要であるが、1),2)がプロセスが解体されるときに自動でなされるので、この処理は特に入れない。
	}

#else

	// 評価関数のプロセス間共有を行わないときは、普通に
	// load_eval_impl()を呼び出すだけで良い。
	void load_eval()
	{
        if (eval_loaded)
            return;
        eval_loaded = true;  // 📌 読み込みに失敗したらプロセスが終了するだろうから..


		eval_malloc();
		load_eval_impl();
	}

#endif

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
		// を呼び出すので通常この関数が呼び出されることはないのだが、学習関係でこれが出来ないと
		// コードが書きにくいのでEVAL_LEARNのときは、このチェックをする。
		if (
#if defined (EVAL_LEARN)
			prev == nullptr ||
#endif
			!prev->sum.evaluated())
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

#if defined(EVAL_LEARN)
	// KKのKの値を出力する実験的コード
	void kk_stat()
	{
		EvalLearningTools::init();

		auto for_all_sq = [](std::function<void(Square)> func) {
			for (int r = RANK_1; r <= RANK_9; ++r)
			{
				for (int f = FILE_1; f <= FILE_9; ++f)
				{
					auto sq = (File)f | (Rank)r;
					func(sq);
				}
				cout << endl;
			}
			cout << endl;
		};

		// 先手から。
		cout << "BK = " << endl;
		for_all_sq([](Square sq) {
			array<float, 2> sum_kk = { 0,0 };
			array<float, 2> sum_kkp = { 0,0 };
			array<float, 2> sum_kpp = { 0,0 };
			for (auto sq2 = 0; sq2 < SQ_NB; ++sq2)
			{
				sum_kk += kk[sq][sq2];
				for (auto p = 0; p < fe_end; ++p)
					sum_kkp += kkp[sq][sq2][p];
			}
			for (auto p1 = 0; p1 < fe_end; ++p1)
				for (auto p2 = 0; p2 < fe_end; ++p2)
					sum_kpp[0] += kpp[sq][p1][p2];

			for (int i = 0; i < 2; ++i)
			{
				sum_kk[i] /= SQ_NB;
				sum_kkp[i] = 38 * sum_kkp[i] / (fe_end * (int)SQ_NB);
				sum_kpp[i] = (38 * 37 / 2) * sum_kpp[i] / (fe_end * (int)fe_end);
			}
			cout << "{" << (int)sum_kk[0] << ":" << (int)sum_kkp[0] << ":" << (int)sum_kpp[0] << ","
				<< (int)sum_kk[1] << ":" << (int)sum_kkp[1] << ":" << (int)sum_kpp[1] << "} ";
		});

		// 後手から。
		cout << "WK = " << endl;
		for_all_sq([](Square sq) {
			array<float, 2> sum_kk = { 0,0 };
			array<float, 2> sum_kkp = { 0,0 };
			array<float, 2> sum_kpp = { 0,0 };
			for (Square sq2 = SQ_ZERO; sq2 < SQ_NB; ++sq2)
			{
				sum_kk += kk[sq2][sq];
				for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
					sum_kkp += kkp[sq2][sq][p];
			}
			for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
				{
					// kpp、invしたときも、手番は先手から見た値なので符号逆にしない
					sum_kpp[0] -= kpp[Inv(sq)][inv_piece(p1)][inv_piece(p2)];

					//sum_kpp[0] -= kpp[Inv(sq)][inv_piece(p1)][inv_piece(p2)][0];
					//sum_kpp[1] += kpp[Inv(sq)][inv_piece(p1)][inv_piece(p2)][1];
				}

			for (int i = 0; i < 2; ++i)
			{
				sum_kk[i] /= SQ_NB;
				sum_kkp[i] = 38 * sum_kkp[i] / (fe_end * (int)SQ_NB);
				sum_kpp[i] = (38 * 37 / 2) * sum_kpp[i] / (fe_end * (int)fe_end);
			}
			cout << "{" << (int)sum_kk[0] << ":" << (int)sum_kkp[0] << ":" << (int)sum_kpp[0] << ","
				        << (int)sum_kk[1] << ":" << (int)sum_kkp[1] << ":" << (int)sum_kpp[1] << "} ";
		});
	}
#endif

	// 現在の局面の評価値の内訳を表示する。
	void print_eval_stat(Position& pos)
	{
		cout << "--- EVAL STAT\n";

		Square sq_bk = pos.square<KING>(BLACK);
		Square sq_wk = pos.square<KING>(WHITE);
		const auto* ppkppb = kpp[sq_bk];
		const auto* ppkppw = kpp[Inv(sq_wk)];

		auto& pos_ = *const_cast<Position*>(&pos);

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

		EvalLearningTools::init();
		for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i)
		{
			// 組み替えて異なる番号になったものだけ出力。
			auto fb = pos_.eval_list()->piece_list_fb()[i];
			auto fw = pos_.eval_list()->piece_list_fw()[i];
			auto fb_new = list_fb[i];
			auto fw_new = list_fw[i];

			// この変換後のfb,fwに対して、きちんと情報が設定されているかの確認。
			if (fb != fb_new || fw != fw_new)
				std::cout << "PieceNumber = " << i << " , fb = " << (int)fb << ":" << fb << " , fw = " << (int)fw << ":" << fw
				<< " , fb_new = " << (int)fb_new << " , fw_new = " << (int)fw_new
				<< " , mir(fb_new) = " << (int)EvalLearningTools::mir_piece(fb_new)
				<< " , mir(fw_new) = " << (int)EvalLearningTools::mir_piece(fw_new)
				<< " , inv(fb_new) = " << (int)EvalLearningTools::inv_piece(fb_new)
				<< " , inv(fw_new) = " << (int)EvalLearningTools::inv_piece(fw_new)
				<< std::endl;
		}
#endif

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

	// とりあえずここに書いておく。あとで移動させるかも。
#if defined(EVAL_LEARN)

	// regularize_kk()の下請け
	void regularize_kk_impl()
	{
		EvalLearningTools::init();

		typedef array<float, 2> kkt;

		array<kkt, SQ_NB> kk_offset, kkp_offset, kpp_offset;

		kkt zero = { 0, 0 };
		kk_offset.fill(zero);
		kkp_offset.fill(zero);
		kpp_offset.fill(zero);

		for (Square sq = SQ_ZERO; sq < SQ_NB; ++sq)
		{
			// sq2,p1,p2に依存しないkkの値を求める
			kkt sum_kkp = zero;
			kkt sum_kpp = zero;

			for (Square sq2 = SQ_ZERO; sq2 < SQ_NB; ++sq2)
			{
				for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
				{
					sum_kkp += kkp[sq][sq2][p];

					//sum_kkp[0] -= kkp[Inv(sq2)][Inv(sq)][EvalLearningTools::inv_piece(p)][0];
					//sum_kkp[1] += kkp[Inv(sq2)][Inv(sq)][EvalLearningTools::inv_piece(p)][1];
				}
			}
			for (auto p1 = 0; p1 < fe_end; ++p1)
				for (auto p2 = 0; p2 < fe_end; ++p2)
					sum_kpp[0] += kpp[sq][p1][p2];

			for (int i = 0; i < 2; ++i)
			{
				// kkpとkppの平均を求める。この分をあとでそれぞれの要素から引く。
				kkp_offset[sq][i] = sum_kkp[i] / (fe_end * (int)SQ_NB);
				kpp_offset[sq][i] = sum_kpp[i] / (fe_end * (int)fe_end);

				// kkpの計算のときにこれが38枚分、重なってくる
				// kppの計算のときにこれが38*37/2枚分、重なってくる
				kk_offset[sq][i] = 38 * kkp_offset[sq][i] + (38 * 37 / 2) * kpp_offset[sq][i];
			}
		}

		// offsetの計算が終わったので先後にこれを適用してやる。
		for (Square sq = SQ_ZERO; sq < SQ_NB; ++sq)
		{
			for (Square sq2 = SQ_ZERO; sq2 < SQ_NB; ++sq2)
			{
				kk[sq][sq2] += kk_offset[sq];

				// ここむっちゃ計算ややこしいが、これで合っとる。
				kk[Inv(sq2)][Inv(sq)][0] -= (int)kk_offset[sq][0];
				kk[Inv(sq2)][Inv(sq)][1] += (int)kk_offset[sq][1];

				for (auto p = 0; p < fe_end; ++p)
				{
					// ゼロの要素は書き換えない。(本来値がつくべきでないところを破壊すると困るため)
					if (kkp[sq][sq2][p][0])
					{
						kkp[sq][sq2][p] -= kkp_offset[sq];

						kkp[Inv(sq2)][Inv(sq)][inv_piece(BonaPiece(p))][0] += (int)kkp_offset[sq][0];
						kkp[Inv(sq2)][Inv(sq)][inv_piece(BonaPiece(p))][1] -= (int)kkp_offset[sq][1];
					}
				}
			}

			for (auto p1 = 0; p1 < fe_end; ++p1)
				for (auto p2 = 0; p2 < fe_end; ++p2)
					// ゼロの要素は書き換えない　またp1==0とかp2==0とかp1==p2のところは0になっているべき。
					if (kpp[sq][p1][p2] && p1 != p2 && p1 && p2)
						kpp[sq][p1][p2] = (ValueKpp)(kpp[sq][p1][p2] -  kpp_offset[sq][0]);
		}

	}

	// KKを正規化する関数。元の評価関数と完全に等価にはならないので注意。
	// kkp,kppの値をなるべくゼロに近づけることで、学習中に出現しなかった特徴因子の値(ゼロになっている)が
	// 妥当であることを保証しようという考え。
	void regularize_kk()
	{
		kk_stat();
		regularize_kk_impl();
		kk_stat();
	}
#endif

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
