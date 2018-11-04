#include "../../shogi.h"

//
// Apery WCSC26の評価関数バイナリを読み込むための仕組み。
//
// このコードを書くに当たって、Apery、Silent Majorityのコードを非常に参考にさせていただきました。
// Special thanks to Takuya Hiraoka and Jangia , I was very impressed by their devouring enthusiasm.
//
// AVX2化に関してはtanuki-.さんにコードを提供していただきました。
// The evaluate function of AVX2 version is written by tanuki-.
// I pay my respects to his great achievements.
//

#if defined (EVAL_KPPT)

#include <fstream>
#include <iostream>
#include <unordered_set>

#include "evaluate_kppt.h"
#include "../evaluate_io.h"
#include "../../evaluate.h"
#include "../../position.h"
#include "../../misc.h"
#include "../../extra/bitop.h"

// 実験中の評価関数を読み込む。(現状非公開)
#if defined (EVAL_EXPERIMENTAL)
#include "../experimental/evaluate_experimental.h"
#endif

// EvalShareの機能を使うために必要
#if defined (USE_SHARED_MEMORY_IN_EVAL) && defined(_WIN32)
#include <codecvt>	 // mkdirするのにwstringが欲しいのでこれが必要
#include <locale>    // wstring_convertにこれが必要。
#include <windows.h>
#endif

#if defined(EVAL_LEARN)
#include "../../learn/learning_tools.h"
using namespace EvalLearningTools;
#endif

using namespace std;

namespace Eval
{

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
		auto make_name = [&](std::string filename) { return Path::Combine((string)Options["EvalDir"], filename); };
		auto input = EvalIO::EvalInfo::build_kppt32(make_name(KK_BIN), make_name(KKP_BIN), make_name(KPP_BIN));
		auto output = EvalIO::EvalInfo::build_kppt32((void*)kk, (void*)kkp, (void*)kpp);

		// 評価関数の実験のためにfe_endをKPPT32から変更しているかも知れないので現在のfe_endの値をもとに読み込む。
		input.fe_end = output.fe_end = Eval::fe_end;

		if (!EvalIO::eval_convert(input, output, nullptr))
			goto Error;

		{
			// 読み込み後に値を補正するとcheck sumが変化してしまうので、値の補正はlearnコマンド実行時にやるように変更した。
			// ここではそれ以外の実験用の補正コードを書いておく。

#if 0
			// Aperyの評価関数バイナリ、kkptは意味があるけどkpptはあまり意味がないので
			// 手番価値をクリアする実験用のコード

			for (auto sq : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
					for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
						kpp[sq][p1][p2][1] = 0;
#endif

#if 0
			// KPPTをPPTで代替えできないかを検証するためのコード
			// KPPTの値を平均化してPPTとして代入する。

			for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
				{
					int sum = 0;
					for (auto sq : SQ)
						sum += kpp[sq][p1][p2][1];

					int z = sum / SQ_NB;

					for (auto sq : SQ)
						kpp[sq][p1][p2][1] = z;
				}
#endif

#if 0
			// KPPTの手駒は手番必要ないのではないかを検証するためのコード
			for (auto sq : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_hand_end; ++p1)
					for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
					{
						kpp[sq][p1][p2][1] = 0;
						kpp[sq][p2][p1][1] = 0;
					}
#endif

#if 0
			// どんな値がついているのかダンプして観察する用。
			for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_hand_end; ++p1)
				for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
				{
					cout << p1 << "," << p2 << " = " << (int)kpp[SQ_88][p1][p2][1] << endl;
				}
#endif
		}

		// 読み込みは成功した。

		return;

	Error:;
		// 評価関数ファイルの読み込みに失敗した場合、思考を開始しないように抑制したほうがいいと思う。
		sync_cout << "\ninfo string Error! open evaluation file failed.\n" << sync_endl;
		my_exit();
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

		add_sum(reinterpret_cast<u16*>(kk) , size_of_kk  / sizeof(u16));
		add_sum(reinterpret_cast<u16*>(kkp), size_of_kkp / sizeof(u16));
		add_sum(reinterpret_cast<u16*>(kpp), size_of_kpp / sizeof(u16));

		return sum;
	}

	void init()
	{
#if defined(EVAL_EXPERIMENTAL)
		init_eval_experimental();
#endif
	}

	// 与えられたsize_of_evalサイズの連続したalign 32されているメモリに、kk_,kkp_,kpp_を割り当てる。
	void eval_assign(void* ptr)
	{
		s8* p = (s8*)ptr;
		kk_ = (ValueKk(*)[SQ_NB][SQ_NB]) (p);
		kkp_ = (ValueKkp(*)[SQ_NB][SQ_NB][fe_end]) (p + size_of_kk);
		kpp_ = (ValueKpp(*)[SQ_NB][fe_end][fe_end]) (p + size_of_kk + size_of_kkp);
	}

	void eval_malloc()
	{
		// benchコマンドなどでOptionsを保存して復元するのでこのときEvalDirが変更されたことになって、
		// 評価関数の再読込の必要があるというフラグを立てるため、この関数は2度呼び出されることがある。
		if (kk_ != nullptr)
		{
			aligned_free((void*)kk_);
			kk_ = nullptr;
		}

		// メモリ確保は一回にして、連続性のある確保にする。
		// このメモリは、プロセス終了のときに自動開放されることを期待している。
		eval_assign(aligned_malloc(size_of_eval, 32));
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

		// エンジンのバージョンによって評価関数は一意に定まるものとする。
		// Numaで確保する名前を変更しておく。

		auto dir_name = (string)Options["EvalDir"];
		// Mutex名にbackslashは使えないらしいので、escapeする。念のため'/'もescapeする。
		replace(dir_name.begin(), dir_name.end(), '\\', '_');
		replace(dir_name.begin(), dir_name.end(), '/', '_');
		// wchar_t*が必要なので変換する。
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
		cv.from_bytes(dir_name).c_str();

		auto mapped_file_name = TEXT("YANEURAOU_KPPT_MMF" ENGINE_VERSION) + cv.from_bytes(dir_name);
		auto mutex_name = TEXT("YANEURAOU_KPPT_MUTEX" ENGINE_VERSION) + cv.from_bytes(dir_name);

		// プロセス間の排他用mutex
		auto hMutex = CreateMutex(NULL, FALSE, mutex_name.c_str());

		// ファイルマッピングオブジェクトの処理をプロセス間で排他したい。
		WaitForSingleObject(hMutex, INFINITE);
		{

			// ファイルマッピングオブジェクトの作成
			auto hMap = CreateFileMapping(INVALID_HANDLE_VALUE,
				NULL,
				PAGE_READWRITE, // | /**SEC_COMMIT/**/ /*SEC_RESERVE/**/,
				(u32)(size_of_eval>>32), (u32)size_of_eval,
				mapped_file_name.c_str());

			bool already_exists = (GetLastError() == ERROR_ALREADY_EXISTS);

			// ビュー
			auto shared_eval_ptr = (void *)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size_of_eval);

			// メモリが確保できないときはshared_eval_ptr == null。このチェックをしたほうがいいような..。
			if (shared_eval_ptr == nullptr)
			{
				sync_cout << "info string can't allocate shared eval memory." << sync_endl;
				my_exit();
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

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);
		const auto* ppkppb = kpp[sq_bk];
		const auto* ppkppw = kpp[Inv(sq_wk)];

		auto& pos_ = *const_cast<Position*>(&pos);
		auto length = pos_.eval_list()->length();
		
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
		sum.p[0][0] = sum.p[0][1] = sum.p[1][0] = sum.p[1][1] = 0;
#endif

		// KK
		sum.p[2] = kk[sq_bk][sq_wk];

		for (i = 0; i < length ; ++i)
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
#if defined(USE_SSE41)
				// SSEによる実装

				// pkppw[l1][0],pkppw[l1][1],pkppb[l0][0],pkppb[l0][1]の16bit変数4つを整数拡張で32bit化して足し合わせる
				__m128i tmp;
				tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const int32_t*>(&pkppw[l1][0]), *reinterpret_cast<const int32_t*>(&pkppb[l0][0]));
				// この命令SSE4.1の命令のはず..
				tmp = _mm_cvtepi16_epi32(tmp);
				sum.m[0] = _mm_add_epi32(sum.m[0], tmp);
#else
				sum.p[0] += pkppb[l0];
				sum.p[1] += pkppw[l1];
#endif
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
	std::array<s32, 2> do_a_black(const Position& pos, const ExtBonaPiece ebp) {
		const Square sq_bk = pos.king_square(BLACK);
		const auto* list0 = pos.eval_list()->piece_list_fb();
		const int length = pos.eval_list()->length();

		const auto* pkppb = kpp[sq_bk][ebp.fb];
		std::array<s32, 2> sum = { { pkppb[list0[0]][0], pkppb[list0[0]][1] } };
		for (int i = 1; i < length ; ++i)
			sum += pkppb[list0[i]];
		return sum;
	}

	// 先手玉が移動したときの後手玉に対する差分
	std::array<s32, 2> do_a_white(const Position& pos, const ExtBonaPiece ebp) {
		const Square sq_wk = pos.king_square(WHITE);
		const auto* list1 = pos.eval_list()->piece_list_fw();
		const int length = pos.eval_list()->length();

		const auto* pkppw = kpp[Inv(sq_wk)][ebp.fw];
		std::array<s32, 2> sum = { { pkppw[list1[0]][0], pkppw[list1[0]][1] } };
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
		const Square sq_bk = pos.king_square(BLACK);
		const Square sq_wk = pos.king_square(WHITE);
		const auto list0 = pos.eval_list()->piece_list_fb();
		const auto list1 = pos.eval_list()->piece_list_fw();
		const int length = pos.eval_list()->length();

		EvalSum sum;

		// sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
#if defined(USE_SSE2)
		sum.m[0] = _mm_setzero_si128();
#else
		sum.p[0] = { 0, 0 };
		sum.p[1] = { 0, 0 };
#endif
		// KK
		sum.p[2] = kkp[sq_bk][sq_wk][ebp.fb];

		const auto* pkppb = kpp[sq_bk     ][ebp.fb];
		const auto* pkppw = kpp[Inv(sq_wk)][ebp.fw];

		// ここ、AVX512なら、_mm512_i32gather_epi32()が使える。
		// 気が向いたらコード書く。

#if defined (USE_AVX2)

		__m256i zero = _mm256_setzero_si256();
		__m256i sum0 = zero;
		__m256i sum1 = zero;

		// list0[38],list0[39],list1[38],list1[39]が0であることを期待したコード
		ASSERT_LV3(list0[38] == 0);

		for (int i = 0; i < length ; i += 8)
		{
			__m256i indexes0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&list0[i]));
			__m256i indexes1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&list1[i]));
			__m256i w0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppb), indexes0, 4);
			__m256i w1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppw), indexes1, 4);

			__m256i w0lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w0, 0));
			sum0 = _mm256_add_epi32(sum0, w0lo);
			__m256i w0hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w0, 1));
			sum0 = _mm256_add_epi32(sum0, w0hi);

			__m256i w1lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w1, 0));
			sum1 = _mm256_add_epi32(sum1, w1lo);
			__m256i w1hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w1, 1));
			sum1 = _mm256_add_epi32(sum1, w1hi);
		}

		// sum0とsum0の上位128ビットと下位128ビットを独立して8バイトシフトしたものを足し合わせる
		sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
		// sum0の上位128ビットと下位128ビットを足しあわせてsum0_128に代入する
		__m128i sum0_128 = _mm_add_epi32(_mm256_extracti128_si256(sum0, 0), _mm256_extracti128_si256(sum0, 1));
		// sum0_128の下位64ビットをdiff.p[1]にストアする
		std::array<int32_t, 2> sum0_array;
		_mm_storel_epi64(reinterpret_cast<__m128i*>(&sum0_array), sum0_128);
		sum.p[0] += sum0_array;

		// sum1とsum1の上位128ビットと下位128ビットを独立して8バイトシフトしたものを足し合わせる
		sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
		// sum1の上位128ビットと下位128ビットを足しあわせてsum1_128に代入する
		__m128i sum1_128 = _mm_add_epi32(_mm256_extracti128_si256(sum1, 0), _mm256_extracti128_si256(sum1, 1));
		// sum1_128の下位64ビットをdiff.p[1]にストアする
		std::array<int32_t, 2> sum1_array;
		_mm_storel_epi64(reinterpret_cast<__m128i*>(&sum1_array), sum1_128);
		sum.p[1] += sum1_array;

#elif defined (USE_SSE41)

		sum.m[0] = _mm_set_epi32(0, 0, *reinterpret_cast<const s32*>(&pkppw[list1[0]][0]), *reinterpret_cast<const s32*>(&pkppb[list0[0]][0]));
		sum.m[0] = _mm_cvtepi16_epi32(sum.m[0]);
		for (int i = 1; i < length ; ++i) {
			__m128i tmp;
			tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const s32*>(&pkppw[list1[i]][0]), *reinterpret_cast<const s32*>(&pkppb[list0[i]][0]));
			tmp = _mm_cvtepi16_epi32(tmp);
			sum.m[0] = _mm_add_epi32(sum.m[0], tmp);
		}
#else
		sum.p[0] = { pkppb[list0[0]][0] , pkppb[list0[0]][1] };
		sum.p[1] = { pkppw[list1[0]][0] , pkppw[list1[0]][1] };
		for (int i = 1; i < length ; ++i) {
			sum.p[0] += pkppb[list0[i]];
			sum.p[1] += pkppw[list1[i]];
		}
#endif

		return sum;
	}


#if defined (USE_EVAL_HASH)
	EvaluateHashTable g_evalTable;

	// prefetchする関数も用意しておく。
	void prefetch_evalhash(const Key key)
	{
		prefetch(g_evalTable[key >> 1]);
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
		const int length = pos.eval_list()->length();

		auto dirty = dp.pieceNo[0];

		// 移動させた駒は王か？
		if (dirty >= PIECE_NUMBER_KING)
		{
			// 前のnodeの評価値からの増分を計算していく。
			// (直接この変数に加算していく)
			// この意味においてdiffという名前は少々不適切ではあるが。
			EvalSum diff = prev->sum;

			auto sq_bk = pos.king_square(BLACK);
			auto sq_wk = pos.king_square(WHITE);

			// ΣKKPは最初から全計算するしかないので初期化する。
			diff.p[2] = kk[sq_bk][sq_wk];
			diff.p[2][0] += now->materialValue * FV_SCALE;

			// 後手玉の移動(片側分のKPPを丸ごと求める)
			if (dirty == PIECE_NUMBER_WKING)
			{
				const auto ppkppw = kpp[Inv(sq_wk)];

				// ΣWKPP
				diff.p[1] = { 0 , 0 };

#if defined(USE_AVX2)
				
				__m256i zero = _mm256_setzero_si256();
				__m256i diffp1 = zero;

				//#pragma unroll
				for (int i = 0; i < length ; ++i)
				{
					// KKPの値は、後手側から見た計算だとややこしいので、先手から見た計算でやる。
					// 後手から見た場合、kkp[inv(sq_wk)][inv(sq_bk)][k1]になるが、これ次元下げで同じ値を書いているとは限らない。
					diff.p[2] += kkp[sq_bk][sq_wk][list0[i]];

					const int k1 = list1[i];
					const auto* pkppw = ppkppw[k1];
					int j = 0;
					for (; j + 8 < i; j += 8)
					{
						// list1[j]から8要素ロードする
						__m256i indexes = _mm256_load_si256(reinterpret_cast<const __m256i*>(&list1[j]));
						// indexesのオフセットに従い、pkppwから8要素ギャザーする
						__m256i w = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppw), indexes, 4);
						// 下位128ビットを16ビット整数→32ビット整数に変換する
						__m256i wlo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 0));
						// diffp1に足し合わせる
						diffp1 = _mm256_add_epi32(diffp1, wlo);
						// 上位128ビットを16ビット整数→32ビット整数に変換する
						__m256i whi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));
						// diffp1に足し合わせる
						diffp1 = _mm256_add_epi32(diffp1, whi);
					}

					for (; j + 4 < i; j += 4) {
						// list1[j]から4要素ロードする
						__m128i indexes = _mm_load_si128(reinterpret_cast<const __m128i*>(&list1[j]));
						// indexesのオフセットに従い、pkppwから4要素ギャザーする
						__m128i w = _mm_i32gather_epi32(reinterpret_cast<const int*>(pkppw), indexes, 4);
						// 16ビット整数→32ビット整数に変換する
						__m256i wlo = _mm256_cvtepi16_epi32(w);
						// diffp1に足し合わせる
						diffp1 = _mm256_add_epi32(diffp1, wlo);
					}

					for (; j < i; ++j)
					{
						const int l1 = list1[j];
						diff.p[1] += pkppw[l1];
					}
				}

				// diffp1とdiffp1の上位128ビットと下位128ビットを独立して8バイトシフトしたものを足し合わせる
				diffp1 = _mm256_add_epi32(diffp1, _mm256_srli_si256(diffp1, 8));
				// diffp1の上位128ビットと下位128ビットを足しあわせてdiffp1_128に代入する
				__m128i diffp1_128 = _mm_add_epi32(_mm256_extracti128_si256(diffp1, 0), _mm256_extracti128_si256(diffp1, 1));
				// diffp1_128の下位64ビットをdiff.p[1]にストアする
				std::array<int32_t, 2> diffp1_sum;
				_mm_storel_epi64(reinterpret_cast<__m128i*>(&diffp1_sum), diffp1_128);
				diff.p[1] += diffp1_sum;
#else

				for (int i = 0; i < length ; ++i)
				{
					diff.p[2] += kkp[sq_bk][sq_wk][list0[i]];

					const int k1 = list1[i];
					const auto* pkppw = ppkppw[k1];
					for (int j = 0; j < i; ++j)
					{
						const int l1 = list1[j];
						diff.p[1] += pkppw[l1];
					}
				}
#endif

				// 動かした駒が２つ
				if (moved_piece_num == 2)
				{
					// 瞬間的にeval_listの移動させた駒の番号を変更してしまう。
					// こうすることで前nodeのpiece_listを持たなくて済む。

					const int listIndex_cap = dp.pieceNo[1];
					diff.p[0] += do_a_black(pos, dp.changed_piece[1].new_piece);
					list0[listIndex_cap] = dp.changed_piece[1].old_piece.fb;
					diff.p[0] -= do_a_black(pos, dp.changed_piece[1].old_piece);
					list0[listIndex_cap] = dp.changed_piece[1].new_piece.fb;
				}

			} else {

				// 先手玉の移動
				// さきほどの処理と同様。

				const auto* ppkppb = kpp[sq_bk];
				// ΣBKPP
				diff.p[0] = { 0,0 };

#if defined(USE_AVX2)

				__m256i zero = _mm256_setzero_si256();
				__m256i diffp0 = zero;

				//#pragma unroll
				for (int i = 0; i < length; ++i)
				{
					const int k0 = list0[i];
					const auto* pkppb = ppkppb[k0];

					// KKP
					diff.p[2] += kkp[sq_bk][sq_wk][k0];

					int j = 0;
					for (; j + 8 < i; j += 8)
					{
						// list0[j]から8要素ロードする
						__m256i indexes = _mm256_load_si256(reinterpret_cast<const __m256i*>(&list0[j]));
						// indexesのオフセットに従い、pkppwから8要素ギャザーする
						__m256i w = _mm256_i32gather_epi32(reinterpret_cast<const int*>(pkppb), indexes, 4);
						// 下位128ビットを16ビット整数→32ビット整数に変換する
						__m256i wlo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 0));
						// diffp0に足し合わせる
						diffp0 = _mm256_add_epi32(diffp0, wlo);
						// 上位128ビットを16ビット整数→32ビット整数に変換する
						__m256i whi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));
						// diffp0に足し合わせる
						diffp0 = _mm256_add_epi32(diffp0, whi);
					}

					for (; j + 4 < i; j += 4) {
						// list0[j]から4要素ロードする
						__m128i indexes = _mm_load_si128(reinterpret_cast<const __m128i*>(&list0[j]));
						// indexesのオフセットに従い、pkppwから4要素ギャザーする
						__m128i w = _mm_i32gather_epi32(reinterpret_cast<const int*>(pkppb), indexes, 4);
						// 16ビット整数→32ビット整数に変換する
						__m256i wlo = _mm256_cvtepi16_epi32(w);
						// diffp0に足し合わせる
						diffp0 = _mm256_add_epi32(diffp0, wlo);
					}

					for (; j < i; ++j)
					{
						const int l0 = list0[j];
						diff.p[0] += pkppb[l0];
					}
				}

				// diffp0とdiffp0の上位128ビットと下位128ビットを独立して8バイトシフトしたものを足し合わせる
				diffp0 = _mm256_add_epi32(diffp0, _mm256_srli_si256(diffp0, 8));
				// diffp0の上位128ビットと下位128ビットを足しあわせてdiffp0_128に代入する
				__m128i diffp0_128 = _mm_add_epi32(_mm256_extracti128_si256(diffp0, 0), _mm256_extracti128_si256(diffp0, 1));
				// diffp0_128の下位64ビットをdiff.p[1]にストアする
				std::array<int32_t, 2> diffp0_sum;
				_mm_storel_epi64(reinterpret_cast<__m128i*>(&diffp0_sum), diffp0_128);
				diff.p[0] += diffp0_sum;
#else
				for (int i = 0; i < length ; ++i)
				{
					const int k0 = list0[i];
					const auto* pkppb = ppkppb[k0];
					for (int j = 0; j < i; ++j) {
						const int l0 = list0[j];
						diff.p[0] += pkppb[l0];
					}
					diff.p[2] += kkp[sq_bk][sq_wk][k0];
				}
#endif

				if (moved_piece_num == 2) {
					const int listIndex_cap = dp.pieceNo[1];
					diff.p[1] += do_a_white(pos, dp.changed_piece[1].new_piece);
					list1[listIndex_cap] = dp.changed_piece[1].old_piece.fw;
					diff.p[1] -= do_a_white(pos, dp.changed_piece[1].old_piece);
					list1[listIndex_cap] = dp.changed_piece[1].new_piece.fw;
				}
			}

			// sumの計算が終わったのでpos.state()->sumに反映させておく。(これがこの関数の返し値に相当する。)
			now->sum = diff;

		} else {

			// 王以外の駒が移動したケース
			// 今回の差分を計算して、そこに加算する。

			const int listIndex = dp.pieceNo[0];

			// do_a_pc()の最適化の関係で使わない玉の情報を吹き飛ばし、0にする
			list0[38] = static_cast<BonaPiece>(0);
			list0[39] = static_cast<BonaPiece>(0);
			list1[38] = static_cast<BonaPiece>(0);
			list1[39] = static_cast<BonaPiece>(0);

			auto diff = do_a_pc(pos, dp.changed_piece[0].new_piece);
			if (moved_piece_num == 1) {

				// 動いた駒が1つ。
				list0[listIndex] = dp.changed_piece[0].old_piece.fb;
				list1[listIndex] = dp.changed_piece[0].old_piece.fw;
				diff -= do_a_pc(pos, dp.changed_piece[0].old_piece);

			} else {

				// 動いた駒が2つ。

				auto sq_bk = pos.king_square(BLACK);
				auto sq_wk = pos.king_square(WHITE);

				diff += do_a_pc(pos, dp.changed_piece[1].new_piece);
				diff.p[0] -= kpp[    sq_bk ][dp.changed_piece[0].new_piece.fb][dp.changed_piece[1].new_piece.fb];
				diff.p[1] -= kpp[Inv(sq_wk)][dp.changed_piece[0].new_piece.fw][dp.changed_piece[1].new_piece.fw];

				const PieceNumber listIndex_cap = dp.pieceNo[1];
				list0[listIndex_cap] = dp.changed_piece[1].old_piece.fb;
				list1[listIndex_cap] = dp.changed_piece[1].old_piece.fw;

				list0[listIndex] = dp.changed_piece[0].old_piece.fb;
				list1[listIndex] = dp.changed_piece[0].old_piece.fw;
				diff -= do_a_pc(pos, dp.changed_piece[0].old_piece);
				diff -= do_a_pc(pos, dp.changed_piece[1].old_piece);

				diff.p[0] += kpp[    sq_bk ][dp.changed_piece[0].old_piece.fb][dp.changed_piece[1].old_piece.fb];
				diff.p[1] += kpp[Inv(sq_wk)][dp.changed_piece[0].old_piece.fw][dp.changed_piece[1].old_piece.fw];
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
		// 手番を消した局面hash key
		const Key keyExcludeTurn = st->key() >> 1;

		// evaluate hash tableにはあるかも。

		//		cout << "EvalSum " << hex << g_evalTable[keyExcludeTurn] << endl;
		EvalSum entry = *g_evalTable[keyExcludeTurn];   // atomic にデータを取得する必要がある。
		entry.decode();
		if (entry.key == keyExcludeTurn)
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

#if defined ( USE_EVAL_HASH )
		// せっかく計算したのでevaluate hash tableに保存しておく。
		sum.key = keyExcludeTurn;
		sum.encode();
		*g_evalTable[keyExcludeTurn] = sum;
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
		//ASSERT_LV3(abs(v) < VALUE_MAX_EVAL);
#if 0
		if (!((abs(v) < VALUE_MAX_EVAL)))
			std::cout << pos << std::endl;
#endif

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
					sum_kpp += kpp[sq][p1][p2];

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
					sum_kpp[0] -= kpp[Inv(sq)][inv_piece(p1)][inv_piece(p2)][0];
					sum_kpp[1] += kpp[Inv(sq)][inv_piece(p1)][inv_piece(p2)][1];
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

		Square sq_bk = pos.king_square(BLACK);
		Square sq_wk = pos.king_square(WHITE);
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

		const int length = pos_.eval_list()->length();

		// 38枚の駒を表示
		for (i = 0; i < length ; ++i)
			cout << int(list_fb[i]) << " = " << list_fb[i] << " , " << int(list_fw[i]) << " =  " << list_fw[i] << endl;

		// 評価値の合計
		EvalSum sum;

#if defined(USE_SSE2)
		// sum.p[0](BKPP)とsum.p[1](WKPP)をゼロクリア
		sum.m[0] = _mm_setzero_si128();
#else
		sum.p[0][0] = sum.p[0][1] = sum.p[1][0] = sum.p[1][1] = 0;
#endif

		// KK
		sum.p[2] = kk[sq_bk][sq_wk];
		cout << "KKC : " << sq_bk << " " << sq_wk << " = " << kk[sq_bk][sq_wk][0] << " + " << kk[sq_bk][sq_wk][1] << endl;

		for (i = 0; i < length ; ++i)
		{
			k0 = list_fb[i];
			k1 = list_fw[i];
			const auto* pkppb = ppkppb[k0];
			const auto* pkppw = ppkppw[k1];
			for (j = 0; j < i; ++j)
			{
				l0 = list_fb[j];
				l1 = list_fw[j];

#if defined(USE_SSE41)
				// SSEによる実装

				// pkppw[l1][0],pkppw[l1][1],pkppb[l0][0],pkppb[l0][1]の16bit変数4つを整数拡張で32bit化して足し合わせる
				__m128i tmp;
				tmp = _mm_set_epi32(0, 0, *reinterpret_cast<const int32_t*>(&pkppw[l1][0]), *reinterpret_cast<const int32_t*>(&pkppb[l0][0]));
				// この命令SSE4.1の命令のはず
				tmp = _mm_cvtepi16_epi32(tmp);
				sum.m[0] = _mm_add_epi32(sum.m[0], tmp);

				cout << "BKPP : " << sq_bk << " " << k0 << " " << l0 << " = " << pkppb[l0][0] << " + " << pkppb[l0][1] << endl;
				cout << "WKPP : " << sq_wk << " " << k1 << " " << l1 << " = " << pkppw[l1][0] << " + " << pkppw[l1][1] << endl;

#else
				sum.p[0] += pkppb[l0];
				sum.p[1] += pkppw[l1];
#endif
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

		array<kkt, SQ_NB> kk_offset , kkp_offset , kpp_offset;

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
					sum_kpp += kpp[sq][p1][p2];

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
					if (kpp[sq][p1][p2][0] && p1!=p2 && p1 && p2)
						kpp[sq][p1][p2] -= kpp_offset[sq];
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
	void foreach_eval_param(std::function<void(s32,s32)>f , int type)
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
				f(v[0], v[1]);
			}
		}
	}


}

#endif // defined (EVAL_KPPT)
