#ifndef _EVALUATE_IO_H_
#define _EVALUATE_IO_H_

// あらゆる形式の評価関数のファイル←→メモリ間、ファイル←→ファイル間の入力/出力、フォーマットの変換を行なう。

#include "../shogi.h"

namespace EvalIO
{

	// 特徴因子名
	enum EvalFeature { KK , KKP, KPP, PP , KPPP, KKPP };

	// 実体がメモリにあるときは、そのポインタ、
	// 実体がファイルにあるときは、そのファイル名を保持する構造体
	struct FileOrMemory
	{
		FileOrMemory(std::string filename_) : filename(filename_) , ptr(nullptr) {}
		FileOrMemory(void* ptr_) : ptr(ptr_){}

		// メモリを対象とするのか？
		bool memory() const { return ptr != nullptr; }

		// ファイルを対象とするのか？
		bool file() const { return ptr == nullptr; }

		// 実体を指し示すポインタ。
		void* ptr;
		
		// ファイル名
		std::string filename;
	};


	// ある配列の内部形式を定義する。
	struct EvalArrayInfo
	{
		EvalArrayInfo(EvalFeature feature_, u64 element_size_ , u64 element_num_ , FileOrMemory file_or_memory_) :
			feature(feature_) , element_size(element_size_) , element_num(element_num_) , file_or_memory(file_or_memory_)
		{}
			
		// 特徴因子名
		EvalFeature feature;
		
		// 1つの要素のバイト長。
		u64 element_size;

		// 1つの評価項目は、いくつの要素から成るか。
		u64 element_num;

		// そのデータの実体(のある場所)
		FileOrMemory file_or_memory;
	};

	// よく使う評価関数の型情報を返すためのbuilder
	struct EvalInfo
	{
		EvalInfo(u64 SQ_NB_, u64 fe_end_) : sq_nb(SQ_NB_), fe_end(fe_end_) {}

		// Kの配置できる升の数
		u64 sq_nb;

		// Pの数
		u64 fe_end;

		// 評価関数すべての型を定義する型
		std::vector<EvalArrayInfo> eval_info_array;

		// --- 以下、よく使いそうなものだけ定義しておく。あとは自分で定義して使うべし。

		// やねうら王2015のKPP型評価関数の型定義を返すbuilder。
		// 引数にはFileOrMemoryのコンストラクタに渡す、std::string filenameかvoid* ptr を渡す。
		template <typename T1,typename T2, typename T3>
		static EvalInfo build_kpp(T1 kk_, T2 kkp_, T3 kpp_)
		{
			EvalInfo ei(81 /* SQ_NB */ ,1535 /* EvalKPP::fe_end */);
			ei.eval_info_array.emplace_back(EvalArrayInfo(KK , 4, 1 , FileOrMemory(kk_ ))); // KK は4バイト。(手番なしなので1つ)
			ei.eval_info_array.emplace_back(EvalArrayInfo(KKP, 4, 1 , FileOrMemory(kkp_))); // KKPは4バイト。
			ei.eval_info_array.emplace_back(EvalArrayInfo(KPP, 2, 1 , FileOrMemory(kpp_))); // KPPは2バイト。
			return ei;
		}

		// やねうら王2016 , Apery(WCSC26)のKPPT型評価関数の型定義を返すbuilder。
		// 引数にはFileOrMemoryのコンストラクタに渡す、std::string filenameかvoid* ptr を渡す。
		template <typename T1, typename T2, typename T3>
		static EvalInfo build_kppt32(T1 kk_, T2 kkp_, T3 kpp_)
		{
			EvalInfo ei(81 /* SQ_NB */, 1548 /* EvalKPPT::fe_end */);
			ei.eval_info_array.emplace_back(EvalArrayInfo(KK , 4, 2 , FileOrMemory(kk_ ))); // KK は4バイト。(手番ありなので2つ)
			ei.eval_info_array.emplace_back(EvalArrayInfo(KKP, 4, 2 , FileOrMemory(kkp_))); // KKPは4バイト。
			ei.eval_info_array.emplace_back(EvalArrayInfo(KPP, 2, 2 , FileOrMemory(kpp_))); // KPPは2バイト。
			return ei;
		}

		// Apery(WCSC27)のKPPT型評価関数の型定義を返すbuilder。KK,KKPが16bit化されている。
		// 引数にはFileOrMemoryのコンストラクタに渡す、std::string filenameかvoid* ptr を渡す。
		template <typename T1, typename T2, typename T3>
		static EvalInfo build_kppt16(T1 kk_, T2 kkp_, T3 kpp_)
		{
			EvalInfo ei(81 /* SQ_NB */, 1548 /* EvalKPPT::fe_end */);
			ei.eval_info_array.emplace_back(EvalArrayInfo(KK , 2, 2 , FileOrMemory(kk_  ))); // KK は2バイト。(手番ありなので2つ)
			ei.eval_info_array.emplace_back(EvalArrayInfo(KKP, 2, 2 , FileOrMemory(kkp_ ))); // KKPは2バイト。
			ei.eval_info_array.emplace_back(EvalArrayInfo(KPP, 2, 2 , FileOrMemory(kpp_ ))); // KPPは2バイト。
			return ei;
		}

	};

	// 評価関数の変換＋αを行なう。
	//
	// ファイルからメモリ、メモリからファイル、ファイルからファイル、メモリからメモリ間での
	// 評価関数のフォーマットの変換や、読み込み、書き込みが出来る。
	//
	// この関数の使用例としては、test_cmd.cppやevaluate_kppt.cppなどを見ること。
	// この関数の返し値は、変換/読み込み/書き込みに成功するとtrueが返る。何らか失敗するとfalseが返る。
	//
	// また、inputとoutputに関してフォーマットが同じで、変換を要しないときは、単なるメモリコピーや
	// ファイルコピーで済む実装になっており、オーバーヘッドは最小限で済むように実装されている。
	//
	// また、引数のmap変数で、KPPTなどのP(BonaPiece)の値の変換テーブルを指定できる。
	// 出力側のPがaのときに入力側のmap[a]のPとして扱う。
	// このmapを指定したくないとき(Pに関して恒等変換で良い場合)は、mapとしてnullptrを渡すこと。
	//
	// 注意)
	// input.fe_end < output.fe_endのように、fe_endを拡張するとき、
	// mapを引数で渡して、拡張された領域が元の領域とどう対応するのか表現する必要がある。
	//
	extern bool eval_convert(const EvalInfo& input, const EvalInfo& output, const std::vector<u16 /*BonaPiece*/>* map);

}

#endif // _EVALUATE_IO_H_
