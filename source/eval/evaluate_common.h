#ifndef _EVALUATE_COMMON_H_
#define _EVALUATE_COMMON_H_

// いまどきの手番つき評価関数(EVAL_KPPTとEVAL_KPP_KKPT)の共用header的なもの。

#if defined (EVAL_KPPT) || defined(EVAL_KPP_KKPT) || defined(EVAL_NNUE)
#include <functional>

namespace YaneuraOu {
namespace Eval
{

#if defined (EVAL_KPPT) || defined(EVAL_KPP_KKPT) 

	// KKファイル名
	constexpr const char* KK_BIN = "KK_synthesized.bin";

	// KKPファイル名
	constexpr const char* KKP_BIN = "KKP_synthesized.bin";

	// KPPファイル名
	constexpr const char* KPP_BIN = "KPP_synthesized.bin";

#endif

#if defined(USE_EVAL_HASH)
	// prefetchする関数
    using Key64 = uint64_t;
	void prefetch_evalhash(const Key64 key);
#endif

	// 評価関数のそれぞれのパラメーターに対して関数fを適用してくれるoperator。
	// パラメーターの分析などに用いる。
	// typeは調査対象を表す。
	//   type = -1 : KK,KKP,KPPすべて
	//   type = 0  : KK のみ 
	//   type = 1  : KKPのみ 
	//   type = 2  : KPPのみ 
	void foreach_eval_param(std::function<void(s32, s32)>f, int type = -1);


} // Eval
} // namespace YaneuraOu

#endif

#endif // _EVALUATE_KPPT_COMMON_H_
