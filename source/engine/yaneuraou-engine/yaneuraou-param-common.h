// 探索部以外でパラメーターを自動調整したい時に読み込むヘッダ

#if defined(TUNING_SEARCH_PARAMETERS)
	// ハイパーパラメーターを自動調整するときは変数にしておいて変更できるようにする。
	#include "param/yaneuraou-param-extern.h"
#else
	// 変更しないとき
	#define PARAM_DEFINE constexpr int
	#include "yaneuraou-param.h"
#endif

