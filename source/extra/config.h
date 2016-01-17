#ifndef _CONFIG_H_
#define _CONFIG_H_

// --- 協力詰めエンジンとして実行ファイルを公開するとき用の設定集

#ifdef COOPERATIVE_MATE_SOLVER
#undef ASSERT_LV
#define KEEP_LAST_MOVE
#undef  MAX_PLY_
#define MAX_PLY_ 65000
#undef HASH_KEY_BITS
#define HASH_KEY_BITS 128
#undef USE_EVAL_TABLE
#undef MATE_1PLY
#endif

// --- 通常の思考エンジンとして実行ファイルを公開するとき用の設定集

#ifdef YANEURAOU_MINI // 思考エンジンとしてリリースする場合(実行速度を求める場合)
#undef ASSERT_LV
#undef KEEP_LAST_MOVE
#undef MATE_1PLY
#endif



#endif // _CONFIG_H_

