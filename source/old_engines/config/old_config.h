#ifdef YANEURAOU_NANO_ENGINE
#define ENGINE_NAME "YaneuraOu nano"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_TT_PV
#define KEEP_LAST_MOVE
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_NANO_PLUS_ENGINE
#define ENGINE_NAME "YaneuraOu nano plus"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_TT_PV
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_MINI_ENGINE
#define ENGINE_NAME "YaneuraOu mini"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define USE_DROPBIT_IN_STATS
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_CLASSIC_ENGINE
#define ENGINE_NAME "YaneuraOu classic"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_DROPBIT_IN_STATS
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_CLASSIC_TCE_ENGINE
#define ENGINE_NAME "YaneuraOu classic-tce"
#define ENABLE_TEST_CMD
#define EVAL_KPP
#define USE_SEE
#define USE_MOVE_PICKER_2015
#define LONG_EFFECT_LIBRARY
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define USE_DROPBIT_IN_STATS
#define KEEP_PIECE_IN_GENERATE_MOVES
#endif

#ifdef YANEURAOU_2016_MID_ENGINE
#define ENGINE_NAME "YaneuraOu 2016 Mid"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2016Q2
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define ENABLE_TEST_CMD
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
#endif

#ifdef YANEURAOU_2016_LATE_ENGINE
#define ENGINE_NAME "YaneuraOu 2016 Late"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2016Q3
#define USE_MATE_1PLY
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define PER_THREAD_COUNTERMOVEHISTORY
#define PER_STACK_HISTORY

#define ENABLE_TEST_CMD
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY
#endif


#ifdef MUST_CAPTURE_SHOGI_ENGINE
#define ENGINE_NAME "YaneuraOu MustCaptureShogi"
#define EVAL_KPPT
//#define USE_EVAL_HASH
#define USE_SEE
#define USE_MOVE_PICKER_2016Q3
#define USE_ENTERING_KING_WIN
#define USE_TIME_MANAGEMENT
#define KEEP_PIECE_IN_GENERATE_MOVES
#define ONE_PLY_EQ_1
#define PER_THREAD_COUNTERMOVEHISTORY
#define PER_STACK_HISTORY

// デバッグ絡み
#define ASSERT_LV 3
#define USE_DEBUG_ASSERT

#define ENABLE_TEST_CMD
// 学習絡みのオプション
#define USE_SFEN_PACKER
#define EVAL_LEARN
// 定跡生成絡み
#define ENABLE_MAKEBOOK_CMD
// 評価関数を共用して複数プロセス立ち上げたときのメモリを節約。(いまのところWindows限定)
#define USE_SHARED_MEMORY_IN_EVAL
// パラメーターの自動調整絡み
#define USE_GAMEOVER_HANDLER
//#define LONG_EFFECT_LIBRARY
#endif


#ifdef RANDOM_PLAYER_ENGINE
#define ENGINE_NAME "YaneuraOu random player"
#define EVAL_NO_USE
#define ASSERT_LV 3
#endif




