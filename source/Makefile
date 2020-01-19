
# ビルドターゲット
# normal     : 通常使用用
# evallearn  : 教師局面からの学習用
# tournament : 大会で使う用
# gensfen    : 教師生成用(現状、非公開)


# やねうら王の使いたいエディションを指定する
# YANEURAOU_ENGINE_NNUE       : NNUE型評価関数(halfKP256),標準NNUE型
# YANEURAOU_ENGINE_NNUE_KP256 : NNUE型評価関数(KP256)
# YANEURAOU_ENGINE_KPPT       : KPPT型評価関数
# YANEURAOU_ENGINE_KPP_KKPT   : KPP_KKPT型評価関数
# YANEURAOU_ENGINE_MATERIAL   : 駒得のみの評価関数
# MATE_ENGINE                 : 詰将棋エンジン
# USER_ENGINE                 : USER定義エンジン

YANEURAOU_EDITION = YANEURAOU_ENGINE_NNUE
#YANEURAOU_EDITION = YANEURAOU_ENGINE_NNUE_KP256
#YANEURAOU_EDITION = YANEURAOU_ENGINE_KPPT
#YANEURAOU_EDITION = YANEURAOU_ENGINE_KPP_KKPT
#YANEURAOU_EDITION = YANEURAOU_ENGINE_MATERIAL
#YANEURAOU_EDITION = MATE_ENGINE
#YANEURAOU_EDITION = USER_ENGINE


# ターゲットCPU
# 利用できるCPU拡張命令を指定する。
# ARM系ならOTHERを指定する。
# 32bit環境用はNO_SSE
# AMDのZen2はZEN2を選択すると良いかも？(clang9以降で対応)

#TARGET_CPU = AVX512
TARGET_CPU = AVX2
#TARGET_CPU = SSE42
#TARGET_CPU = SSE41
#TARGET_CPU = SSE2
#TARGET_CPU = NO_SSE
#TARGET_CPU = OTHER
#TARGET_CPU = ZEN2


# デバッガーを使用するか
DEBUG = OFF
#DEBUG = ON


# clangでコンパイルしたほうがgccより数%速いっぽい。
#COMPILER = g++
COMPILER = clang++


# 標準的なコンパイルオプション
CFLAGS   = -std=c++17 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive
WCFLAGS  =
LDFLAGS  =
LIBS     =
INCLUDE  = # -I../include

# デバッガーを使わないなら、NDEBUGをdefineする。
ifeq ($(DEBUG),OFF)
	CFLAGS += -DNDEBUG
endif

# clang用にCFLAGSなどを変更
ifeq ($(findstring clang++,$(COMPILER)),clang++)
	# stdlib
	CFLAGS += -stdlib=libstdc++

	# C++17のfilesystem
	# LDFLAGS += -lstdc++fs

	# 関数の引数が関数本体で使われていないときに警告出るのうざすぎるので抑制。
	CFLAGS += -Wno-unused-parameter

	# static リンクを行う際に __cxa_guard_acquire __cxa_guard_release の生成を抑止
	#   undefined reference to `__imp___cxa_guard_acquire'
	#   undefined reference to `__imp___cxa_guard_release'
	# static 変数を初期化したかを pthread_mutex_t でロックを取って確認し、
	# 最初の実行なら初期化するスレッドセーフなコードを生成するためのもの。
	# → 本当に消してしまっても大丈夫か？
	WCFLAGS += -fno-threadsafe-statics

else
	ifeq ($(findstring g++,$(COMPILER)),g++)
		# mingw g++ で AVX512 向けビルドを行おうとするとエラーになる問題の回避
		# https://stackoverflow.com/questions/43152633/invalid-register-for-seh-savexmm-in-cygwin
		# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782
		WCFLAGS += -fno-asynchronous-unwind-tables

		# C++17のfilesystem
		# LDFLAGS += -lstdc++fs

	endif
endif

ifeq ($(OS),Windows_NT)
	CFLAGS += $(WCFLAGS)
	LDFLAGS += -static -Wl,--stack,25000000
	TARGET = YaneuraOu-by-gcc.exe
else
	CFLAGS += -D_LINUX
	TARGET = YaneuraOu-by-gcc
endif

# リンク時最適化。
# CFLAGSとLDFLAGSの両方で-fltoを指定する必要がある。
# cf. https://gcc.gnu.org/onlinedocs/gcc-6.3.0/gcc/Optimize-Options.html#Optimize-Options
LTOFLAGS = -flto

# wstringを使うためにこのシンボル定義が必要。
CFLAGS  += -DUNICODE

# stripの指示。(実行ファイルのサイズを小さく)
LDFLAGS += -Wl,-s

# mingw64では-D_WIN64,-D_WIN32は環境に応じて自動で設定されるので指定すべきではない。
# CFLAGS += -D_WIN64

# これを指定していると、各CPU向けの実行ファイルが生成されないので指定すべきではない。
# CFLAGS   += -march=native

# デバッグ情報を付加
# CFLAGS  += -g3 -ggdb

# OpenMPを使うときにCFLAGSとして指定するオプション
# ※ 学習部ではOpenMpを用いるので、学習用のビルドではこのオプションを指定する。
# clangでOPENMPを有効にしてビルドする方法については、解説.txtを参照のこと。

# エンジンの表示名("usi"コマンドに対して出力される)
#ENGINE_NAME =

# 開発中のbranchならdevと指定する
#ENGINE_BRANCH = dev

ifeq ($(findstring g++,$(COMPILER)),g++)
	OPENMP   = -fopenmp
	OPENMP_LDFLAGS =
endif
ifeq ($(findstring clang++,$(COMPILER)),clang++)
	ifeq ($(MSYSTEM),MINGW64)
		# MSYS2 MINGW64 env
		# libgompを指定した場合、ビルドは通るがOpenMPは無効化される？
		OPENMP = -fopenmp=libgomp
		OPENMP_LDFLAGS =
	else
		ifeq ($(findstring w64-mingw32,$(COMPILER)),w64-mingw32)
			# Ubuntu mingw-w64 clang++ env (experimental)
			OPENMP = -fopenmp=libgomp
			OPENMP_LDFLAGS =
		else
			# Other (normal clang++/libomp env)
			OPENMP = -fopenmp
			OPENMP_LDFLAGS = -lomp
		endif
	endif
endif

# NNUE評価関数 学習バイナリ用 OpenBLAS
ifeq ($(findstring YANEURAOU_ENGINE_NNUE,$(YANEURAOU_EDITION)),YANEURAOU_ENGINE_NNUE)
	BLAS = -DUSE_BLAS
	BLAS_LDFLAGS = -lopenblas
	ifeq ($(MSYSTEM),MINGW64)
		BLAS += -I$(shell cygpath -aw /mingw64/include/OpenBLAS)
	endif
endif

CFLAGS += -DNO_EXCEPTIONS
LDFLAGS += -lpthread
LDFLAGS += -v

OBJDIR   = ../obj
ifeq "$(strip $(OBJDIR))" ""
	OBJDIR = ..
endif

#SOURCES  = $(wildcard *.cpp)
SOURCES  = \
	main.cpp                                                                   \
	types.cpp                                                                  \
	bitboard.cpp                                                               \
	misc.cpp                                                                   \
	movegen.cpp                                                                \
	position.cpp                                                               \
	usi.cpp                                                                    \
	usi_option.cpp                                                             \
	thread.cpp                                                                 \
	tt.cpp                                                                     \
	movepick.cpp                                                               \
	timeman.cpp                                                                \
	extra/book/apery_book.cpp                                                  \
	extra/book/book.cpp                                                        \
	extra/book/makebook2019.cpp                                                \
	extra/bitop.cpp                                                            \
	extra/long_effect.cpp                                                      \
	extra/mate/mate1ply_with_effect.cpp                                        \
	extra/mate/mate1ply_without_effect.cpp                                     \
	extra/mate/mate_n_ply.cpp                                                  \
	extra/benchmark.cpp                                                        \
	extra/test_cmd.cpp                                                         \
	extra/see.cpp                                                              \
	extra/sfen_packer.cpp                                                      \
	extra/kif_converter/kif_convert_tools.cpp                                  \
	eval/evaluate_bona_piece.cpp                                               \
	eval/evaluate.cpp                                                          \
	eval/evaluate_io.cpp                                                       \
	eval/evaluate_mir_inv_tools.cpp                                            \
	learn/learner.cpp                                                          \
	learn/learning_tools.cpp                                                   \
	learn/multi_think.cpp

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPPT)
	SOURCES += \
		eval/kppt/evaluate_kppt.cpp                                            \
		eval/kppt/evaluate_kppt_learner.cpp                                    \
		engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPP_KKPT)
	SOURCES += \
		eval/kppt/evaluate_kppt.cpp                                            \
		eval/kpp_kkpt/evaluate_kpp_kkpt.cpp                                    \
		eval/kpp_kkpt/evaluate_kpp_kkpt_learner.cpp                            \
		engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_MATERIAL)
	SOURCES += \
		engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(findstring YANEURAOU_ENGINE_NNUE,$(YANEURAOU_EDITION)),YANEURAOU_ENGINE_NNUE)
	ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE)
	else
		CFLAGS += -DYANEURAOU_ENGINE_NNUE
	endif
	ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_KP256)
		CFLAGS += -DEVAL_NNUE_KP256
	else
		ifeq ($(NNUE_EVAL_ARCH),KP256)
			CFLAGS += -DEVAL_NNUE_KP256
		endif
	endif
	SOURCES += \
		eval/nnue/evaluate_nnue.cpp                                            \
		eval/nnue/evaluate_nnue_learner.cpp                                    \
		eval/nnue/nnue_test_command.cpp                                        \
		eval/nnue/features/k.cpp                                               \
		eval/nnue/features/p.cpp                                               \
		eval/nnue/features/half_kp.cpp                                         \
		eval/nnue/features/half_relative_kp.cpp                                \
		engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),MATE_ENGINE)
	SOURCES += engine/mate-engine/mate-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),USER_ENGINE)
	SOURCES += engine/user-engine/user-search.cpp
endif

ifneq ($(ENGINE_NAME),)
	CFLAGS += -DENGINE_NAME_FROM_MAKEFILE=$(ENGINE_NAME)
endif

# 開発用branch
ifeq ($(findstring dev,$(ENGINE_BRANCH)),dev)
	# SuperSort使ってみよう。
	SOURCES += extra/super_sort.cpp
endif


# CPU判別

# AMDなど特定のCPU
ifeq ($(TARGET_CPU),ZEN2)
	CFLAGS += -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -march=znver2

# それ以外は、AVX512,AVX2,SSE4.2,SSE4.1,SSE2のように利用できるCPU拡張命令で振り分ける。
else ifeq ($(TARGET_CPU),AVX512)
	# skylake     : -DUSE_AVX512 -DUSE_AVX512VLBWDQ -march=skylake-avx512
	# cascadelake : -DUSE_AVX512 -DUSE_AVX512VLBWDQ -DUSE_AVX512VNNI -march=cascadelake
	# icelake     : -DUSE_AVX512 -DUSE_AVX512VLBWDQ -DUSE_AVX512VNNI -DUSE_AVX512VBMI -DUSE_AVX512IFMA -USE_GFNI -march=icelake-client
	CFLAGS += -DUSE_AVX512 -DUSE_AVX512VLBWDQ -march=skylake-avx512

else ifeq ($(TARGET_CPU),AVX2)
	CFLAGS += -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -march=corei7-avx

else ifeq ($(TARGET_CPU),SSE42)
	CFLAGS += -DUSE_SSE42 -msse4.2 -march=corei7

else ifeq ($(TARGET_CPU),SSE41)
	CFLAGS += -DUSE_SSE41 -msse4.1 -march=core2

else ifeq ($(TARGET_CPU),SSE2)
	CFLAGS += -DUSE_SSE2 -msse2 -march=core2

else ifeq ($(TARGET_CPU),NO_SSE)
	# 32bit用。-m32は、MSYS2 MinGW-64だと無視されるので、
	# MinGW-64の32bit環境用でコンパイルする必要がある。
	CFLAGS += -DNO_SSE -m32 -march=pentium3

else ifeq ($(TARGET_CPU),OTHER)
	CFLAGS += -DNO_SSE

endif


CFLAGS += -DUSE_MAKEFILE -D$(YANEURAOU_EDITION)

OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS  = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(CFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html


# ビルドターゲット

# 通常使用。
normal:
	$(MAKE) CFLAGS='$(CFLAGS)' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

# 学習用。openmp , openblasなどを有効にする。
evallearn:
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) $(BLAS)' LDFLAGS='$(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS) $(LTOFLAGS)' $(TARGET)

# トーナメント用
tournament:
	$(MAKE) CFLAGS='$(CFLAGS) -DFOR_TOURNAMENT' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

# 教師棋譜生成用
gensfen:
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) $(BLAS) -DGENSFEN2019' LDFLAGS='$(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS) $(LTOFLAGS)' $(TARGET)


#　とりあえずPGOはAVX2とSSE4.2専用
prof:
	$(MAKE) CFLAGS='$(CFLAGS) -pg' tournament

profgen:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-generate -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-generate -lgcov'

profuse:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-use -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-use -lgcov $(LTOFLAGS)'

pgo:
	$(MAKE) profgen
	@./$(TARGET) EvalDir ../build/eval , bench , quit
	@touch $(SOURCES)
	$(MAKE) profuse

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET) ${OBJECTS:.o=.gcda}

-include $(DEPENDS)
