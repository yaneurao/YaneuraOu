# > Select YaneuraOu edition below.

YANEURAOU_EDITION = YANEURAOU_2018_OTAFUKU_ENGINE
#YANEURAOU_EDITION = MATE_ENGINE
#YANEURAOU_EDITION = YANEURAOU_2018_TNK_ENGINE


# clangでコンパイルしたほうがgccより数%速いっぽい。
#COMPILER = g++
COMPILER = clang++

# 標準的なコンパイルオプション
CFLAGS   = -std=c++14 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive
WCFLAGS  =
LDFLAGS  =
LIBS     =
INCLUDE  = # -I../include

# clang用にCFLAGSなどを変更
ifeq ($(findstring clang++,$(COMPILER)),clang++)
	# stdlib
	CFLAGS += -stdlib=libstdc++

	# 関数の引数が関数本体で使われていないときに警告出るのうざすぎるので抑制。
	CFLAGS += -Wno-unused-parameter

	# static リンクを行う際に __cxa_guard_acquire __cxa_guard_release の生成を抑止
	#   undefined reference to `__imp___cxa_guard_acquire'
	#   undefined reference to `__imp___cxa_guard_release'
	# static 変数を初期化したかを pthread_mutex_t でロックを取って確認し、
	# 最初の実行なら初期化するスレッドセーフなコードを生成するためのもの。
	# → 本当に消してしまっても大丈夫か？
	WCFLAGS += -fno-threadsafe-statics

	# msys2用のclangでは、OpenMP使えない？インスコ仕方がわからん。

endif

ifeq ($(OS),Windows_NT)
	CFLAGS += $(WCFLAGS)
	LDFLAGS += -static -Wl,--stack,25000000
	TARGET = YaneuraOu-by-gcc.exe
else
	CFLAGS += -D_LINUX
	TARGET = YaneuraOu-by-gcc
endif

# リンク時最適化。これをつけるとmsys2環境だとセグフォで落ちる。
# 落ちない環境なら有効にしたほうが速い実行ファイルが生成されるはず。
# CFLAGS += -flto

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

ifeq ($(findstring g++,$(COMPILER)),g++)
	OPENMP   = -fopenmp
	OPENMP_LDFLAGS =
	ifeq ($(YANEURAOU_EDITION),YANEURAOU_2018_TNK_ENGINE)
		BLAS = -DUSE_BLAS
		BLAS_LDFLAGS = -lopenblas
		ifeq ($(MSYSTEM),MINGW64)
			BLAS += -I$(shell cygpath -aw /mingw64/include/OpenBLAS)
		endif
	endif
endif
ifeq ($(findstring clang++,$(COMPILER)),clang++)
	OPENMP   = -fopenmp
	OPENMP_LDFLAGS = -lomp
	ifeq ($(YANEURAOU_EDITION),YANEURAOU_2018_TNK_ENGINE)
		BLAS = -DUSE_BLAS
		BLAS_LDFLAGS = -lopenblas
		ifeq ($(MSYSTEM),MINGW64)
			BLAS += -I$(shell cygpath -aw /mingw64/include/OpenBLAS)
		endif
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
SOURCES  = main.cpp                                                            \
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
	extra/entering_king_win.cpp                                                \
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
	eval/kppt/evaluate_kppt.cpp                                                \
	eval/kppt/evaluate_kppt_learner.cpp                                        \
	eval/kpp_kkpt/evaluate_kpp_kkpt.cpp                                        \
	eval/kpp_kkpt/evaluate_kpp_kkpt_learner.cpp                                \
	eval/evaluate.cpp                                                          \
	eval/evaluate_io.cpp                                                       \
	eval/evaluate_mir_inv_tools.cpp                                            \
	engine/user-engine/user-search.cpp                                         \
	engine/2018-otafuku-engine/2018-otafuku-search.cpp                         \
	learn/learner.cpp                                                          \
	learn/learning_tools.cpp                                                   \
	learn/multi_think.cpp

ifeq ($(YANEURAOU_EDITION),MATE_ENGINE)
	SOURCES += engine/mate-engine/mate-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_2018_TNK_ENGINE)
	SOURCES += eval/nnue/evaluate_nnue.cpp                                     \
	           eval/nnue/evaluate_nnue_learner.cpp                             \
	           eval/nnue/nnue_test_command.cpp                                 \
	           eval/nnue/features/k.cpp                                        \
	           eval/nnue/features/p.cpp                                        \
	           eval/nnue/features/half_kp.cpp                                  \
	           eval/nnue/features/half_relative_kp.cpp
endif

OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS  = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(CFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

# 学習用バイナリのときは、openmpを有効にする。
evallearn:
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) $(BLAS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS) $(LTOFLAGS)' $(TARGET)
evallearn-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) $(BLAS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE42 -msse4.2 -march=corei7' LDFLAGS='$(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS) $(LTOFLAGS)' $(TARGET)

# 教師棋譜生成用
gensfen:
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) $(BLAS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -DGENSFEN2019 -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS) $(LTOFLAGS)' $(TARGET)

tournament:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -DFOR_TOURNAMENT -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)
tournament-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE42 -msse4.2 -DFOR_TOURNAMENT -march=corei7' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

avx2:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

sse42:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE42 -msse4.2 -march=corei7' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

sse41:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE41 -msse4.1 -march=core2' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

sse2:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE2 -msse2 -march=core2' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

# 32bit用。-m32は、MSYS2 MinGW-64だと無視されるので、
# MinGW-64の32bit環境用でコンパイルする必要がある。
nosse:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DNO_SSE -m32 -march=pentium3' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

# ARMなどのCPU
other:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DNO_SSE' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

#　とりあえずPGOはAVX2とSSE4.2専用
prof:
	$(MAKE) CFLAGS='$(CFLAGS) -pg' avx2
prof-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) -pg' sse42

profgen:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-generate -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-generate -lgcov' avx2
profgen-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-generate -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-generate -lgcov' sse42

profuse:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-use -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-use -lgcov $(LTOFLAGS)' avx2
profuse-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-use -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-use -lgcov $(LTOFLAGS)' sse42

pgo:
	$(MAKE) profgen
	@./$(TARGET) EvalDir ../build/eval , bench , quit
	@touch $(SOURCES)
	$(MAKE) profuse
pgo-sse42:
	$(MAKE) profgen-sse42
	@./$(TARGET) EvalDir ../build/eval , bench , quit
	@touch $(SOURCES)
	$(MAKE) profuse-sse42

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET) ${OBJECTS:.o=.gcda}

-include $(DEPENDS)
