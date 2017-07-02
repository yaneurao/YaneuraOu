# > Select YaneuraOu edition below.

YANEURAOU_EDITION = YANEURAOU_2017_EARLY_ENGINE
#YANEURAOU_EDITION = HELP_MATE_ENGINE
#YANEURAOU_EDITION = MATE_ENGINE


# clangでコンパイルしたほうがgccより数%速いっぽい。
# COMPILER = g++
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
  LDFLAGS += -static -Wl,--stack,100000000
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
OPENMP   = -fopenmp

CFLAGS += -DNO_EXCEPTIONS
LDFLAGS += -lpthread
LDFLAGS += -v

OBJDIR   = ../obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = ..
endif
#SOURCES  = $(wildcard *.cpp)
SOURCES  = shogi.cpp                                                           \
	bitboard.cpp                                                               \
	misc.cpp                                                                   \
	movegen.cpp                                                                \
	position.cpp                                                               \
	usi.cpp                                                                    \
	thread.cpp                                                                 \
	tt.cpp                                                                     \
	extra/book/apery_book.cpp                                                  \
	extra/book/book.cpp                                                        \
	extra/bitop.cpp                                                            \
	extra/entering_king_win.cpp                                                \
	extra/long_effect.cpp                                                      \
	extra/mate/mate1ply_with_effect.cpp                                        \
	extra/mate/mate1ply_without_effect.cpp                                     \
	extra/mate/mate_n_ply.cpp                                                  \
	extra/move_picker/move_picker_2017Q2.cpp                                   \
	extra/benchmark.cpp                                                        \
	extra/test_cmd.cpp                                                         \
	extra/timeman.cpp                                                          \
	extra/see.cpp                                                              \
	extra/sfen_packer.cpp                                                      \
	extra/kif_converter/kif_convert_tools.cpp                                  \
	eval/evaluate_bona_piece.cpp                                               \
	eval/evaluate_kpp.cpp                                                      \
	eval/evaluate_kppt.cpp                                                     \
	eval/evaluate.cpp                                                          \
	eval/evaluate_no_eval.cpp                                                  \
	eval/evaluate_io.cpp                                                       \
	eval/progress.cpp                                                          \
	engine/user-engine/user-search.cpp                                         \
	engine/help-mate-engine/help-mate-search.cpp                               \
	engine/2017-early-engine/2017-early-search.cpp                             \
	learn/learner.cpp                                                          \
	learn/learning_tools.cpp                                                   \
	learn/multi_think.cpp                                                      \
	learn/evaluate_kppt_learn.cpp   

#ifeq ($(YANEURAOU_EDITION),YANEURAOU_2017_GOKU_ENGINE)                        
#	SOURCES += engine/2017-goku-engine/2017-goku-search.cpp                    \
#	           eval/evaluate_ppap.cpp                                          
#endif

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
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi2 -mavx2 -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)
evallearn-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) $(OPENMP) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE42 -msse4.2 -march=corei7' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

tournament:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi2 -mavx2 -DFOR_TOURNAMENT -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)
tournament-sse42:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_SSE42 -msse4.2 -DFOR_TOURNAMENT -march=corei7' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

avx2:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi2 -mavx2 -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

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
