# > Select YaneuraOu edition below.

#YANEURAOU_EDITION = RANDOM_PLAYER_ENGINE
#YANEURAOU_EDITION = HELP_MATE_ENGINE
#YANEURAOU_EDITION = YANEURAOU_NANO_ENGINE
#YANEURAOU_EDITION = YANEURAOU_NANO_PLUS_ENGINE
#YANEURAOU_EDITION = YANEURAOU_MINI_ENGINE
#YANEURAOU_EDITION = YANEURAOU_CLASSIC_ENGINE
#YANEURAOU_EDITION = YANEURAOU_CLASSIC_TCE_ENGINE
#YANEURAOU_EDITION = YANEURAOU_2016_MID_ENGINE
#YANEURAOU_EDITION = YANEURAOU_2016_LATE_ENGINE
YANEURAOU_EDITION = YANEURAOU_2017_EARLY_ENGINE
#YANEURAOU_EDITION = CHECK_SHOGI_ENGINE
#YANEURAOU_EDITION = MUST_CAPTURE_SHOGI_ENGINE

# clangでコンパイルしたほうが数%速いっぽい。
# ToDo : あとでclang用のMakefileは分離する。
COMPILER = g++
# COMPILER = clang++

# 標準的なコンパイルオプション
CFLAGS   =  -std=c++14 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive

# リンク時最適化。これをつけるとmsys2環境だとセグフォで落ちる。
# 落ちない環境なら有効にしたほうが速い実行ファイルが生成されるはず。
# CFLAGS += -flto

# CFLAGSとLDFLAGSの両方で-fltoを指定する必要がある。
# cf. https://gcc.gnu.org/onlinedocs/gcc-6.3.0/gcc/Optimize-Options.html#Optimize-Options
LTOFLAGS = -flto

# wstringを使うためにこのシンボル定義が必要。
CFLAGS  += -DUNICODE

# stripの指示。(実行ファイルのサイズを小さく)
CFLAGS  += -Wl,-s

# mingw64では-D_WIN64,-D_WIN32は環境に応じて自動で設定されるので指定すべきではない。
# CFLAGS += -D_WIN64

# これを指定していると、各CPU向けの実行ファイルが生成されないので指定すべきではない。
# CFLAGS   += -march=native

# デバッグ情報を付加
# CFLAGS  += -g3 -ggdb

# OpenMPを使うときにCFLAGSとして指定するオプション
# ※ 学習部ではOpenMpを用いるので、学習用のビルドではこのオプションを指定する。
OPENMP   = -fopenmp

LDFLAGS  = -lpthread
LIBS     =
INCLUDE  = # -I../include

ifeq ($(OS),Windows_NT)
  CFLAGS += -Wl,--stack,100000000
  LDFLAGS += -static
  TARGET = YaneuraOu-by-gcc.exe
else
  CFLAGS += -D_LINUX
  TARGET = YaneuraOu-by-gcc
endif

# clang用にCFLAGSなどを変更
# あとでclang用のMakefileは分離する。
ifeq ($(COMPILER),clang++)
  # gccと共通
  CFLAGS  = -std=c++14 -fno-exceptions -fno-rtti -Wextra
  CFLAGS += -Ofast -MMD -MP -fpermissive -DUNICODE -DNO_EXCEPTIONS

  # lib  // static linkの仕方わからん。
  CFLAGS += -stdlib=libstdc++

  # 関数の引数が関数本体で使われていないときに警告出るのうざすぎるので抑制。
  CFLAGS += -Wno-unused-parameter

  # msys2用のclangならば-fltoつけてもセグフォで落ちないらしい。
  # CFLAGS += -flto
  # →　LDFLAGSのほうで指定すれば十分？

  LDFLAGS  = -lpthread -Wl,-s -Wl,--stack,100000000

  # msys2用のclangでは、OpenMP使えない？インスコ仕方がわからん。

endif

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
	extra/move_picker/move_picker_2016Q2.cpp                                   \
	extra/move_picker/move_picker_2016Q3.cpp                                   \
	extra/move_picker/move_picker_2017Q2.cpp                                   \
	extra/benchmark.cpp                                                        \
	extra/test_cmd.cpp                                                         \
	extra/timeman.cpp                                                          \
	extra/see.cpp                                                              \
	extra/sfen_packer.cpp                                                      \
	eval/evaluate_bona_piece.cpp                                               \
	eval/evaluate_kpp.cpp                                                      \
	eval/evaluate_kppt.cpp                                                     \
	eval/evaluate_material.cpp                                                 \
	eval/evaluate_no_eval.cpp                                                  \
	eval/progress.cpp                                                          \
	engine/user-engine/user-search.cpp                                         \
	engine/random-player-engine/random-player-search.cpp                       \
	engine/help-mate-engine/help-mate-search.cpp                               \
	engine/nano-engine/nano-search.cpp                                         \
	engine/nano-plus-engine/nano-plus-search.cpp                               \
	engine/mini-engine/mini-search.cpp                                         \
	engine/classic-engine/classic-search.cpp                                   \
	engine/classic-tce-engine/classic-tce-search.cpp                           \
	engine/2016-mid-engine/2016-mid-search.cpp                                 \
	engine/2016-late-engine/2016-late-search.cpp                               \
	engine/2017-early-engine/2017-early-search.cpp                             \
	engine/check-shogi-engine/check-shogi.cpp                                  \
	engine/must-capture-shogi-engine/must-capture-shogi.cpp                    \
	learn/learner.cpp                                                          \
	learn/learning_tools.cpp                                                   \
	learn/multi_think.cpp                                                      \
	learn/evaluate_kppt_learn.cpp   
																               
#ifeq ($(YANEURAOU_EDITION),YANEURAOU_2017_GOKU_ENGINE)				           
#	SOURCES += engine/2017-goku-engine/2017-goku-search.cpp                    \
#				eval/evaluate_ppap.cpp                                         
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

tournament:
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -DUSE_MAKEFILE -D$(YANEURAOU_EDITION) -DUSE_AVX2 -mbmi2 -mavx2 -DFOR_TOURNAMENT -march=corei7-avx' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

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

#　とりあえずPGOはavx2専用
prof:
	$(MAKE) CFLAGS='$(CFLAGS) -pg' avx2

profgen:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-generate -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-generate -lgcov' avx2

profuse:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-use -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-use -lgcov -flto' avx2

pgo:
	$(MAKE) profgen
	@./$(TARGET) bench , quit
	@touch $(SOURCES)
	$(MAKE) profuse

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET) ${OBJECTS:.o=.gcda}

-include $(DEPENDS)
