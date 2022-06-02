
# === ビルドターゲット (build target) ===

# normal     : 通常使用用
# evallearn  : 教師局面からの学習用
# tournament : 大会で使う用

# === ビルドオプション (build options) ===

# やねうら王の使いたいエディションを指定する (select YaneuraOu Edition)
# YANEURAOU_ENGINE_NNUE          : NNUE型評価関数(halfKP256),標準NNUE型
# YANEURAOU_ENGINE_NNUE_KP256    : NNUE型評価関数(KP256)
# YANEURAOU_ENGINE_NNUE_HALFKPE9 : NNUE型評価関数(halfKPE9)
# YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32 : NNUE型評価関数(halfkp_vm_256x2-32-32)
# YANEURAOU_ENGINE_KPPT          : KPPT型評価関数
# YANEURAOU_ENGINE_KPP_KKPT      : KPP_KKPT型評価関数
# YANEURAOU_ENGINE_MATERIAL      : 駒得のみの評価関数(複数変種あり。MATERIAL_LEVELで選択できる)
# TANUKI_MATE_ENGINE             : tanuki- 詰め将棋solver  (2017/05/06～)
# YANEURAOU_MATE_ENGINE          : やねうら王 詰将棋solver (2020/12/29～)
# USER_ENGINE                    : USER定義エンジン

YANEURAOU_EDITION = YANEURAOU_ENGINE_NNUE
#YANEURAOU_EDITION = YANEURAOU_ENGINE_NNUE_KP256
#YANEURAOU_EDITION = YANEURAOU_ENGINE_NNUE_HALFKPE9
#YANEURAOU_EDITION = YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32
#YANEURAOU_EDITION = YANEURAOU_ENGINE_KPPT
#YANEURAOU_EDITION = YANEURAOU_ENGINE_KPP_KKPT
#YANEURAOU_EDITION = YANEURAOU_ENGINE_MATERIAL
#YANEURAOU_EDITION = TANUKI_MATE_ENGINE
#YANEURAOU_EDITION = YANEURAOU_MATE_ENGINE
#YANEURAOU_EDITION = USER_ENGINE

# やねうら王のcluster機能を使いたいなら、これもdefineする。(define if you want to use YO-cluster)
YO_CLUSTER = OFF
#YO_CLUSTER = ON


# ターゲットCPU (target cpu)

# 利用できるCPU拡張命令を指定する。
#
# AMDのRyzen(Zen/Zen2)はZEN1/ZEN2を選択するとBMI2命令が使わずに速くなる。(10%程度高速化)
# Ryzen Zen3シリーズはZEN3を選択。(clang/llvm 13.0.0以降、gcc 11.2以降)
#
# ARM系はGraviton2ならGRAVITON2、これ以外ならOTHERを指定する。
# Windows x86(32bit環境)用はNO_SSEを指定する。
#
# AVX-512対応(サーバー向けSkylake以降)ならAVX512を指定する。
# AVX-512対応でさらにVNNI命令対応(Cascade Lake以降)なら、AVX512VNNIを指定する。


# --- Intel/AMD系 (x86/x64 Platform)
#TARGET_CPU = AVX512VNNI
#TARGET_CPU = AVX512
TARGET_CPU = AVX2
#TARGET_CPU = SSE42
#TARGET_CPU = SSE41
#TARGET_CPU = SSSE3
#TARGET_CPU = SSE2
#TARGET_CPU = NO_SSE
# --- AMD Ryzen
#TARGET_CPU = ZEN1
#TARGET_CPU = ZEN2
#TARGET_CPU = ZEN3
# -- ARMとか
#TARGET_CPU = OTHER
#TARGET_CPU = GRAVITON2
#TARGET_CPU = WASM


# デバッガーを使用するか (debugger)
DEBUG = OFF
#DEBUG = ON


# 使用するコンパイラ (compiler)
# ※ clangでコンパイルしたほうがgccより数%速いっぽい。
#COMPILER = g++
COMPILER = clang++
#COMPILER = em++


# エンジンの表示名 (engine displayname)
# ("usi"コマンドに対して出力される)
#ENGINE_NAME =


# developing branch // 現状、非公開 (currently private)
# dev : 開発中のbranchならdevと指定する (developing branch) :
# abe : abe
#ENGINE_BRANCH = dev


# 標準的なコンパイルオプション (standard compile options)
CPPFLAGS   = -std=c++17 -fno-exceptions -fno-rtti -Wextra -MMD -MP -fpermissive
NVCC_CPPFLAGS = -std=c++17 -x cu -maxrregcount=0 --machine 64 --compile -cudart static -DNDEBUG -Xcompiler "-Ofast"
WCPPFLAGS  =
LDFLAGS  = $(EXTRA_LDFLAGS)
LIBS     =
INCLUDE  = # -I../include

# makeするときにCPPFLAGSを追加で指定したいときはこれを用いる。
EXTRA_CPPFLAGS =

# YANEURAOU_EDITION = YANEURAOU_ENGINE_MATERIALのときに指定できる、評価関数の通し番号
#  1 : 普通の駒得のみの評価関数
#  2 : …
# cf.【連載】評価関数を作ってみよう！その1 : http://yaneuraou.yaneu.com/2020/11/17/make-evaluate-function/
MATERIAL_LEVEL = 1

# evallearnで使用するBLASの種類を指定する。
BLAS = OPENBLAS
# BLAS = BLIS

# NNUE評価関数を実行ファイルに内蔵する。 `eval/nnue/embedded_nnue.cpp` が別途必要。
EVAL_EMBEDDING = OFF
# EVAL_EMBEDDING = ON

# === implementations ===

# 以下では、
#   ifneq (,$(findstring A,B))
# というイデオムが出てくる。
#
# findstringは文字列Bに文字列Aが含まれているかを判定し、含まれていればAを返す。含まれていなければ空の文字列を返す。
# なので、AがBに含まれている場合は、findstringはAを返し、ifneq(if not equal)でそれが空かどうかを判定した場合、空ではないから、ifneqは成立する。
# つまり、このイデオムは、「BにAが含まれているならば、以下を実行してください。」という意味である。
#
# ※ ifeq (A,B) は「AとBが等しいならば、以下を実行してください。」という意味。紛らわしいので気をつけよう。


# em++ では -Ofast の指定は対応していない
ifneq (,$(findstring em++,$(COMPILER)))
	CPPFLAGS += -O2
else
	CPPFLAGS += -Ofast
endif

# デバッガーを使わないなら、NDEBUGをdefineする。
ifeq ($(DEBUG),OFF)
	CPPFLAGS += -DNDEBUG
endif

# clang用にCPPFLAGSなどを変更
ifneq (,$(findstring clang++,$(COMPILER)))
	# stdlib
	# CPPFLAGS += -stdlib=libstdc++

	# C++17のfilesystem
	# LDFLAGS += -lstdc++fs

	# 関数の引数が関数本体で使われていないときに警告出るのうざすぎるので抑制。
	CPPFLAGS += -Wno-unused-parameter

	# static リンクを行う際に __cxa_guard_acquire __cxa_guard_release の生成を抑止
	#   undefined reference to `__imp___cxa_guard_acquire'
	#   undefined reference to `__imp___cxa_guard_release'
	# static 変数を初期化したかを pthread_mutex_t でロックを取って確認し、
	# 最初の実行なら初期化するスレッドセーフなコードを生成するためのもの。
	# → 本当に消してしまっても大丈夫か？
	WCPPFLAGS += -fno-threadsafe-statics

else
	ifneq (,$(findstring g++,$(COMPILER)))
		# mingw g++ で AVX512 向けビルドを行おうとするとエラーになる問題の回避
		# https://stackoverflow.com/questions/43152633/invalid-register-for-seh-savexmm-in-cygwin
		# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782
		WCPPFLAGS += -fno-asynchronous-unwind-tables

		# C++17のfilesystem
		# LDFLAGS += -lstdc++fs
	endif
	ifneq (,$(findstring em++,$(COMPILER)))
		EM_EXPORT_NAME=YaneuraOu
		EM_INITIAL_MEMORY_SIZE=$(shell if [ $(MATERIAL_LEVEL) -ge 5 ]; then echo 385875968; else echo 138412032; fi)
		CPPFLAGS += -Wno-unused-parameter
		CPPFLAGS += -DUSE_WASM_SIMD -msimd128
		CPPFLAGS += -DUSE_SSE42 -msse4.2
		CPPFLAGS += -march=wasm32
		# CPPFLAGS += -march=corei7
		CPPFLAGS += -s USE_PTHREADS=1
		CPPFLAGS += -s STRICT=1
		LDFLAGS += --pre-js wasm_pre.js
		# LDFLAGS += --preload-file eval/nn.bin
		LDFLAGS += -s MODULARIZE=1 -s EXPORT_NAME="$(EM_EXPORT_NAME)" -s ENVIRONMENT=web,worker,node
		LDFLAGS += -s PTHREAD_POOL_SIZE=32
		# -s PTHREAD_POOL_SIZE_STRICT=32
		LDFLAGS += -s FILESYSTEM=1 -s EXIT_RUNTIME=0 -s "EXPORTED_RUNTIME_METHODS=['FS','ccall']"
		LDFLAGS += -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=$(EM_INITIAL_MEMORY_SIZE) -s MAXIMUM_MEMORY=4294967296
		LDFLAGS += -s TOTAL_STACK=67108864
		LDFLAGS += -s LLD_REPORT_UNDEFINED
		# LDFLAGS += -s PROXY_TO_PTHREAD
		# LDFLAGS += -s ALLOW_UNIMPLEMENTED_SYSCALLS
		# LDFLAGS += -s ASSERTIONS
		LDFLAGS += -s --closure 1
	endif
endif

TARGETDIR = .
ifneq (,$(findstring em++,$(COMPILER)))
	TARGET = $(TARGETDIR)/yaneuraou.js
else
	ifeq ($(OS),Windows_NT)
		CPPFLAGS += $(WCPPFLAGS)
		LDFLAGS += -static -Wl,--stack,25000000
		TARGET = $(TARGETDIR)/YaneuraOu-by-gcc.exe
	else
		CPPFLAGS += -D_LINUX
		TARGET = $(TARGETDIR)/YaneuraOu-by-gcc
	endif
endif

# リンク時最適化。
# CPPFLAGSとLDFLAGSの両方で-fltoを指定する必要がある。
# cf. https://gcc.gnu.org/onlinedocs/gcc-6.3.0/gcc/Optimize-Options.html#Optimize-Options
LTOFLAGS = -flto

# MSYS2 clang++ で -flto 指定時にリンクを行えない問題の回避
ifneq (,$(findstring clang++,$(COMPILER)))
	ifeq ($(OS),Windows_NT)
		LDFLAGS += -fuse-ld=lld
	endif
endif

# wstringを使うためにこのシンボル定義が必要。
CPPFLAGS  += -DUNICODE

# em++ では strip: -Wl,-s の指定はしない
ifeq (,$(findstring em++,$(COMPILER)))
# stripの指示。(実行ファイルのサイズを小さく)
LDFLAGS += -Wl,-s
endif

# mingw64では-D_WIN64,-D_WIN32は環境に応じて自動で設定されるので指定すべきではない。
# CPPFLAGS += -D_WIN64

# これを指定していると、各CPU向けの実行ファイルが生成されないので指定すべきではない。
# CPPFLAGS   += -march=native

# デバッグ情報を付加
# CPPFLAGS  += -g3 -ggdb

# OpenMPを使うときにCPPFLAGSとして指定するオプション
# ※ 学習部ではOpenMpを用いるので、学習用のビルドではこのオプションを指定する。
# clangでOPENMPを有効にしてビルドする方法については、解説.txtを参照のこと。


ifneq (,$(findstring g++,$(COMPILER)))
	OPENMP   = -fopenmp
	OPENMP_LDFLAGS =
endif
ifneq (,$(findstring clang++,$(COMPILER)))
		ifeq ($(MSYSTEM),MINGW64)
		# MSYS2 MINGW64 env
		# libgompを指定した場合、ビルドは通るがOpenMPは無効化される？
		OPENMP = -fopenmp=libgomp
		OPENMP_LDFLAGS =
	else
		ifneq (,$(findstring w64-mingw32,$(COMPILER)))
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
ifneq (,$(findstring YANEURAOU_ENGINE_NNUE,$(YANEURAOU_EDITION)))
	ifeq ($(BLAS),OPENBLAS)
		BLAS_CPPFLAGS = -DUSE_BLAS
		BLAS_LDFLAGS = -lopenblas
		ifeq ($(MSYSTEM),MINGW64)
			BLAS_CPPFLAGS += -I$(shell cygpath -aw /mingw64/include/OpenBLAS)
		endif
	endif

	ifeq ($(BLAS),BLIS)
		BLAS_CPPFLAGS = -DUSE_BLAS
		BLAS_CPPFLAGS += -I$(shell cygpath -aw /usr/local/include/blis)
		BLAS_LDFLAGS = -lblis.dll
		BLAS_LDFLAGS += -L$(shell cygpath -aw /usr/local/lib)
	endif
endif

CPPFLAGS += -DNO_EXCEPTIONS
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
	book/book.cpp                                                              \
	book/apery_book.cpp                                                        \
	extra/bitop.cpp                                                            \
	extra/long_effect.cpp                                                      \
	extra/sfen_packer.cpp                                                      \
	extra/super_sort.cpp                                                       \
	mate/mate.cpp                                                              \
	mate/mate1ply_without_effect.cpp                                           \
	mate/mate1ply_with_effect.cpp                                              \
	mate/mate_solver.cpp                                                       \
	eval/evaluate_bona_piece.cpp                                               \
	eval/evaluate.cpp                                                          \
	eval/evaluate_io.cpp                                                       \
	eval/evaluate_mir_inv_tools.cpp                                            \
	eval/material/evaluate_material.cpp                                        \
	testcmd/unit_test.cpp                                                      \
	testcmd/mate_test_cmd.cpp                                                  \
	testcmd/normal_test_cmd.cpp                                                \
	testcmd/benchmark.cpp

CUDA_SOURCES =

OBJC_SOURCES =

#ifneq (,$(findstring evallearn,$@))
# →これこの時点で$@を取得できないのか…。なんぞこれ。

# 仕方ないので無条件で追加しておく。cppファイルのほうで、
#  if!defined(EVAL_LEARN)..とか書いてあるのでとりあえずいいや。

ifeq (,)
	SOURCES  +=                                                                \
		book/makebook.cpp                                                      \
		book/makebook2015.cpp                                                  \
		book/makebook2019.cpp                                                  \
		book/makebook2021.cpp                                                  \
		learn/learner.cpp                                                      \
		learn/learning_tools.cpp                                               \
		learn/multi_think.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPPT)
	SOURCES  +=                                                                \
		eval/kppt/evaluate_kppt.cpp                                            \
		eval/kppt/evaluate_kppt_learner.cpp                                    \
		engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPP_KKPT)
	SOURCES  +=                                                                \
		eval/kppt/evaluate_kppt.cpp                                            \
		eval/kpp_kkpt/evaluate_kpp_kkpt.cpp                                    \
		eval/kpp_kkpt/evaluate_kpp_kkpt_learner.cpp                            \
		engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_MATERIAL)
	SOURCES  +=                                                                \
		engine/yaneuraou-engine/yaneuraou-search.cpp

	CPPFLAGS += -DMATERIAL_LEVEL=$(MATERIAL_LEVEL)
endif

ifneq (,$(findstring YANEURAOU_ENGINE_DEEP,$(YANEURAOU_EDITION)))
	ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_DEEP)
	else
		# YANEURAOU_EDITIONがYANEURAOU_ENGINE_DEEPの後ろに文字が入っている名前なら、
		# とりあえず"YANEURAOU_ENGINE_DEEP"というシンボルを定義してやる。
		CPPFLAGS += -DYANEURAOU_ENGINE_DEEP

		ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_DEEP_TENSOR_RT)
			CPPFLAGS += -DTENSOR_RT
			NVCC_CPPFLAGS += -DTENSOR_RT
			LDFLAGS += -lnvinfer -lnvonnxparser -lcudart
			# CPPFLAGS += -I/usr/local/cuda/include
			# LDFLAGS += -L/usr/local/cuda/lib64
			# LDFLAGS += -L/usr/local/cuda/lib64/stubs
			CUDA_SOURCES += eval/deep/unpack.cu

		else ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_DEEP_ORT_CPU)
			CPPFLAGS += -DONNXRUNTIME -DORT_NO_EXCEPTIONS -DORT_CPU
			LDFLAGS += -lonnxruntime

		else ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_DEEP_ORT_DML)
			CPPFLAGS += -DONNXRUNTIME -DORT_NO_EXCEPTIONS -DORT_DML
			LDFLAGS += -lonnxruntime

		else ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_DEEP_ORT_TRT)
			CPPFLAGS += -DONNXRUNTIME -DORT_NO_EXCEPTIONS -DORT_TRT
			LDFLAGS += -lonnxruntime

		else ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_DEEP_COREML)
			CPPFLAGS += -DCOREML
			LDFLAGS += -framework Foundation -framework CoreML
			OBJC_SOURCES += eval/deep/nn_coreml.mm

		endif
	endif

	# 大して大きなファイルではないので全部追加する。
	SOURCES += \
		eval/deep/nn_types.cpp                                          \
		eval/deep/nn.cpp                                                \
		eval/deep/nn_onnx_runtime.cpp                                   \
		eval/deep/nn_tensorrt.cpp                                       \
		engine/dlshogi-engine/dlshogi_searcher.cpp                      \
		engine/dlshogi-engine/PrintInfo.cpp                             \
		engine/dlshogi-engine/UctSearch.cpp                             \
		engine/dlshogi-engine/Node.cpp                                  \
		engine/dlshogi-engine/YaneuraOu_dlshogi_bridge.cpp
endif

ifneq (,$(findstring YANEURAOU_ENGINE_NNUE,$(YANEURAOU_EDITION)))
	ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE)
	else
		# YANEURAOU_EDITIONがYANEURAOU_ENGINE_NNUEの後ろに文字が入っている名前なら、これは
		# 標準NNUE型の亜種なので、とりあえず"YANEURAOU_ENGINE_NNUE"というシンボルを定義してやる。
		CPPFLAGS += -DYANEURAOU_ENGINE_NNUE

		ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_KP256)
			CPPFLAGS += -DEVAL_NNUE_KP256

		else ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_HALFKPE9)
			CPPFLAGS += -DEVAL_NNUE_HALFKPE9

		else ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32)
			CPPFLAGS += -DEVAL_NNUE_HALFKP_VM_256X2_32_32

		endif
	endif

	# 大して大きなファイルではないので全部してしまう。
	SOURCES += \
		eval/nnue/evaluate_nnue.cpp                                     \
		eval/nnue/evaluate_nnue_learner.cpp                             \
		eval/nnue/nnue_test_command.cpp                                 \
		eval/nnue/features/k.cpp                                        \
		eval/nnue/features/p.cpp                                        \
		eval/nnue/features/half_kp.cpp                                  \
		eval/nnue/features/half_kp_vm.cpp                               \
		eval/nnue/features/half_relative_kp.cpp                         \
		eval/nnue/features/half_kpe9.cpp                                \
		eval/nnue/features/pe9.cpp                                      \
		engine/yaneuraou-engine/yaneuraou-search.cpp

	ifneq (,$(findstring em++,$(COMPILER)))
		SOURCES += \
			eval/nnue/wasm_simd.cpp
	endif
	ifeq ($(EVAL_EMBEDDING),ON)
		SOURCES += \
			eval/nnue/embedded_nnue.cpp
		CPPFLAGS += -DEVAL_EMBEDDING
	endif
endif


ifeq ($(YANEURAOU_EDITION),TANUKI_MATE_ENGINE)
	SOURCES += \
		engine/tanuki-mate-engine/tanuki-mate-search.cpp
endif


ifeq ($(YANEURAOU_EDITION),YANEURAOU_MATE_ENGINE)
	SOURCES += \
		engine/yaneuraou-mate-engine/yaneuraou-mate-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),USER_ENGINE)
	SOURCES += \
		engine/user-engine/user-search.cpp
endif


ifneq ($(ENGINE_NAME),)
	CPPFLAGS += -DENGINE_NAME_FROM_MAKEFILE=$(ENGINE_NAME)
endif

ifeq ($(YO_CLUSTER),ON)
	SOURCES += \
		engine/yo-cluster/ClusterCommon.cpp       \
		engine/yo-cluster/ProcessNegotiator.cpp   \
		engine/yo-cluster/EngineNegotiator.cpp    \
		engine/yo-cluster/ClusterObserver.cpp     \
		engine/yo-cluster/ClusterStrategy.cpp
	CPPFLAGS += -DUSE_YO_CLUSTER
endif


# 開発用branch
ifneq (,$(findstring dev,$(ENGINE_BRANCH)))
	# SuperSort使ってみよう。
	CPPFLAGS += -DDEV_BRANCH
endif

# abe
ifneq (,$(findstring abe,$(ENGINE_BRANCH)))
	CPPFLAGS += -DPV_OUTPUT_DRAW_ONLY -DFORCE_BIND_THIS_THREAD
endif


# CPU判別

# TARGET_CPUシンボルを定義してやる。
CPPFLAGS += -DTARGET_CPU=\"$(TARGET_CPU)\"

# AMD Ryzenシリーズ

# ZEN1/ZEN2はBMI2命令を使わずPEXTはSoftware Emulationしたほうが速い。
# さらに、そのコードを使う代わりにMagic Bitboardを使うことで高速化を図る。
# (USE_BMI2が定義されていない時、自動的にmagic bitboardが用いられる)
# ZEN3でPEXTの速度がまともになったらしいのでZEN3以降ではBMI2命令を使うようにする。

ifeq ($(TARGET_CPU),ZEN1)
	CPPFLAGS += -DUSE_AVX2 -mbmi -mno-bmi2 -mavx2 -march=znver1

else ifeq ($(TARGET_CPU),ZEN2)
	CPPFLAGS += -DUSE_AVX2 -mbmi -mno-bmi2 -mavx2 -march=znver2

else ifeq ($(TARGET_CPU),ZEN3)
	CPPFLAGS += -DUSE_AVX2 -DUSE_BMI2 -mbmi -mbmi2 -mavx2 -march=znver3

# それ以外は、AVX512,AVX2,SSE4.2,SSE4.1,SSE2のように利用できるCPU拡張命令で振り分ける。
# AVX2より上位のCPUなら普通は(Intel系なら)BMI2命令を使ったほうが速いので"USE_BMI2"を指定しておく。

# メモ) AVX512について
#   skylake     : -DUSE_AVX512 -DUSE_AVX512VLBWDQ -march=skylake-avx512
#   cascadelake : -DUSE_AVX512 -DUSE_AVX512VLBWDQ -DUSE_VNNI -march=cascadelake
#   icelake     : -DUSE_AVX512 -DUSE_AVX512VLBWDQ -DUSE_VNNI -DUSE_AVX512VBMI -DUSE_AVX512IFMA -USE_GFNI -march=icelake-client

else ifeq ($(TARGET_CPU),AVX512VNNI)

	CPPFLAGS += -DUSE_AVX512 -DUSE_BMI2 -DUSE_VNNI -march=cascadelake
	# NNUEのコード、USE_VNNIが指定されているとVNNI対応のコードになる。
	# cascadelakeとicelakeとの違いがあるのかは知らないので、cascadelakeのみでいいや。

else ifeq ($(TARGET_CPU),AVXVNNI)
	CPPFLAGS += -DUSE_AVX2 -DUSE_BMI2 -DUSE_VNNI -march=alderlake
	# NNUEのコード、USE_VNNIが指定されているとVNNI対応のコードになる。
	# Intel Alder Lake、Sapphire Rapids 以降追加の命令群。LLVM12, GCC11 以降。
	# AVXVNNI (AVX2VNNI という表記も有り?) は AVX512VNNIの256bit幅以下限定版。

else ifeq ($(TARGET_CPU),AVX512)

	CPPFLAGS += -DUSE_AVX512 -DUSE_BMI2 -march=skylake-avx512

else ifeq ($(TARGET_CPU),AVX2)
	CPPFLAGS += -DUSE_AVX2 -DUSE_BMI2 -mbmi -mbmi2 -mavx2 -march=corei7-avx

else ifeq ($(TARGET_CPU),SSE42)
	CPPFLAGS += -DUSE_SSE42 -msse4.2 -march=corei7

else ifeq ($(TARGET_CPU),SSE41)
	CPPFLAGS += -DUSE_SSE41 -msse4.1 -march=core2

else ifeq ($(TARGET_CPU),SSSE3)
	CPPFLAGS += -DUSE_SSSE3 -msse3 -march=core2

else ifeq ($(TARGET_CPU),SSE2)
	CPPFLAGS += -DUSE_SSE2 -msse2 -march=core2

else ifeq ($(TARGET_CPU),NO_SSE)
	# 32bit用。-m32は、MSYS2 MinGW-64だと無視されるので、
	# MinGW-64の32bit環境用でコンパイルする必要がある。
	CPPFLAGS += -DNO_SSE -m32 -march=pentium3
else ifeq ($(TARGET_CPU),GRAVITON2)
	# for Amazon Web Servece EC2, the Graviton2 CPU [M6g/M6gd, C6g/C6gd/C6gn, R6g/R6gd, T4g, X2gd] instances
	# https://github.com/aws/aws-graviton-getting-started/blob/main/c-c++.md
	CPPFLAGS += -DIS_64BIT -DUSE_NEON -march=armv8.2-a+fp16+rcpc+dotprod+crypto
else ifeq ($(TARGET_CPU),APPLEAVX2)
	CPPFLAGS += -DIS_64BIT -DUSE_AVX2 -DUSE_BMI2 -target x86_64-apple-macos11 -mbmi -mbmi2 -mavx2 -mpopcnt
else ifeq ($(TARGET_CPU),APPLESSE42)
	CPPFLAGS += -DIS_64BIT -DUSE_SSE42 -target x86_64-apple-macos11
else ifeq ($(TARGET_CPU),APPLEM1)
	CPPFLAGS += -DIS_64BIT -DUSE_NEON -target arm64-apple-macos11
else ifeq ($(TARGET_CPU),WASM)

else ifeq ($(TARGET_CPU),OTHER)
	CPPFLAGS += -DNO_SSE

endif

# YANEURAOU_EDITION , CPUターゲット名 などのシンボルをコンパイル時に定義してやる。
CPPFLAGS += -D$(YANEURAOU_EDITION) -DTARGET_CPU=\"$(TARGET_CPU)\" $(EXTRA_CPPFLAGS)
NVCC_CPPFLAGS += -D$(YANEURAOU_EDITION) -DTARGET_CPU=\"$(TARGET_CPU)\"

OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
CUDA_OBJECTS = $(addprefix $(OBJDIR)/, $(CUDA_SOURCES:.cu=.o))
OBJC_OBJECTS = $(addprefix $(OBJDIR)/, $(OBJC_SOURCES:.mm=.o))
DEPENDS  = $(OBJECTS:.o=.d)

all: clean $(TARGET)
.PHONY : all normal evallearn tournament prof profgen profuse pgo clean

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS) $(OBJC_OBJECTS)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(CPPFLAGS) $(LIBS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJDIR)/%.o: %.cu
	nvcc $(NVCC_CPPFLAGS) $(INCLUDE) -o $@ $<

$(OBJDIR)/%.o: %.mm
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) $(CPPFLAGS) -fobjc-arc $(INCLUDE) -o $@ -c $<

# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html


# ビルドターゲット

# 通常使用。
normal:
	$(MAKE) CPPFLAGS='$(CPPFLAGS) $(LTOFLAGS)' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

# 学習用。openmp , openblasなどを有効にする。
evallearn:
	$(MAKE) CPPFLAGS='$(CPPFLAGS) $(LTOFLAGS) $(OPENMP) $(BLAS_CPPFLAGS) -DEVAL_LEARN' LDFLAGS='$(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS) $(LTOFLAGS)' $(TARGET)

# トーナメント用
tournament:
	$(MAKE) CPPFLAGS='$(CPPFLAGS) $(LTOFLAGS) -DFOR_TOURNAMENT' LDFLAGS='$(LDFLAGS) $(LTOFLAGS)' $(TARGET)

#　とりあえずPGOはAVX2とSSE4.2専用
prof:
	$(MAKE) CPPFLAGS='$(CPPFLAGS) -pg' tournament

profgen:
	$(MAKE) CPPFLAGS='$(CPPFLAGS) -fprofile-generate -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-generate -lgcov'

profuse:
	$(MAKE) CPPFLAGS='$(CPPFLAGS) $(LTOFLAGS) -fprofile-use -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-use -lgcov $(LTOFLAGS)'

pgo:
	$(MAKE) profgen
	@./$(TARGET) EvalDir ../build/eval , bench , quit
	@touch $(SOURCES)
	$(MAKE) profuse

clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(OBJC_OBJECTS) $(DEPENDS) $(TARGET) ${OBJECTS:.o=.gcda}

-include $(DEPENDS)
