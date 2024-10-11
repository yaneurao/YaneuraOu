# Copyright (C) 2009 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

CPPFLAGS := -DTARGET_ARCH="$(TARGET_ARCH_ABI)"

# example: (default build target)
# $ ndk-build

# example: EVAL_KPPT
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_KPPT

# example: EVAL_KPP_KKPT
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_KPP_KKPT

# example: EVAL_MATERIAL (MaterialLv1)
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_MATERIAL

# example: EVAL_NNUE_HALFKP_256x2_32_32 (2018 T.N.K.)
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE

# example: EVAL_NNUE_HALFKP_VM_256X2_32_32
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32

# example: EVAL_NNUE_HALFKPE9_256x2_32_32
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_HALFKPE9

# example: EVAL_NNUE_K_P_256x2_32_32
# $ ndk-build YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_KP256

# example: MATE_ENGINE (tanuki_MATE)
# $ ndk-build YANEURAOU_EDITION=MATE_ENGINE

YANEURAOU_EDITION := YANEURAOU_ENGINE_NNUE
#YANEURAOU_EDITION := YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32
#YANEURAOU_EDITION := YANEURAOU_ENGINE_NNUE_HALFKPE9
#YANEURAOU_EDITION := YANEURAOU_ENGINE_NNUE_KP256
#YANEURAOU_EDITION := YANEURAOU_ENGINE_KPPT
#YANEURAOU_EDITION := YANEURAOU_ENGINE_KPP_KKPT
#YANEURAOU_EDITION := YANEURAOU_ENGINE_MATERIAL
#YANEURAOU_EDITION := MATE_ENGINE
#YANEURAOU_EDITION := USER_ENGINE

# やねうら王のcluster機能を使いたいなら、これもdefineする。(define if you want to use YO-cluster)
YO_CLUSTER = OFF
#YO_CLUSTER = ON

# エンジンの表示名 (engine displayname)
# ("usi"コマンドに対して出力される)
#ENGINE_NAME :=

# developing branch // 現状、非公開 (currently private)
# dev : 開発中のbranchならdevと指定する (developing branch) :
# abe : abe
#ENGINE_BRANCH := dev

# makeするときにCPPFLAGSを追加で指定したいときはこれを用いる。
EXTRA_CPPFLAGS =

# YANEURAOU_EDITION = YANEURAOU_ENGINE_MATERIALのときに指定できる、評価関数の通し番号
# 001 : 普通の駒得のみの評価関数
# 002 : …
# cf.【連載】評価関数を作ってみよう！その1 : http://yaneuraou.yaneu.com/2020/11/17/make-evaluate-function/
MATERIAL_LEVEL = 1

# NNUE評価関数を実行ファイルに内蔵する。 `eval/nnue/embedded_nnue.cpp` が別途必要。
EVAL_EMBEDDING = OFF
# EVAL_EMBEDDING = ON

CPPFLAGS += $(EXTRA_CPPFLAGS)

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPPT)
  CPPFLAGS += -DUSE_MAKEFILE -DYANEURAOU_ENGINE_KPPT
  ENGINE_NAME := YaneuraOu_KPPT
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPP_KKPT)
  CPPFLAGS += -DUSE_MAKEFILE -DYANEURAOU_ENGINE_KPP_KKPT
  ENGINE_NAME := YaneuraOu_KPP_KKPT
endif

ifeq ($(findstring YANEURAOU_ENGINE_MATERIAL,$(YANEURAOU_EDITION)),YANEURAOU_ENGINE_MATERIAL)
  CPPFLAGS += -DUSE_MAKEFILE -DYANEURAOU_ENGINE_MATERIAL
  ENGINE_NAME := YaneuraOu_MaterialLv$(MATERIAL_LEVEL)
endif

ifeq ($(findstring YANEURAOU_ENGINE_NNUE,$(YANEURAOU_EDITION)),YANEURAOU_ENGINE_NNUE)
  CPPFLAGS += -DUSE_MAKEFILE -DYANEURAOU_ENGINE_NNUE
  ENGINE_NAME := YaneuraOu_NNUE
  ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_KP256)
    ENGINE_NAME := YaneuraOu_NNUE_KP256
    CPPFLAGS += -DEVAL_NNUE_KP256
  else
    ifeq ($(NNUE_EVAL_ARCH),KP256)
      ENGINE_NAME := YaneuraOu_NNUE_KP256
      CPPFLAGS += -DEVAL_NNUE_KP256
    endif
  endif
  ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_HALFKPE9)
    ENGINE_NAME := YaneuraOu_NNUE_HALFKPE9
    CPPFLAGS += -DEVAL_NNUE_HALFKPE9
  else
    ifeq ($(NNUE_EVAL_ARCH),HALFKPE9)
      ENGINE_NAME := YaneuraOu_NNUE_HALFKPE9
      CPPFLAGS += -DEVAL_NNUE_HALFKPE9
    endif
  endif
  ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_NNUE_HALFKPE9)
    ENGINE_NAME := YaneuraOu_NNUE_HALFKP_VM
    CPPFLAGS += -DEVAL_NNUE_HALFKP_VM_256X2_32_32
  else
    ifeq ($(NNUE_EVAL_ARCH),EVAL_NNUE_HALFKP_VM_256X2_32_32)
      ENGINE_NAME := YaneuraOu_NNUE_HALFKP_VM
      CPPFLAGS += -DEVAL_NNUE_HALFKP_VM_256X2_32_32
    endif
  endif
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_MATE_ENGINE)
  CPPFLAGS += -DUSE_MAKEFILE -DYANEURAOU_MATE_ENGINE
  ENGINE_NAME := YaneuraOu_MATE
endif

ifeq ($(YANEURAOU_EDITION),TANUKI_MATE_ENGINE)
  CPPFLAGS += -DUSE_MAKEFILE -DTANUKI_MATE_ENGINE
  ENGINE_NAME := tanuki_MATE
endif

ifeq ($(YANEURAOU_EDITION),USER_ENGINE)
  CPPFLAGS += -DUSE_MAKEFILE -DUSER_ENGINE
  ENGINE_NAME := YaneuraOu_USER
endif

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
  CPPFLAGS += -DIS_64BIT -DUSE_NEON -mfpu=neon
  LOCAL_ARM_NEON := true
endif

ifeq ($(TARGET_ARCH_ABI),x86_64)
  CPPFLAGS += -DIS_64BIT -DUSE_SSE42 -msse4.2
endif

ifeq ($(TARGET_ARCH_ABI),x86)
  CPPFLAGS += -DNO_SSE
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  CPPFLAGS += -DUSE_NEON -mfpu=neon
  LOCAL_ARM_NEON := true
endif

LOCAL_SRC_FILES := \
  ../source/main.cpp                                                   \
  ../source/types.cpp                                                  \
  ../source/bitboard.cpp                                               \
  ../source/misc.cpp                                                   \
  ../source/movegen.cpp                                                \
  ../source/position.cpp                                               \
  ../source/usi.cpp                                                    \
  ../source/usi_option.cpp                                             \
  ../source/thread.cpp                                                 \
  ../source/tt.cpp                                                     \
  ../source/movepick.cpp                                               \
  ../source/timeman.cpp                                                \
  ../source/memory.cpp                                                 \
  ../source/book/apery_book.cpp                                        \
  ../source/book/book.cpp                                              \
  ../source/extra/bitop.cpp                                            \
  ../source/extra/long_effect.cpp                                      \
  ../source/extra/sfen_packer.cpp                                      \
  ../source/extra/super_sort.cpp                                       \
  ../source/mate/mate.cpp                                              \
  ../source/mate/mate1ply_without_effect.cpp                           \
  ../source/mate/mate1ply_with_effect.cpp                              \
  ../source/mate/mate_solver.cpp                                       \
  ../source/eval/evaluate_bona_piece.cpp                               \
  ../source/eval/evaluate.cpp                                          \
  ../source/eval/evaluate_io.cpp                                       \
  ../source/eval/evaluate_mir_inv_tools.cpp                            \
  ../source/eval/material/evaluate_material.cpp                        \
  ../source/testcmd/benchmark.cpp                                      \
  ../source/testcmd/mate_test_cmd.cpp                                  \
  ../source/testcmd/normal_test_cmd.cpp                                \
  ../source/testcmd/unit_test.cpp

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPPT)
LOCAL_SRC_FILES += \
  ../source/eval/kppt/evaluate_kppt.cpp                                \
  ../source/eval/kppt/evaluate_kppt_learner.cpp                        \
  ../source/engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_KPP_KKPT)
LOCAL_SRC_FILES += \
  ../source/eval/kppt/evaluate_kppt.cpp                                \
  ../source/eval/kpp_kkpt/evaluate_kpp_kkpt.cpp                        \
  ../source/eval/kpp_kkpt/evaluate_kpp_kkpt_learner.cpp                \
  ../source/engine/yaneuraou-engine/yaneuraou-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_ENGINE_MATERIAL)
LOCAL_SRC_FILES += \
  ../source/engine/yaneuraou-engine/yaneuraou-search.cpp

CPPFLAGS += -DMATERIAL_LEVEL=$(MATERIAL_LEVEL)
endif

ifeq ($(findstring YANEURAOU_ENGINE_NNUE,$(YANEURAOU_EDITION)),YANEURAOU_ENGINE_NNUE)
LOCAL_SRC_FILES += \
  ../source/eval/nnue/evaluate_nnue.cpp                                \
  ../source/eval/nnue/evaluate_nnue_learner.cpp                        \
  ../source/eval/nnue/nnue_test_command.cpp                            \
  ../source/eval/nnue/features/k.cpp                                   \
  ../source/eval/nnue/features/p.cpp                                   \
  ../source/eval/nnue/features/half_kp.cpp                             \
  ../source/eval/nnue/features/half_kp_vm.cpp                          \
  ../source/eval/nnue/features/half_relative_kp.cpp                    \
  ../source/eval/nnue/features/half_kpe9.cpp                           \
  ../source/eval/nnue/features/pe9.cpp                                 \
  ../source/engine/yaneuraou-engine/yaneuraou-search.cpp
	ifeq ($(EVAL_EMBEDDING),ON)
		LOCAL_SRC_FILES += \
			../source/eval/nnue/embedded_nnue.cpp
		CPPFLAGS += -DEVAL_EMBEDDING
	endif
endif

ifeq ($(YANEURAOU_EDITION),TANUKI_MATE_ENGINE)
LOCAL_SRC_FILES += \
  ../source/engine/tanuki-mate-engine/tanuki-mate-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),YANEURAOU_MATE_ENGINE)
LOCAL_SRC_FILES += \
  ../source/engine/yaneuraou-mate-engine/yaneuraou-mate-search.cpp
endif

ifeq ($(YANEURAOU_EDITION),USER_ENGINE)
LOCAL_SRC_FILES += \
  ../source/engine/user-engine/user-search.cpp
endif

ifneq ($(ENGINE_NAME),)
CPPFLAGS += -DENGINE_NAME_FROM_MAKEFILE=$(ENGINE_NAME)
endif

# cluster
ifeq ($(YO_CLUSTER),ON)
	LOCAL_SRC_FILES += \
		../source/engine/yo-cluster/ClusterCommon.cpp                      \
		../source/engine/yo-cluster/EngineNegotiator.cpp                   \
		../source/engine/yo-cluster/ProcessNegotiator.cpp                  \
		../source/engine/yo-cluster/ClusterObserver.cpp                    \
		../source/engine/yo-cluster/ClusterStrategy.cpp
	CPPFLAGS += -DUSE_YO_CLUSTER
endif

# 開発用branch
ifeq ($(findstring dev,$(ENGINE_BRANCH)),dev)
CPPFLAGS += -DDEV_BRANCH
endif

# abe
ifeq ($(findstring abe,$(ENGINE_BRANCH)),abe)
CPPFLAGS += -DPV_OUTPUT_DRAW_ONLY -DFORCE_BIND_THIS_THREAD
endif


LOCAL_MODULE    := $(ENGINE_NAME)_$(TARGET_ARCH_ABI)
LOCAL_CXXFLAGS  := -std=c++17 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive -D__STDINT_MACROS -D__STDC_LIMIT_MACROS $(CPPFLAGS)
LOCAL_CXXFLAGS += -DNDEBUG -fPIE -Wno-unused-parameter -flto
LOCAL_LDFLAGS += -fPIE -pie -flto
LOCAL_LDLIBS =
LOCAL_C_INCLUDES :=
#LOCAL_CPP_FEATURES += exceptions rtti
#LOCAL_STATIC_LIBRARIES    := -lpthread

include $(BUILD_EXECUTABLE)
