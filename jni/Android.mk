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

ARCH_DEF := -DTARGET_ARCH="$(TARGET_ARCH_ABI)"

# example: (default build target)
# $ ndk-build

# example: 2018 Otafuku (EVAL_KPPT)
# $ ndk-build ENGINE_TARGET=YANEURAOU_2018_OTAFUKU_ENGINE_KPPT

# example: 2018 Otafuku (EVAL_KPP_KKPT)
# $ ndk-build ENGINE_TARGET=YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT

# example: 2018 Otafuku (EVAL_MATERIAL)
# $ ndk-build ENGINE_TARGET=YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL

# example: 2018 T.N.K. (EVAL_NNUE_HALFKP_256x2_32_32)
# $ ndk-build ENGINE_TARGET=YANEURAOU_2018_TNK_ENGINE

# example: 2018 T.N.K. (EVAL_NNUE_HALFKP_256x2_32_32)
# $ ndk-build ENGINE_TARGET=YANEURAOU_2018_TNK_ENGINE_HALFKP

# example: 2018 T.N.K. (EVAL_NNUE_K_P_256x2_32_32)
# $ ndk-build ENGINE_TARGET=YANEURAOU_2018_TNK_ENGINE_K_P

# example: tnk- Mate (MATE_ENGINE)
# $ ndk-build ENGINE_TARGET=MATE_ENGINE

ENGINE_TARGET := YANEURAOU_2018_OTAFUKU_ENGINE
#ENGINE_TARGET := YANEURAOU_2018_OTAFUKU_ENGINE_KPPT
#ENGINE_TARGET := YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT
#ENGINE_TARGET := YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL
#ENGINE_TARGET := YANEURAOU_2018_TNK_ENGINE
#ENGINE_TARGET := YANEURAOU_2018_TNK_ENGINE_HALFKP
#ENGINE_TARGET := YANEURAOU_2018_TNK_ENGINE_K_P
#ENGINE_TARGET := MATE_ENGINE

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_OTAFUKU_ENGINE)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_OTAFUKU_ENGINE
  ENGINE_NAME := YaneuraOu-otafuku
endif

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_OTAFUKU_ENGINE_KPPT)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_OTAFUKU_ENGINE
  ENGINE_NAME := YaneuraOu-otafuku-kppt
endif

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT
  ENGINE_NAME := YaneuraOu-otafuku-kpp_kkpt
endif

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL
  ENGINE_NAME := YaneuraOu-otafuku-material
endif

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_TNK_ENGINE)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_TNK_ENGINE
  ENGINE_NAME := YaneuraOu-tnk
endif

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_TNK_ENGINE_HALFKP)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_TNK_ENGINE_HALFKP
  ENGINE_NAME := YaneuraOu-tnk-half_kp
endif

ifeq ($(ENGINE_TARGET),YANEURAOU_2018_TNK_ENGINE_K_P)
  ARCH_DEF += -DUSE_MAKEFILE -DYANEURAOU_2018_TNK_ENGINE_K_P
  ENGINE_NAME := YaneuraOu-tnk-k_p
endif

ifeq ($(ENGINE_TARGET),MATE_ENGINE)
  ARCH_DEF += -DUSE_MAKEFILE -DMATE_ENGINE
  ENGINE_NAME := YaneuraOu-tnk-mate
endif

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
  ARCH_DEF += -DIS_64BIT -DIS_ARM -mfpu=neon
  LOCAL_ARM_NEON := true
endif

ifeq ($(TARGET_ARCH_ABI),x86_64)
  ARCH_DEF += -DUSE_SSE42 -msse4.2
endif

ifeq ($(TARGET_ARCH_ABI),x86)
  ARCH_DEF +=
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  ARCH_DEF += -DIS_ARM -mfpu=neon
  LOCAL_ARM_NEON := true
endif

LOCAL_MODULE    := $(ENGINE_NAME)-$(TARGET_ARCH_ABI)
LOCAL_CXXFLAGS  := -std=c++14 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive -D__STDINT_MACROS -D__STDC_LIMIT_MACROS $(ARCH_DEF)
LOCAL_CXXFLAGS += -fPIE -Wno-unused-parameter
LOCAL_LDFLAGS += -fPIE -pie -flto
LOCAL_LDLIBS =
LOCAL_C_INCLUDES :=
LOCAL_CPP_FEATURES += exceptions rtti
#LOCAL_STATIC_LIBRARIES    := -lpthread

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
  ../source/extra/book/apery_book.cpp                                  \
  ../source/extra/book/book.cpp                                        \
  ../source/extra/book/makebook2019.cpp                                \
  ../source/extra/bitop.cpp                                            \
  ../source/extra/entering_king_win.cpp                                \
  ../source/extra/long_effect.cpp                                      \
  ../source/extra/mate/mate1ply_with_effect.cpp                        \
  ../source/extra/mate/mate1ply_without_effect.cpp                     \
  ../source/extra/mate/mate_n_ply.cpp                                  \
  ../source/extra/benchmark.cpp                                        \
  ../source/extra/test_cmd.cpp                                         \
  ../source/extra/see.cpp                                              \
  ../source/extra/sfen_packer.cpp                                      \
  ../source/extra/kif_converter/kif_convert_tools.cpp                  \
  ../source/eval/evaluate_bona_piece.cpp                               \
  ../source/eval/kppt/evaluate_kppt.cpp                                \
  ../source/eval/kppt/evaluate_kppt_learner.cpp                        \
  ../source/eval/kpp_kkpt/evaluate_kpp_kkpt.cpp                        \
  ../source/eval/kpp_kkpt/evaluate_kpp_kkpt_learner.cpp                \
  ../source/eval/evaluate.cpp                                          \
  ../source/eval/evaluate_io.cpp                                       \
  ../source/eval/evaluate_mir_inv_tools.cpp                            \
  ../source/engine/user-engine/user-search.cpp                         \
  ../source/engine/2018-otafuku-engine/2018-otafuku-search.cpp         \
  ../source/learn/learner.cpp                                          \
  ../source/learn/learning_tools.cpp                                   \
  ../source/learn/multi_think.cpp

ifeq ($(ENGINE_TARGET),MATE_ENGINE)
LOCAL_SRC_FILES += \
	../source/engine/mate-engine/mate-search.cpp
endif

ifeq ($(findstring YANEURAOU_2018_TNK_ENGINE,$(ENGINE_TARGET)),YANEURAOU_2018_TNK_ENGINE)
LOCAL_SRC_FILES += \
  ../source/eval/nnue/evaluate_nnue.cpp                                \
  ../source/eval/nnue/evaluate_nnue_learner.cpp                        \
  ../source/eval/nnue/nnue_test_command.cpp                            \
  ../source/eval/nnue/features/k.cpp                                   \
  ../source/eval/nnue/features/p.cpp                                   \
  ../source/eval/nnue/features/half_kp.cpp                             \
  ../source/eval/nnue/features/half_relative_kp.cpp
endif

include $(BUILD_EXECUTABLE)
