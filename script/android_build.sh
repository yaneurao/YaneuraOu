#!/bin/bash

cd `dirname $0`
cd ..

mkdir -p build/android/kppt
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPPT
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KPPT 2>&1 | tee build/android/kppt/kppt.log
cp -r libs/* build/android/kppt
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPPT

mkdir -p build/android/kpp_kkpt
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT 2>&1 | tee build/android/kpp_kkpt/kpp_kkpt.log
cp -r libs/* build/android/kpp_kkpt
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT

mkdir -p build/android/material
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL 2>&1 | tee build/android/material/material.log
cp -r libs/* build/android/material
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL

mkdir -p build/android/nnue-halfkp_256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_HALFKP
ndk-build ENGINE_TARGET=YANEURAOU_2018_TNK_ENGINE 2>&1 | tee build/android/nnue-halfkp_256/nnue-halfkp_256.log
cp -r libs/* build/android/nnue-halfkp_256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_HALFKP256

mkdir -p build/android/nnue-k_p_256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KP256 2>&1 | tee build/android/nnue-k_p_256/nnue-k_p_256.log
cp -r libs/* build/android/nnue-k_p_256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256

mkdir -p build/android/mate
ndk-build clean ENGINE_TARGET=MATE_ENGINE
ndk-build ENGINE_TARGET=MATE_ENGINE 2>&1 | tee build/android/mate/mate.log
cp -r libs/* build/android/mate
ndk-build clean ENGINE_TARGET=MATE_ENGINE
