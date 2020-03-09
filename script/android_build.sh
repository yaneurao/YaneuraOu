#!/bin/bash

cd `dirname $0`
cd ..

mkdir -p build/android/KPPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPPT
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KPPT 2>&1 | tee build/android/KPPT/KPPT.log
cp -r libs/* build/android/KPPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPPT

mkdir -p build/android/KPP_KKPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT 2>&1 | tee build/android/KPP_KKPT/KPP_KKPT.log
cp -r libs/* build/android/KPP_KKPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT

mkdir -p build/android/KOMA
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL 2>&1 | tee build/android/KOMA/KOMA.log
cp -r libs/* build/android/KOMA
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL

mkdir -p build/android/NNUE
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_NNUE 2>&1 | tee build/android/NNUE/NNUE.log
cp -r libs/* build/android/NNUE
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE

mkdir -p build/android/NNUE_KP256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256 2>&1 | tee build/android/NNUE_KP256/NNUE_KP256.log
cp -r libs/* build/android/NNUE_KP256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256

mkdir -p build/android/MATE
ndk-build clean ENGINE_TARGET=MATE_ENGINE
ndk-build ENGINE_TARGET=MATE_ENGINE 2>&1 | tee build/android/MATE/MATE.log
cp -r libs/* build/android/MATE
ndk-build clean ENGINE_TARGET=MATE_ENGINE
