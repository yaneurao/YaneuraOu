#!/bin/bash

cd `dirname $0`
cd ..

mkdir -p build/android/KPPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPPT
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KPPT > >(tee build/android/KPPT/KPPT.log) || exit $?
cp -r libs/* build/android/KPPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPPT

mkdir -p build/android/KPP_KKPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT > >(tee build/android/KPP_KKPT/KPP_KKPT.log) || exit $?
cp -r libs/* build/android/KPP_KKPT
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_KPP_KKPT

mkdir -p build/android/KOMA
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL > >(tee build/android/KOMA/KOMA.log) || exit $?
cp -r libs/* build/android/KOMA
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_MATERIAL

mkdir -p build/android/NNUE
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_NNUE > >(tee build/android/NNUE/NNUE.log) || exit $?
cp -r libs/* build/android/NNUE
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE

mkdir -p build/android/NNUE_KP256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256
ndk-build ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256 > >(tee build/android/NNUE_KP256/NNUE_KP256.log) || exit $?
cp -r libs/* build/android/NNUE_KP256
ndk-build clean ENGINE_TARGET=YANEURAOU_ENGINE_NNUE_KP256

mkdir -p build/android/MATE
ndk-build clean ENGINE_TARGET=MATE_ENGINE
ndk-build ENGINE_TARGET=MATE_ENGINE > >(tee build/android/MATE/MATE.log) || exit $?
cp -r libs/* build/android/MATE
ndk-build clean ENGINE_TARGET=MATE_ENGINE

mkdir -p build/android/USER
ndk-build clean ENGINE_TARGET=USER_ENGINE
ndk-build ENGINE_TARGET=USER_ENGINE > >(tee build/android/USER/USER.log) || exit $?
cp -r libs/* build/android/USER
ndk-build clean ENGINE_TARGET=USER_ENGINE
