#!/bin/bash
# -*- coding: utf-8 -*-
# MSYS2 (MinGW 32-bit) 上で Windows バイナリのビルド
# ビルド用パッケージの導入
# $ pacman --needed --noconfirm -Syuu pactoys-git
# $ pacboy --needed --noconfirm -Syuu clang:m openblas:x openmp:x toolchain:m base-devel:
# MSYS2パッケージの更新、更新出来る項目が無くなるまで繰り返し実行、場合によってはMSYS2の再起動が必要
# $ pacman -Syuu --noconfirm

# Example 1: 全パターンのビルド
# msys2_build32.sh

# Example 2: 指定パターンのビルド(-c: コンパイラ名, -e: エディション名, -t: ターゲット名)
# msys2_build32.sh -c clang++ -e YANEURAOU_ENGINE_NNUE

# Example 3: 特定パターンのビルド(複数指定時はカンマ区切り、 -e, -t オプションのみワイルドカード使用可、ワイルドカード使用時はシングルクォートで囲む)
# msys2_build32.sh -c clang++,g++ -e '*KPPT*,*NNUE*'

OS=Windows_NT
MAKE=mingw32-make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

COMPILERS="clang++,g++"
EDITIONS='*'
TARGETS='*'

while getopts c:e:t: OPT
do
  case $OPT in
    c) COMPILERS="$OPTARG"
      ;;
    e) EDITIONS="$OPTARG"
      ;;
    t) TARGETS="$OPTARG"
      ;;
  esac
done

set -f
IFS=, eval 'COMPILERSARR=($COMPILERS)'
IFS=, eval 'EDITIONSARR=($EDITIONS)'
IFS=, eval 'TARGETSARR=($TARGETS)'

pushd `dirname $0`
pushd ../source

EDITIONS=(
  YANEURAOU_ENGINE_NNUE
  YANEURAOU_ENGINE_NNUE_HALFKPE9
  YANEURAOU_ENGINE_NNUE_KP256
  YANEURAOU_ENGINE_KPPT
  YANEURAOU_ENGINE_KPP_KKPT
  YANEURAOU_ENGINE_MATERIAL
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=2"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=3"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=4"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=5"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=6"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=7"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=8"
  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=9"
#  "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=10"
  YANEURAOU_MATE_ENGINE
  TANUKI_MATE_ENGINE
  USER_ENGINE
)

TARGETS=(
  normal
  tournament
  evallearn
  gensfen
)

declare -A DIRSTR;
DIRSTR=(
  ["YANEURAOU_ENGINE_NNUE"]="NNUE"
  ["YANEURAOU_ENGINE_NNUE_HALFKPE9"]="NNUE_HALFKPE9"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="NNUE_KP256"
  ["YANEURAOU_ENGINE_KPPT"]="KPPT"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="KPP_KKPT"
  ["YANEURAOU_ENGINE_MATERIAL"]="MaterialLv1"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=2"]="MaterialLv2"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=3"]="MaterialLv3"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=4"]="MaterialLv4"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=5"]="MaterialLv5"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=6"]="MaterialLv6"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=7"]="MaterialLv7"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=8"]="MaterialLv8"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=9"]="MaterialLv9"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=10"]="MaterialLv10"
  ["YANEURAOU_MATE_ENGINE"]="YaneuraOu_MATE"
  ["TANUKI_MATE_ENGINE"]="tanuki_MATE"
  ["USER_ENGINE"]="USER"
);

declare -A FILESTR;
FILESTR=(
  ["YANEURAOU_ENGINE_NNUE"]="YaneuraOu_NNUE"
  ["YANEURAOU_ENGINE_NNUE_HALFKPE9"]="YaneuraOu_NNUE_KPE9"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="YaneuraOu_NNUE_KP256"
  ["YANEURAOU_ENGINE_KPPT"]="YaneuraOu_KPPT"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="YaneuraOu_KPP_KKPT"
  ["YANEURAOU_ENGINE_MATERIAL"]="YaneuraOu_MaterialLv1"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=2"]="YaneuraOu_MaterialLv2"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=3"]="YaneuraOu_MaterialLv3"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=4"]="YaneuraOu_MaterialLv4"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=5"]="YaneuraOu_MaterialLv5"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=6"]="YaneuraOu_MaterialLv6"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=7"]="YaneuraOu_MaterialLv7"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=8"]="YaneuraOu_MaterialLv8"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=9"]="YaneuraOu_MaterialLv9"
  ["YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=10"]="YaneuraOu_MaterialLv10"
  ["YANEURAOU_MATE_ENGINE"]="YaneuraOu_MATE"
  ["TANUKI_MATE_ENGINE"]="tanuki_MATE"
  ["USER_ENGINE"]="user"
);

set -f
for COMPILER in ${COMPILERSARR[@]}; do
  echo "* compiler: ${COMPILER}"
  CSTR=${COMPILER##*/}
  CSTR=${CSTR##*\\}
  for EDITION in ${EDITIONS[@]}; do
    for EDITIONPTN in ${EDITIONSARR[@]}; do
      set +f
      if [[ $EDITION == $EDITIONPTN ]]; then
        set -f
        echo "* edition: ${EDITION}"
        BUILDDIR=../build/windows/${DIRSTR[$EDITION]}
        mkdir -p ${BUILDDIR}
        for TARGET in ${TARGETS[@]}; do
          for TARGETPTN in ${TARGETSARR[@]}; do
            set +f
            if [[ $TARGET == $TARGETPTN ]]; then
              echo "* target: ${TARGET}"
              TGSTR=YaneuraOu-${FILESTR[$EDITION]}-msys2-${CSTR}-${TARGET}
              ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
              nice ${MAKE} -f ${MAKEFILE} -j${JOBS} ${TARGET} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} TARGET_CPU=NO_SSE > >(tee ${BUILDDIR}/${TGSTR}.log) || exit $?
              cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TGSTR}.exe
              ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
              set -f
              break
            fi
            set -f
          done
        done
        break
      fi
      set -f
    done
  done
done

popd
popd
