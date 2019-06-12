#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Linux バイナリのビルド
# sudo apt install build-essential clang g++-9 libomp-8-dev libopenblas-dev

# Example 1: 全パターンのビルド
# linux_build.sh

# Example 2: 指定パターンのビルド(-c: コンパイラ名, -e: エディション名, -t: ターゲット名)
# linux_build.sh -c clang++ -e YANEURAOU_ENGINE_NNUE_HALFKP256 -t avx2

# Example 3: 特定パターンのビルド(複数指定時はカンマ区切り、 -e, -t オプションのみワイルドカード使用可、ワイルドカード使用時はシングルクォートで囲む)
# linux_build.sh -c clang++,g++-9 -e '*KPPT*,*HALFKP*' -t '*avx2*'

MAKE=make
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

cd `dirname $0`
cd ../source

EDITIONS=(
  YANEURAOU_ENGINE_KPPT
  YANEURAOU_ENGINE_KPP_KKPT
  YANEURAOU_ENGINE_MATERIAL
  YANEURAOU_ENGINE_NNUE_HALFKP256
  YANEURAOU_ENGINE_NNUE_KP256
  MATE_ENGINE
)

TARGETS=(
  icelake
  cascadelake
  avx512
  avx2
  sse42
  sse2
  tournament-icelake
  tournament-cascadelake
  tournament-avx512
  tournament-avx2
  tournament-sse42
  evallearn-icelake
  evallearn-cascadelake
  evallearn-avx512
  evallearn-avx2
  evallearn-sse42
)

declare -A FILESTR;
FILESTR=(
  ["YANEURAOU_ENGINE_KPPT"]="kppt"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="kpp_kkpt"
  ["YANEURAOU_ENGINE_MATERIAL"]="material"
  ["YANEURAOU_ENGINE_NNUE_HALFKP256"]="nnue-halfkp_256"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="nnue-k_p_256"
  ["MATE_ENGINE"]="mate"
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
        BUILDDIR=../build/linux/${FILESTR[$EDITION]}
        mkdir -p ${BUILDDIR}
        for TARGET in ${TARGETS[@]}; do
          for TARGETPTN in ${TARGETSARR[@]}; do
            set +f
            if [[ $TARGET == $TARGETPTN ]]; then
              echo "* target: ${TARGET}"
              TGSTR=YaneuraOu-${FILESTR[$EDITION]}-linux-${CSTR}-${TARGET}
              ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
              nice ${MAKE} -f ${MAKEFILE} -j${JOBS} ${TARGET} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee ${BUILDDIR}/${TGSTR}.log
              cp YaneuraOu-by-gcc ${BUILDDIR}/${TGSTR}
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
