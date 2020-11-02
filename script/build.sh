#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Linux バイナリのビルド
# sudo apt install build-essential clang g++-9 libomp-8-dev libopenblas-dev

# Example 1: 全パターンのビルド
# build.sh

# Example 2: 指定パターンのビルド(-c: コンパイラ名, -e: エディション名, -t: ターゲット名)
# build.sh -c clang++ -e YANEURAOU_ENGINE_NNUE

# Example 3: 特定パターンのビルド(複数指定時はカンマ区切り、 -e, -t オプションのみワイルドカード使用可、ワイルドカード使用時はシングルクォートで囲む)
# build.sh -c clang++,g++-9 -e '*KPPT*,*NNUE*'

MAKE=make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

ARCHCPUS='*'
COMPILERS="clang++,g++"
EDITIONS='*'
OS='linux'
TARGETS='*'

while getopts a:c:e:o:t: OPT
do
  case $OPT in
    a) ARCHCPUS="$OPTARG"
      ;;
    c) COMPILERS="$OPTARG"
      ;;
    e) EDITIONS="$OPTARG"
      ;;
    o) OS="$OPTARG"
      ;;
    t) TARGETS="$OPTARG"
      ;;
  esac
done

set -f
IFS=, eval 'ARCHCPUSARR=($ARCHCPUS)'
IFS=, eval 'COMPILERSARR=($COMPILERS)'
IFS=, eval 'EDITIONSARR=($EDITIONS)'
IFS=, eval 'TARGETSARR=($TARGETS)'

cd `dirname $0`
cd ../source

ARCHCPUS=(
  AVX512
  AVX2
  SSE42
  SSE41
  SSSE3
  SSE2
  NO_SSE
  OTHER
  ZEN1
  ZEN2
)

EDITIONS=(
  YANEURAOU_ENGINE_KPPT
  YANEURAOU_ENGINE_KPP_KKPT
  YANEURAOU_ENGINE_MATERIAL
  YANEURAOU_ENGINE_NNUE
  YANEURAOU_ENGINE_NNUE_KP256
  MATE_ENGINE
  USER_ENGINE
)

TARGETS=(
  normal
  tournament
  evallearn
  gensfen
)

declare -A FILESTR;
FILESTR=(
  ["YANEURAOU_ENGINE_KPPT"]="kppt"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="kpp_kkpt"
  ["YANEURAOU_ENGINE_MATERIAL"]="material"
  ["YANEURAOU_ENGINE_NNUE"]="nnue"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="nnue-k_p_256"
  ["MATE_ENGINE"]="mate"
  ["USER_ENGINE"]="user"
);

set -f
for ARCHCPU in ${ARCHCPUSARR[@]}; do
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
          BUILDDIR=../build/${OS}/${FILESTR[$EDITION]}
          mkdir -p ${BUILDDIR}
          for TARGET in ${TARGETS[@]}; do
            for TARGETPTN in ${TARGETSARR[@]}; do
              set +f
              if [[ $TARGET == $TARGETPTN ]]; then
                echo "* target: ${TARGET}"
                TGSTR=YaneuraOu-${FILESTR[$EDITION]}-${OS}-${CSTR}-${TARGET}-${ARCHCPU}
                nice ${MAKE} -f ${MAKEFILE} -j${JOBS} ${TARGET} TARGET_CPU=${ARCHCPU} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} > >(tee ${BUILDDIR}/${TGSTR}.log) || exit $?
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
done
