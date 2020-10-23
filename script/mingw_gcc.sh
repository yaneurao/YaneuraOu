#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Windows バイナリのビルド (gcc)
# sudo apt install build-essential mingw-w64 libopenblas-dev

# Example 1: 全パターンのビルド
# mingw_gcc.sh

# Example 2: 指定パターンのビルド(-e: エディション名, -t: ターゲット名)
# mingw_gcc.sh -e YANEURAOU_ENGINE_NNUE

# Example 3: 特定パターンのビルド(ワイルドカード使用時はシングルクォートで囲む)
# mingw_gcc.sh -e '*KPPT*,*NNUE*'

MAKE=make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

ARCHCPUS='*'
COMPILERS="x86_64-w64-mingw32-g++-posix,i686-w64-mingw32-g++-posix"
EDITIONS='*'
OS='Windows_NT'
TARGETS='*'

while getopts a:c:e:t: OPT
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
  OTHER
  ZEN1
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

for ARCHCPU in ${ARCHCPUSARR[@]}; do
  for COMPILER in ${COMPILERSARR[@]}; do
    echo "* compiler: ${COMPILER}"
    CSTR=${COMPILER##*/}
    CSTR=${CSTR##*\\}
    for EDITION in ${EDITIONS[@]}; do
      for EDITIONPTN in ${EDITIONSARR[@]}; do
        if [[ $EDITION == $EDITIONPTN ]]; then
          echo "* edition: ${EDITION}"
          BUILDDIR=../build/windows/${FILESTR[$EDITION]}
          mkdir -p ${BUILDDIR}
          for TARGET in ${TARGETS[@]}; do
            for TARGETPTN in ${TARGETSARR[@]}; do
              if [[ $TARGET == $TARGETPTN ]]; then
                echo "* target: ${TARGET}"
                TGSTR=YaneuraOu-${FILESTR[$EDITION]}-windows-${CSTR}-${TARGET}-${ARCHCPU}
                nice ${MAKE} -f ${MAKEFILE} -j${JOBS} ${TARGET} OS=${OS} TARGET_CPU=${ARCHCPU} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} > >(tee ${BUILDDIR}/${TGSTR}.log) || exit $?
                cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TGSTR}.exe
                ${MAKE} -f ${MAKEFILE} clean OS=${OS} YANEURAOU_EDITION=${EDITION}
                break
              fi
            done
          done
          break
        fi
      done
    done
  done
done
