#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Windows バイナリのビルド (gcc)
# sudo apt install build-essential mingw-w64 libopenblas-dev

# Example 1: 全パターンのビルド
# mingw_gcc.sh

# Example 2: 指定パターンのビルド(-e: エディション名, -t: ターゲット名)
# mingw_gcc.sh -e YANEURAOU_ENGINE_NNUE_HALFKP256 -t avx2

# Example 3: 特定パターンのビルド(ワイルドカード使用時はシングルクォートで囲む)
# mingw_gcc.sh -e '*KPPT*,*HALFKP*' -t '*avx2*'

OS=Windows_NT
MAKE=make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

COMPILER64=x86_64-w64-mingw32-g++-posix
COMPILER32=i686-w64-mingw32-g++-posix
COMPILER=

EDITIONS='*'
TARGETS='*'

while getopts c:e:t: OPT
do
  case $OPT in
    e) EDITIONS="$OPTARG"
      ;;
    t) TARGETS="$OPTARG"
      ;;
  esac
done

set -f
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
  nosse
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
for EDITION in ${EDITIONS[@]}; do
  for EDITIONPTN in ${EDITIONSARR[@]}; do
    set +f
    if [[ $EDITION == $EDITIONPTN ]]; then
      set -f
      echo "* edition: ${EDITION}"
      BUILDDIR=../build/mingw/${FILESTR[$EDITION]}
      mkdir -p ${BUILDDIR}
      for TARGET in ${TARGETS[@]}; do
        for TARGETPTN in ${TARGETSARR[@]}; do
          set +f
          if [[ $TARGET == $TARGETPTN ]]; then
            echo "* target: ${TARGET}"
            if [ $TARGET == 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
            TGSTR=YaneuraOu-${FILESTR[$EDITION]}-msys2-${CSTR}-${TARGET}
            ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
            nice ${MAKE} -f ${MAKEFILE} -j${JOBS} ${TARGET} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee ${BUILDDIR}/${TGSTR}.log
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

BUILDDIR=../build/windows/2018otafuku-kppt
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_KPPT
TARGET=YaneuraOu-2018-otafuku-kppt-mingw-gcc
TGTAIL=(icelake cascadelake avx512 avx2 sse42 sse2 nosse tournament-icelake tournament-cascadelake tournament-avx512 tournament-avx2 tournament-sse42 evallearn-icelake evallearn-cascadelake evallearn-avx512 evallearn-avx2 evallearn-sse42)
for BTG in ${TGTAIL[@]}; do
  if [ ${BTG} = 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
  ${MAKE} -f ${MAKEFILE} -j${JOBS} ${BTG} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}-${BTG}.log
  cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}-${BTG}.exe
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

BUILDDIR=../build/windows/2018otafuku-kpp_kkpt
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT
TARGET=YaneuraOu-2018-otafuku-kpp_kkpt-mingw-gcc
TGTAIL=(icelake cascadelake avx512 avx2 sse42 sse2 nosse tournament-icelake tournament-cascadelake tournament-avx512 tournament-avx2 tournament-sse42 evallearn-icelake evallearn-cascadelake evallearn-avx512 evallearn-avx2 evallearn-sse42)
for BTG in ${TGTAIL[@]}; do
  if [ ${BTG} = 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
  ${MAKE} -f ${MAKEFILE} -j${JOBS} ${BTG} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}-${BTG}.log
  cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}-${BTG}.exe
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

BUILDDIR=../build/windows/2018otafuku-material
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL
TARGET=YaneuraOu-2018-otafuku-material-mingw-gcc
TGTAIL=(icelake cascadelake avx512 avx2 sse42 sse2 nosse tournament-icelake tournament-cascadelake tournament-avx512 tournament-avx2 tournament-sse42 evallearn-icelake evallearn-cascadelake evallearn-avx512 evallearn-avx2 evallearn-sse42)
for BTG in ${TGTAIL[@]}; do
  if [ ${BTG} = 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
  ${MAKE} -f ${MAKEFILE} -j${JOBS} ${BTG} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}-${BTG}.log
  cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}-${BTG}.exe
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

BUILDDIR=../build/windows/2018tnk
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE
TARGET=YaneuraOu-2018-tnk-mingw-gcc
TGTAIL=(icelake cascadelake avx512 avx2 sse42 sse2 nosse tournament-icelake tournament-cascadelake tournament-avx512 tournament-avx2 tournament-sse42 evallearn-icelake evallearn-cascadelake evallearn-avx512 evallearn-avx2 evallearn-sse42)
for BTG in ${TGTAIL[@]}; do
  if [ ${BTG} = 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
  ${MAKE} -f ${MAKEFILE} -j${JOBS} ${BTG} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}-${BTG}.log
  cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}-${BTG}.exe
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

BUILDDIR=../build/windows/2018tnk-k-p
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE_K_P
TARGET=YaneuraOu-2018-tnk-k-p-mingw-gcc
TGTAIL=(icelake cascadelake avx512 avx2 sse42 sse2 nosse tournament-icelake tournament-cascadelake tournament-avx512 tournament-avx2 tournament-sse42 evallearn-icelake evallearn-cascadelake evallearn-avx512 evallearn-avx2 evallearn-sse42)
for BTG in ${TGTAIL[@]}; do
  if [ ${BTG} = 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
  ${MAKE} -f ${MAKEFILE} -j${JOBS} ${BTG} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}-${BTG}.log
  cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}-${BTG}.exe
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

BUILDDIR=../build/windows/tnk-mate
mkdir -p ${BUILDDIR}
EDITION=MATE_ENGINE
TARGET=YaneuraOu-mate-mingw-gcc
TGTAIL=(icelake cascadelake avx512 avx2 sse42 sse2 nosse tournament-icelake tournament-cascadelake tournament-avx512 tournament-avx2 tournament-sse42 evallearn-icelake evallearn-cascadelake evallearn-avx512 evallearn-avx2 evallearn-sse42)
for BTG in ${TGTAIL[@]}; do
  if [ ${BTG} = 'nosse' ]; then COMPILER=${COMPILER32}; else COMPILER=${COMPILER64}; fi
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
  ${MAKE} -f ${MAKEFILE} -j${JOBS} ${BTG} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}-${BTG}.log
  cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}-${BTG}.exe
  ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done
