#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Windows バイナリのビルド (gcc)
OS=Windows_NT
MAKE=make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

cd `dirname $0`
cd ../source

# sudo apt install build-essential mingw-w64
COMPILER=x86_64-w64-mingw32-g++-posix

BUILDDIR=../build/2018otafuku
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE
TARGET=YaneuraOu-2018-otafuku-mingw-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

BUILDDIR=../build/2018tnk
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE
TARGET=YaneuraOu-2018-tnk-mingw-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

BUILDDIR=../build/mate
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_MATE_ENGINE
TARGET=YaneuraOu-mate-mingw-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

${MAKE} -f ${MAKEFILE} clean
