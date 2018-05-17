#!/usr/bin/bash
# -*- coding: utf-8 -*-
# MSYS2 上で Windows バイナリのビルド
OS=Windows_NT
MAKE=mingw32-make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

cd `dirname $0`
cd ../source

COMPILER=clang++
BUILDDIR=../build/2018otafuku
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE
TARGET=YaneuraOu-2018-otafuku-msys2-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

COMPILER=clang++
BUILDDIR=../build/2018tnk
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE
TARGET=YaneuraOu-2018-tnk-msys2-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

COMPILER=clang++
BUILDDIR=../build/mate
mkdir -p ${BUILDDIR}
EDITION=MATE_ENGINE
TARGET=YaneuraOu-mate-msys2-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

COMPILER=g++
BUILDDIR=../build/2018Otafuku
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE
TARGET=YaneuraOu-2018-otafuku-msys2-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

COMPILER=g++
BUILDDIR=../build/2018tnk
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE
TARGET=YaneuraOu-2018-tnk-msys2-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

COMPILER=g++
BUILDDIR=../build/mate
mkdir -p ${BUILDDIR}
EDITION=MATE_ENGINE
TARGET=YaneuraOu-mate-msys2-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} OS=${OS} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TARGET}${TGTAIL[$key]}.exe
done

${MAKE} -f ${MAKEFILE} clean
