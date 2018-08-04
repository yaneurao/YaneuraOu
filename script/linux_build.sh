#!/bin/bash
# -*- coding: utf-8 -*-
# Ubuntu 上で Linux バイナリのビルド
MAKE=make
MAKEFILE=Makefile
JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

cd `dirname $0`
cd ../source

# Bash on Windows (Ubuntu 18.04 Bionic) 環境の場合は http://apt.llvm.org/ を参考に clang++-7 を導入する。
# sudo apt install build-essential clang-7 lldb-7 lld-7
COMPILER=clang++-7
BUILDDIR=../build/2018otafuku-kppt
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_KPPT
TARGET=YaneuraOu-2018-otafuku-kppt-linux-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# Bash on Windows (Ubuntu 18.04 Bionic) 環境の場合は http://apt.llvm.org/ を参考に clang++-7 を導入する。
# sudo apt install build-essential clang-7 lldb-7 lld-7
COMPILER=clang++-7
BUILDDIR=../build/2018otafuku-kpp_kkpt
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT
TARGET=YaneuraOu-2018-otafuku-kpp_kppt-linux-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# Bash on Windows (Ubuntu 18.04 Bionic) 環境の場合は http://apt.llvm.org/ を参考に clang++-7 を導入する。
# sudo apt install build-essential clang-7 lldb-7 lld-7
COMPILER=clang++-7
BUILDDIR=../build/2018otafuku-material
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL
TARGET=YaneuraOu-2018-otafuku-material-linux-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# Bash on Windows (Ubuntu 18.04 Bionic) 環境の場合は http://apt.llvm.org/ を参考に clang++-7 を導入する。
# sudo apt install build-essential clang-7 lldb-7 lld-7 libopenblas-dev
COMPILER=clang++-7
BUILDDIR=../build/2018tnk
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE
TARGET=YaneuraOu-2018-tnk-linux-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# Bash on Windows (Ubuntu 18.04 Bionic) 環境の場合は http://apt.llvm.org/ を参考に clang++-7 を導入する。
# sudo apt install build-essential clang-7 lldb-7 lld-7
COMPILER=clang++-7
BUILDDIR=../build/mate
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_MATE_ENGINE
TARGET=YaneuraOu-mate-linux-clang
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# ここでは、 https://launchpad.net/~jonathonf/+archive/ubuntu/gcc より g++-8 を導入してビルドに用いる。
# sudo add-apt-repository ppa:jonathonf/gcc
# sudo apt update
# sudo apt install g++-8
COMPILER=g++-8
BUILDDIR=../build/2018otafuku-kppt
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_KPPT
TARGET=YaneuraOu-2018-otafuku-kppt-linux-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# ここでは、 https://launchpad.net/~jonathonf/+archive/ubuntu/gcc より g++-8 を導入してビルドに用いる。
# sudo add-apt-repository ppa:jonathonf/gcc
# sudo apt update
# sudo apt install g++-8
COMPILER=g++-8
BUILDDIR=../build/2018otafuku-kpp_kkpt
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_KPP_KKPT
TARGET=YaneuraOu-2018-otafuku-kpp_kkpt-linux-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# ここでは、 https://launchpad.net/~jonathonf/+archive/ubuntu/gcc より g++-8 を導入してビルドに用いる。
# sudo add-apt-repository ppa:jonathonf/gcc
# sudo apt update
# sudo apt install g++-8
COMPILER=g++-8
BUILDDIR=../build/2018otafuku-material
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_OTAFUKU_ENGINE_MATERIAL
TARGET=YaneuraOu-2018-otafuku-material-linux-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# ここでは、 https://launchpad.net/~jonathonf/+archive/ubuntu/gcc より g++-8 を導入してビルドに用いる。
# sudo add-apt-repository ppa:jonathonf/gcc
# sudo apt update
# sudo apt install g++-8 libopenblas-dev
COMPILER=g++-8
BUILDDIR=../build/2018tnk
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_2018_TNK_ENGINE
TARGET=YaneuraOu-2018-tnk-linux-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done

# ここでは、 https://launchpad.net/~jonathonf/+archive/ubuntu/gcc より g++-8 を導入してビルドに用いる。
# sudo add-apt-repository ppa:jonathonf/gcc
# sudo apt update
# sudo apt install g++-8
COMPILER=g++-8
BUILDDIR=../build/mate
mkdir -p ${BUILDDIR}
EDITION=YANEURAOU_MATE_ENGINE
TARGET=YaneuraOu-mate-linux-gcc
declare -A TGTAIL=([avx2]=-avx2 [sse42]=-sse42 [tournament]=-tournament-avx2 [tournament-sse42]=-tournament-sse42 [evallearn]=-evallearn-avx2 [evallearn-sse42]=-evallearn-sse42 [sse41]=-sse41 [sse2]=-sse2)
for key in ${!TGTAIL[*]}
do
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
	${MAKE} -f ${MAKEFILE} -j${JOBS} ${key} YANEURAOU_EDITION=${EDITION} COMPILER=${COMPILER} 2>&1 | tee $BUILDDIR/${TARGET}${TGTAIL[$key]}.log
	cp YaneuraOu-by-gcc ${BUILDDIR}/${TARGET}${TGTAIL[$key]}
	${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITION}
done
