#!/bin/bash
# -*- coding: utf-8 -*-
# MSYS2 (MinGW 64-bit) 上で Windows バイナリのビルド
# ビルド用パッケージの導入
# $ pacman --needed --noconfirm -Syuu
# $ pacman --needed --noconfirm -Syuu pactoys
# $ pacboy --needed --noconfirm -Syuu clang:m lld:m openblas:x openmp:x toolchain:m base-devel:
# MSYS2パッケージの更新、更新出来る項目が無くなるまで繰り返し実行、場合によってはMSYS2の再起動が必要
# $ pacman -Syuu --noconfirm

# Example 1: 全パターンのビルド
# msys2_build.sh

# Example 2: 指定パターンのビルド(-c: コンパイラ名, -e: エディション名, -t: ターゲット名)
# msys2_build.sh -c clang++ -e YANEURAOU_ENGINE_NNUE

# Example 3: 特定パターンのビルド(複数指定時はカンマ区切り、 -e, -t オプションのみワイルドカード使用可、ワイルドカード使用時はシングルクォートで囲む)
# msys2_build.sh -c clang++,g++ -e '*KPPT*,*NNUE*'

OS=Windows_NT
MAKE=mingw32-make
MAKEFILE=Makefile

ARCHCPUS='*'
COMPILERS="clang++,g++"
EDITIONS='*'
TARGETS='*'
EXTRA=''

while getopts a:c:e:t:x: OPT
do
  case $OPT in
    a) ARCHCPUS="$OPTARG"
      ;;
    c) COMPILERS="$OPTARG"
      ;;
    e) EDITIONS="$OPTARG"
      ;;
    t) TARGETS="$OPTARG"
      ;;
    x) EXTRA="$OPTARG"
      ;;
  esac
done

set -f
IFS=, eval 'COMPILERSARR=($COMPILERS)'
IFS=, eval 'EDITIONSARR=($EDITIONS)'
IFS=, eval 'ARCHCPUSARR=($ARCHCPUS)'
IFS=, eval 'TARGETSARR=($TARGETS)'

pushd `dirname $0`
pushd ../source

EDITIONS=(
  YANEURAOU_ENGINE_NNUE
  YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32
  YANEURAOU_ENGINE_NNUE_HALFKPE9
  YANEURAOU_ENGINE_NNUE_KP256
  YANEURAOU_ENGINE_KPPT
  YANEURAOU_ENGINE_KPP_KKPT
  YANEURAOU_ENGINE_MATERIAL
  YANEURAOU_ENGINE_MATERIAL2
  YANEURAOU_ENGINE_MATERIAL3
  YANEURAOU_ENGINE_MATERIAL4
  YANEURAOU_ENGINE_MATERIAL5
  YANEURAOU_ENGINE_MATERIAL6
  YANEURAOU_ENGINE_MATERIAL7
  YANEURAOU_ENGINE_MATERIAL8
  YANEURAOU_ENGINE_MATERIAL9
#  YANEURAOU_ENGINE_MATERIAL10
  YANEURAOU_MATE_ENGINE
  TANUKI_MATE_ENGINE
  USER_ENGINE
)

TARGETS=(
  normal
  tournament
  evallearn
)

ARCHCPUS=(
  ZEN3
  ZEN2
  ZEN1
  AVX512VNNI
  AVX512
  AVXVNNI
  AVX2
  SSE42
  SSE41
  SSSE3
  SSE2
  OTHER
)

declare -A EDITIONSTR;
EDITIONSTR=(
  ["YANEURAOU_ENGINE_NNUE"]="YANEURAOU_ENGINE_NNUE"
  ["YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32"]="YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32"
  ["YANEURAOU_ENGINE_NNUE_HALFKPE9"]="YANEURAOU_ENGINE_NNUE_HALFKPE9"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="YANEURAOU_ENGINE_NNUE_KP256"
  ["YANEURAOU_ENGINE_KPPT"]="YANEURAOU_ENGINE_KPPT"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="YANEURAOU_ENGINE_KPP_KKPT"
  ["YANEURAOU_ENGINE_MATERIAL"]="YANEURAOU_ENGINE_MATERIAL"
  ["YANEURAOU_ENGINE_MATERIAL2"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=2"
  ["YANEURAOU_ENGINE_MATERIAL3"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=3"
  ["YANEURAOU_ENGINE_MATERIAL4"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=4"
  ["YANEURAOU_ENGINE_MATERIAL5"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=5"
  ["YANEURAOU_ENGINE_MATERIAL6"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=6"
  ["YANEURAOU_ENGINE_MATERIAL7"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=7"
  ["YANEURAOU_ENGINE_MATERIAL8"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=8"
  ["YANEURAOU_ENGINE_MATERIAL9"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=9"
  ["YANEURAOU_ENGINE_MATERIAL10"]="YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=10"
  ["YANEURAOU_MATE_ENGINE"]="YANEURAOU_MATE_ENGINE"
  ["TANUKI_MATE_ENGINE"]="TANUKI_MATE_ENGINE"
  ["USER_ENGINE"]="USER_ENGINE"
);

declare -A DIRSTR;
DIRSTR=(
  ["YANEURAOU_ENGINE_NNUE"]="NNUE"
  ["YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32"]="NNUE_HalfKP_VM"
  ["YANEURAOU_ENGINE_NNUE_HALFKPE9"]="NNUE_HALFKPE9"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="NNUE_KP256"
  ["YANEURAOU_ENGINE_KPPT"]="KPPT"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="KPP_KKPT"
  ["YANEURAOU_ENGINE_MATERIAL"]="MaterialLv1"
  ["YANEURAOU_ENGINE_MATERIAL2"]="MaterialLv2"
  ["YANEURAOU_ENGINE_MATERIAL3"]="MaterialLv3"
  ["YANEURAOU_ENGINE_MATERIAL4"]="MaterialLv4"
  ["YANEURAOU_ENGINE_MATERIAL5"]="MaterialLv5"
  ["YANEURAOU_ENGINE_MATERIAL6"]="MaterialLv6"
  ["YANEURAOU_ENGINE_MATERIAL7"]="MaterialLv7"
  ["YANEURAOU_ENGINE_MATERIAL8"]="MaterialLv8"
  ["YANEURAOU_ENGINE_MATERIAL9"]="MaterialLv9"
  ["YANEURAOU_ENGINE_MATERIAL10"]="MaterialLv10"
  ["YANEURAOU_MATE_ENGINE"]="YaneuraOu_MATE"
  ["TANUKI_MATE_ENGINE"]="tanuki_MATE"
  ["USER_ENGINE"]="USER"
);

declare -A FILESTR;
FILESTR=(
  ["YANEURAOU_ENGINE_NNUE"]="YaneuraOu_NNUE"
  ["YANEURAOU_ENGINE_NNUE_HALFKP_VM_256X2_32_32"]="YaneuraOu_NNUE_HalfKP_VM"
  ["YANEURAOU_ENGINE_NNUE_HALFKPE9"]="YaneuraOu_NNUE_HalfKPE9"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="YaneuraOu_NNUE_KP256"
  ["YANEURAOU_ENGINE_KPPT"]="YaneuraOu_KPPT"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="YaneuraOu_KPP_KKPT"
  ["YANEURAOU_ENGINE_MATERIAL"]="YaneuraOu_MaterialLv1"
  ["YANEURAOU_ENGINE_MATERIAL2"]="YaneuraOu_MaterialLv2"
  ["YANEURAOU_ENGINE_MATERIAL3"]="YaneuraOu_MaterialLv3"
  ["YANEURAOU_ENGINE_MATERIAL4"]="YaneuraOu_MaterialLv4"
  ["YANEURAOU_ENGINE_MATERIAL5"]="YaneuraOu_MaterialLv5"
  ["YANEURAOU_ENGINE_MATERIAL6"]="YaneuraOu_MaterialLv6"
  ["YANEURAOU_ENGINE_MATERIAL7"]="YaneuraOu_MaterialLv7"
  ["YANEURAOU_ENGINE_MATERIAL8"]="YaneuraOu_MaterialLv8"
  ["YANEURAOU_ENGINE_MATERIAL9"]="YaneuraOu_MaterialLv9"
  ["YANEURAOU_ENGINE_MATERIAL10"]="YaneuraOu_MaterialLv10"
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
              set -f
              echo "* target: ${TARGET}"
              for ARCHCPU in ${ARCHCPUS[@]}; do
                for ARCHCPUSPTN in ${ARCHCPUSARR[@]}; do
                  set +f
                  if [[ $ARCHCPU == $ARCHCPUSPTN ]]; then
                    echo "* cpu: ${CPU}"
                    TGSTR=${FILESTR[$EDITION]}-msys2-${CSTR}-${TARGET}
                    ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITIONSTR[$EDITION]} ${EXTRA}
                    nice ${MAKE} -f ${MAKEFILE} -j$(nproc) ${TARGET} YANEURAOU_EDITION=${EDITIONSTR[$EDITION]} COMPILER=${COMPILER} TARGET_CPU=${ARCHCPU} ${EXTRA} >& >(tee ${BUILDDIR}/${TGSTR}.log) || exit $?
                    cp YaneuraOu-by-gcc.exe ${BUILDDIR}/${TGSTR}.exe
                    ${MAKE} -f ${MAKEFILE} clean YANEURAOU_EDITION=${EDITIONSTR[$EDITION]} ${EXTRA}
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
        break
      fi
      set -f
    done
  done
done

popd
popd
