#!/bin/bash

JOBS=`grep -c ^processor /proc/cpuinfo 2>/dev/null`

EDITIONS='*'

while getopts c:e:t: OPT
do
  case $OPT in
    e) EDITIONS="$OPTARG"
      ;;
  esac
done

set -f
IFS=, eval 'EDITIONSARR=($EDITIONS)'

pushd `dirname $0`
pushd ..

EDITIONS=(
  YANEURAOU_ENGINE_NNUE
  YANEURAOU_ENGINE_NNUE_HALFKPE9
  YANEURAOU_ENGINE_NNUE_KP256
  YANEURAOU_ENGINE_KPPT
  YANEURAOU_ENGINE_KPP_KKPT
  YANEURAOU_ENGINE_MATERIAL
  MATE_ENGINE
  USER_ENGINE
)

declare -A DIRSTR;
DIRSTR=(
  ["YANEURAOU_ENGINE_NNUE"]="NNUE"
  ["YANEURAOU_ENGINE_NNUE_HALFKPE9"]="NNUE_HALFKPE9"
  ["YANEURAOU_ENGINE_NNUE_KP256"]="NNUE_KP256"
  ["YANEURAOU_ENGINE_KPPT"]="KPPT"
  ["YANEURAOU_ENGINE_KPP_KKPT"]="KPP_KKPT"
  ["YANEURAOU_ENGINE_MATERIAL"]="KOMA"
  ["MATE_ENGINE"]="MATE"
);

set -f

for EDITION in ${EDITIONS[@]}; do
  for EDITIONPTN in ${EDITIONSARR[@]}; do
    set +f
    if [[ $EDITION == $EDITIONPTN ]]; then
      set -f
      echo "* edition: ${EDITION}"
      BUILDDIR=../build/android/${DIRSTR[$EDITION]}
      mkdir -p ${BUILDDIR}
      ndk-build clean ENGINE_TARGET=${EDITION}
      ndk-build ENGINE_TARGET=${EDITION} -j${JOBS} > >(tee build/android/${DIRSTR[$EDITION]}/${DIRSTR[$EDITION]}.log) || exit $?
      cp libs/**/* build/android/${DIRSTR[$EDITION]}
      ndk-build clean ENGINE_TARGET=${EDITION}
      break
    fi
    set -f
  done
done

popd
popd
