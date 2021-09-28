#!/bin/bash
rm -rf build

usage () {
    echo "Usage: ./linux_build [-c=<compiler>] [-t]"
    echo "-c, --compiler,                  Build benchmarks with specified compilatior"
    echo "-t, --target,                    Build OpenMP benchmarks with offloading directives"
    exit
}
BUILD_OFFLOADING=0
options=$(getopt -l "help,compiler:,target" -o "hc:t" -a -- "$@")
#eval set -- "$options"

if [ $? -ne 0 ]
then
    usage
    exit
fi

eval set -- "$options"

while true
do
case $1 in
    -c|--compiler)
        echo "Setting compiler to $2" >&2
        COMPILER=$2
        ;;
    -t|--target)
        echo "Building with offload directives"
        BUILD_OFFLOADING=1
        ;;
    --)
        shift
        break
        ;;
    -h|--help)
        usage
        exit
    ;;
  esac
  shift
done

if [[ "$COMPILER" = "icc" ]] ; then
    export CC=icc
    export CXX=icpc
elif [[ "$COMPILER" = "icx" ]] ; then
    export CC=icx
    export CXX=icpx
elif [[ "$COMPILER" = "clang" ]] ; then
    export CC=clang
    export CXX=clang++
else 
    export CC=gcc
    export CXX=g++
fi



mkdir -p build
cd build
cmake -GNinja -DBUILD_OFFLOADING=${BUILD_OFFLOADING} ..
ninja
cd ..
