
WORKDIR=`pwd`
COMPILERS_PATH=$WORKDIR/compilers
APPS_PATH=$WORKDIR/apps
GCC_PATH=$COMPILERS_PATH/gcc_offload
CLANG_PATH=$COMPILERS_PATH/clang_offload
UTILS_PATH=$WORKDIR/utils


export PATH=$APPS_PATH/bin:$PATH
export LD_LIBRARY_PATH=$APPS_PATH/lib64:$APPS_PATH/lib:$LD_LIBRARY_PATH


#clang vector.c -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
#gcc vector.c -fopenmp -foffload=nvptx-none
