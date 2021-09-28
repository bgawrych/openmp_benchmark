set -uex

eval "$(conda shell.bash hook)"
conda create --name clang_offload --yes
conda activate clang_offload
conda install -c conda-forge fasttext --yes

WORKDIR=`pwd`
COMPILERS_PATH=$WORKDIR/compilers
APPS_PATH=$WORKDIR/apps
GCC_PATH=$COMPILERS_PATH/gcc_offload
CLANG_PATH=$COMPILERS_PATH/clang_offload
UTILS_PATH=$WORKDIR/utils


echo "====Preparing folder structure====="

rm -r $COMPILERS_PATH
rm -r $APPS_PATH
rm -r $UTILS_PATH
mkdir -p $UTILS_PATH
mkdir -p $APPS_PATH
mkdir -p $GCC_PATH
mkdir -p $CLANG_PATH

cd $UTILS_PATH

echo "${APPS_PATH}/include"
echo "=====downloading & installing libefi dependency====="
git clone https://github.com/wolfgangst/libelf
cd libelf
./configure --prefix=$APPS_PATH
make -j4
make install
cd ..


echo "=====Downloading & installing libffi dependency====="
wget https://github.com/libffi/libffi/releases/download/v3.3/libffi-3.3.tar.gz
tar xf libffi-3.3.tar.gz
cd libffi-3.3
./configure --prefix=$APPS_PATH
make -j4
make install
cd ..


# ##################################################
# ################################################## CLANG
# ##################################################


echo "----Build LLVM with support for offloading to NVIDIA GPUs.-----"

cd $CLANG_PATH
git clone https://github.com/llvm/llvm-project.git --branch llvmorg-11.1.0 llvm
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$APPS_PATH \
    -DLIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR:PATH=$APPS_PATH/include \
    -DLIBOMPTARGET_DEP_LIBELF_LIBRARIES:PATH=$APPS_PATH/lib/libelf.so \
    -DLIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR:PATH=$APPS_PATH/include \
    -DLIBOMPTARGET_DEP_LIBFFI_LIBRARIES:PATH=$APPS_PATH/lib/libffi.so \
    -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_61 \
    -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=61 \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_PROJECTS="clang;openmp" ../llvm/llvm
make -j4

cd ..
mkdir build2
cd build2
CC=../build/bin/clang CXX=../build/bin/clang++  \
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$APPS_PATH \
    -DLIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR:PATH=$APPS_PATH/include \
    -DLIBOMPTARGET_DEP_LIBELF_LIBRARIES:PATH=$APPS_PATH/lib/libelf.so \
    -DLIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR:PATH=$APPS_PATH/include \
    -DLIBOMPTARGET_DEP_LIBFFI_LIBRARIES:PATH=$APPS_PATH/lib/libffi.so \
    -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_61 \
    -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=61 \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_PROJECTS="clang;openmp" ../llvm/llvm
make  -j4
make install


# ##################################################
# ################################################## GCC
# ##################################################

echo "----Build GCC with support for offloading to NVIDIA GPUs.-----"
# #
conda create --name gcc_offload --yes
conda activate gcc_offload
conda install -c conda-forge fasttext --yes
conda install -c conda-forge flex --yes
# Location of the installed CUDA toolkit
CUDA_PATH=/usr/local/cuda

cd $GCC_PATH
git clone https://github.com/MentorEmbedded/nvptx-tools
cd nvptx-tools
./configure \
        --with-cuda-driver-include=$CUDA_PATH/include \
        --with-cuda-driver-lib=$CUDA_PATH/lib64 \
        --prefix=$APPS_PATH
make -j4
make install
cd ..


# Set up the GCC source tree
git clone https://github.com/MentorEmbedded/nvptx-newlib
wget -c https://github.com/gcc-mirror/gcc/archive/refs/tags/releases/gcc-10.2.0.tar.gz -O gcc.tar.gz
tar xf gcc.tar.gz
mv gcc-releases-gcc-10.2.0 gcc
cd gcc
contrib/download_prerequisites
ln -s ../nvptx-newlib/newlib newlib
target=$(./config.guess)
cd ..


# Build nvptx GCC
mkdir build-nvptx-gcc
cd build-nvptx-gcc
../gcc/configure \
   --target=nvptx-none \
   --with-build-time-tools=$APPS_PATH/nvptx-none/bin \
   --enable-as-accelerator-for=$target \
   --disable-sjlj-exceptions \
   --enable-newlib-io-long-long \
   --enable-languages="c,c++,fortran,lto" \
   --prefix=$APPS_PATH
make -j4
make install
cd ..
mkdir build-host-gcc
cd  build-host-gcc
../gcc/configure \
    --enable-offload-targets=nvptx-none \
    --with-cuda-driver-include=$CUDA_PATH/include \
    --with-cuda-driver-lib=$CUDA_PATH/lib64 \
    --disable-bootstrap \
    --disable-multilib \
    --enable-languages="c,c++,fortran,lto" \
    --prefix=$APPS_PATH
make -j4
make install
#export PATH=$APPS_PATH/bin:$PATH
#export LD_LIBRARY_PATH=$APPS_PATH/lib:$LD_LIBRARY_PATH
