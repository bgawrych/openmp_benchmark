# OpenMP Benchmarks

This repository contains reproduction scripts and collected results from master thesis titled **"Assessment of OpenMP constructs' performance on modern  multi-core architectures"**.

## Folder Structure
- **compilers/** - this folder contains script which will download, build and install GCC and Clang compilers and download and install ICC and ICX compilers
- **offloading_compilers/** - this folder contains script which will download, build and install GCC and Clang compilers with offloading to NVIDIA GPU's 
- **openmp_benchmarks/** - this folder contains C/C++ benchmarks code, benchmarks descriptors, and scripts to run these benchmarks

## Requirment

To run these benchmarks following packages are required (tested on ubuntu docker image):
- build-essential
- git
- wget
- g++/gcc >= 7.3

## Building OpenMP benchmarks
Building script has been prepared and is located in the *openmp_benchmarks* folder:
Usage:
```
./linux_build.sh [-c=<compiler>] [-t]
```
* -c, --compiler, Build benchmarks with specified compilatior
-  -t, --target, Build OpenMP benchmarks with offloading directives - this option requires to have compilers supporting offload

## Runing OpenMP benchmarks
To run OpenMP benchmarks use following command:
```
OMP_NUM_THREADS=<NUM_THREADS> ./build/OpenMPBench <descriptor_path> <outfile_csv_path>
```
- NUM_THREADS - how many OpenMP threads should be used
- descriptor_path - path to the descriptor file that should be used - one of the already prepare descriptor for master thesis can be used (**openmp_benchmarks/descriptors/**)
- outfile_csv_path - path to the output file with benchmarks results

## Building compilers
To build compilers from source special script has been prepared. It's called *prepare.sh* and is located in both directories: **compilers/** and **offloading_compilers/**:
Usage:
```
bash prepare.sh
```
Remember to edit CUDA installation path in *prepare.sh* script in **offloading_compilers/** directory

## Results
Results collected for master thesis are in **openmp_benchmarks/results/** directory. They can be viewed in raw files, but special Jupyter Notebook was prepared to visualize results as a charts.
### Instruction
- Download and install conda environment (skip if already done):
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```
- Craete and activate new environment to keep base env. clean:
```
conda create -n openmp_results
conda activate openmp_results
```
- Install dependencies required by the script:
```
conda install -c conda-forge notebook  
conda install plotly matplotlib pandas
```
- Run jupyter-notebook by following command:
```
jupyter-notebook
```
- Open link which appeared in terminal (e.g. http://127.0.0.1:8888/?token=21312....
- In the browser there will be listed all files from folder where notebook was run. Open ```openmp_benchmarks/processing_results.ipynb``` file 
- Before running the script, modify the following settings to generate results in the desired form. It is possible to generate charts only for a specific benchmark by editing the ```benchmarks``` variable in the script or for a specific processor using the ```CPUS``` variable. It is also possible to generate charts showing acceleration relative to sequential execution, or showing average execution time with standard deviation - this option is controlled by the ```speedup``` variable variable, set to ```True``` or ```False```. Additionally, it is possible to separate charts for compilers using the ```separate_compilers``` variable, which is especially useful useful when one chart contains too much information.
- Run script by combination of Ctrl+A and Shift + Enter keys or by using the icons on the top of the page
