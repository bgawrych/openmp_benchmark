#!/bin/bash

set -uex

num_cpus=$(grep -c ^processor /proc/cpuinfo)

rm -rf build/*
./linux_build.sh -c gcc -t
benchmarks=(linear conv batchnorm matmul wave)


for b in ${benchmarks[@]}
do
   #create results folders
   result_folder=work_results_offload/gcc_results/$b
   mkdir -p $result_folder

   ./build/OpenMPOffloadBench offloading_descriptors/$b.json $result_folder/out.csv
done

rm -rf build/*
./linux_build.sh -c clang -t


benchmarks=(linear conv batchnorm matmul wave)
for b in ${benchmarks[@]}
do
   #create results folders
   result_folder=work_results_offload/clang_results/$b
   mkdir -p $result_folder

   ./build/OpenMPOffloadBench offloading_descriptors/$b.json $result_folder/out.csv

done
