#include "QuickSort.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>
#include <functional>

void QuickSort::RunParallel() {
//    RunParallelDouble();
//    RunParallelSingle();
}

float PartitionArray(float *array, int left, int right) {
    float partitionValue = array[right];

    int currentPosition = left - 1;
    for(int i=left; i < right - 1; i++) {
        if (array[i] < partitionValue) {
            currentPosition++;
            std::swap(array[i], array[currentPosition]);
        }
    }
    std::swap(array[currentPosition+1], array[right]);
    return currentPosition + 1;
}

void ParallelQuicksortDouble(float* array, int left, int right, int node_id) {
    int nthreads = omp_get_num_threads();
    if ( left < right ) {
        float p = PartitionArray(array, left, right);
        mpragma(omp task default(none) firstprivate(array, left, p, node_id, nthreads) final(node_id >= nthreads)){
            ParallelQuicksortDouble(array, left, p - 1, node_id*2);
        }
        mpragma(omp task default(none) firstprivate(array, right, p, node_id, nthreads) final(node_id >= nthreads)){
            ParallelQuicksortDouble(array, p + 1, right, node_id*2+1);
        }
    }
}

void QuickSort::RunParallelDouble() {
//    auto excel = *this->file;
//
//    BENCHMARK_STRUCTURE(
//        excel,      // name of csv logger
//        "Parallel_Double",   // name of benchmark
//        warmup,     // name of warmup rounds variable
//        rounds,     // name of benchmark rounds variable
//        ELAPSED,    // variable name to store execution time
//        {
//            std::memcpy(data, input_data, size * sizeof(float));
//            mpragma(omp parallel shared(data, size)) {
//                mpragma(omp single) {
//                    ParallelQuicksortDouble(data, 0, size-1, 1);
//                }
//                mpragma(omp taskwait)
//            }
//        }
//   )
//    Print2DArray(&data, 1, size);
//    std::cout << "\n\n=============\n\n";
}


void ParallelQuicksortSingle(float* array, int left, int right, int node_id) {
    int nthreads = omp_get_num_threads();
    if ( left < right ) {

        float p = PartitionArray(array, left, right);
        mpragma(omp task default(none) firstprivate(array, left, p, node_id, nthreads) final(node_id >= nthreads)){
            ParallelQuicksortSingle(array, left, p - 1, node_id*2);
        }
        ParallelQuicksortSingle(array, p + 1, right, node_id*2+1);
    }
}
void QuickSort::RunParallelSingle() {
//    auto excel = *this->file;
//
//    BENCHMARK_STRUCTURE(
//        excel,      // name of csv logger
//        "Parallel_Single",   // name of benchmark
//        warmup,     // name of warmup rounds variable
//        rounds,     // name of benchmark rounds variable
//        ELAPSED,    // variable name to store execution time
//        {
//            std::memcpy(data, input_data, size * sizeof(float));
//            mpragma(omp parallel shared(data, size)) {
//                mpragma(omp single) {
//                    ParallelQuicksortSingle(data, 0, size-1, 1);
//                }
//                mpragma(omp taskwait)
//            }
//        }
//   )
//    Print2DArray(&data, 1, size);
//    std::cout << "\n\n=============\n\n";
}

void Quicksort(float *array, int left, int right){
    if ( left < right ) {
        float p = PartitionArray(array, left, right);
        Quicksort(array, left, p - 1); // Left branch
        Quicksort(array, p + 1, right); // Right branch
    }
}

void QuickSort::RunSerial() {
//    auto excel = *this->file;
//
//
//    BENCHMARK_STRUCTURE(
//        excel,      // name of csv logger
//        "Serial",   // name of benchmark
//        warmup,     // name of warmup rounds variable
//        rounds,     // name of benchmark rounds variable
//        ELAPSED,    // variable name to store execution time
//        {
//            std::memcpy(data, input_data, size * sizeof(float));
//            Quicksort(data, 0, size-1);
//        }
//   )
    // Print2DArray(&data, 1, size);
    // std::cout << "\n\n=============\n\n";
}

bool QuickSort::Validate() {
//    float* out_serial = new float[size];
//    float* out_parallel_1 = new float[size];
//    float* out_parallel_2 = new float[size];
//
//    rounds = 1;
//    warmup = 0;
//
//    float* tmp = data;
//
//    data = out_serial;
//    RunSerial();
//
//    data = out_parallel_1;
//    RunParallelSingle();
//
//    data = out_parallel_2;
//    RunParallelDouble();
//
//    bool is_valid = CompareArray(out_serial, out_parallel_1, size);
//    is_valid &= CompareArray(out_serial, out_parallel_2, size);
//
//    data = tmp;
//    delete[] out_serial;
//    delete[] out_parallel_1;
//    delete[] out_parallel_2;
//
//    return is_valid;
}

void QuickSort::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
//    this->file = file;
//    rounds = properties["rounds"].GetInt();
//    warmup = properties["warmup"].GetInt();
//    size = properties["size"].GetInt();
//    Logger::INFO << VAR(size);
//
//    Reinitialize();
}


void QuickSort::Reinitialize() {
//    if(initialized) {
//        delete[] input_data;
//        delete[] data;
//    }
//
//    input_data = new float[size];
//    data = new float[size];
//    FillRandomArray(input_data, size);
//    this->initialized = true;
}

//static std::shared_ptr<Benchmark> CreateBench() {
//    return std::make_shared<QuickSort>("QuickSort");
//}

//REGISTER_BENCHMARK(QuickSort, CreateBench);
