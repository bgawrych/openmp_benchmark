#include "QuickSort.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>
#include <functional>

void QuickSort::RunParallel() {
    RunParallelDouble();
    RunParallelSingle();
}

float PartitionArrayHelper(float *inarray, float* outarray, int left, int right) {
    float partitionValue = inarray[right];
    int currentPosition = left - 1;
    outarray[0] = inarray[0];
    for(int i=left; i < right; i++) {
	outarray[i] = inarray[i];
        if (outarray[i] < partitionValue) {
            currentPosition++;
            std::swap(outarray[i], outarray[currentPosition]);
        }
    }
    outarray[right] = outarray[currentPosition + 1];
    outarray[currentPosition + 1] = inarray[right];
    return currentPosition + 1;
}

float PartitionArray(float *array, int left, int right) {
    float partitionValue = array[right];

    int currentPosition = left - 1;
    for(int i=left; i < right; i++) {
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

void ParallelQuicksortDoubleHelper(float* original, float* copy, int left, int right, int node_id) {
    int nthreads = omp_get_num_threads();
    if ( left < right ) {
        float p = PartitionArrayHelper(original, copy, left, right);
        mpragma(omp task default(none) firstprivate(copy, left, p, node_id, nthreads) final(node_id >= nthreads)){
            ParallelQuicksortDouble(copy, left, p - 1, node_id*2);
        }
        mpragma(omp task default(none) firstprivate(copy, right, p, node_id, nthreads) final(node_id >= nthreads)){
            ParallelQuicksortDouble(copy, p + 1, right, node_id*2+1);
        }
    }
}


void QuickSort::RunParallelDouble() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_DOUBLE",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel shared(output_data, input_data, size)) {
                mpragma(omp single) {
                    ParallelQuicksortDoubleHelper(input_data, output_data, 0, size-1, 1);
                }
                mpragma(omp taskwait)
            }
        }
   )
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

void ParallelQuicksortSingleHelper(float* original, float* copy, int left, int right, int node_id) {
    int nthreads = omp_get_num_threads();
    if ( left < right ) {

        float p = PartitionArrayHelper(original, copy, left, right);
        mpragma(omp task default(none) firstprivate(copy, left, p, node_id, nthreads) final(node_id >= nthreads)){
            ParallelQuicksortSingle(copy, left, p - 1, node_id*2);
        }
        ParallelQuicksortSingle(copy, p + 1, right, node_id*2+1);
    }
}

void QuickSort::RunParallelSingle() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_SINGLE",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel shared(output_data, input_data, size)) {
                mpragma(omp single) {
                    ParallelQuicksortSingleHelper(input_data, output_data, 0, size-1, 1);
                }
                mpragma(omp taskwait)
            }
        }
   )
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

void QuicksortHelper(float *original, float *copy, int left, int right){
    if ( left < right ) {
        float p = PartitionArrayHelper(original, copy, left, right);
        Quicksort(copy, left, p - 1); // Left branch
        Quicksort(copy, p + 1, right); // Right branch
    }
}

void QuickSort::RunSerial() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "SERIAL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            QuicksortHelper(input_data, output_data, 0, size-1);
        }
   )
    // Print2DArray(&data, 1, size);
    // std::cout << "\n\n=============\n\n";
}

bool QuickSort::Validate() {
    float* out_serial = new float[size];
    float* out_parallel_1 = new float[size];
    float* out_parallel_2 = new float[size];

    rounds = 1;
    warmup = 0;

    float* tmp = output_data;

    output_data = out_serial;
    RunSerial();

    output_data = out_parallel_1;
    RunParallelSingle();

    output_data = out_parallel_2;
    RunParallelDouble();

    bool is_valid = CompareArray(out_serial, out_parallel_1, size);
    is_valid &= CompareArray(out_serial, out_parallel_2, size);

    output_data = tmp;
    delete[] out_serial;
    delete[] out_parallel_1;
    delete[] out_parallel_2;

    return is_valid;
}

void QuickSort::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;
    rounds = properties["rounds"].GetInt();
    warmup = properties["warmup"].GetInt();
    size = properties["size"].GetInt();
    Logger::INFO << VAR(size);

    std::stringstream os;
    os << VAR_(size);
    descriptor = os.str();

    Reinitialize();
}


void QuickSort::Reinitialize() {
    if(initialized) {
        delete[] input_data;
        delete[] output_data;
    }

    input_data = new float[size];
    output_data = new float[size];
    FillRandomArray(input_data, size);
    this->initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<QuickSort>("QuickSort");
}

REGISTER_BENCHMARK(QuickSort, CreateBench);
