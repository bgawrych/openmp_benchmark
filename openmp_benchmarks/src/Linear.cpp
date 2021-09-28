#include "Linear.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>
#include <functional>

void Linear::RunParallel() {
    RunParallel_1();
    RunParallel_2();
    RunParallel_3();
    RunParallel_4();
}


void Linear::RunParallel_1() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_STATIC",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel for schedule(static))
            for(int i=0; i < size; i++) {
                output[i] = input[i]*13 + 2;
            }
        }
   )
}


void Linear::RunParallel_2() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_STATIC_SIMD",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel for simd schedule(static))
            for(int i=0; i < size; i++) {
                output[i] = input[i]*13 + 2;
            }
        }
   )
}



void Linear::RunParallel_3() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_STATIC_100",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel for schedule(static, 100))
            for(int i=0; i < size; i++) {
                output[i] = input[i]*13 + 2;
            }
        }
   )
}

void Linear::RunParallel_4() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_STATIC_1000",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel for schedule(static, 1000))
            for(int i=0; i < size; i++) {
                output[i] = input[i]*13 + 2;
            }
        }
   )
}

void Linear::RunSerial() {
    auto excel = *this->file;
    
    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "SERIAL",   // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            for(int i=0; i < size; i++) {
                output[i] = input[i]*13 + 2;
            }
        }
   )
}

bool Linear::Validate() {
    float* out_serial = new float[size];
    float* out_parallel_1 = new float[size];
    float* out_parallel_2 = new float[size];
    float* out_parallel_3 = new float[size];
    float* out_parallel_4 = new float[size];
    rounds = 1;
    warmup = 0;

    float* tmp = output;

    output = out_serial;
    RunSerial();

    output = out_parallel_1;
    RunParallel_1();

    output = out_parallel_2;
    RunParallel_2();

    output = out_parallel_3;
    RunParallel_3();

    output = out_parallel_4;
    RunParallel_4();
    
    bool is_valid = CompareArray(out_serial, out_parallel_1, size);
    is_valid &= CompareArray(out_serial, out_parallel_2, size);
    is_valid &= CompareArray(out_serial, out_parallel_3, size);
    is_valid &= CompareArray(out_serial, out_parallel_4, size);

    output = tmp;
    delete[] out_serial;
    delete[] out_parallel_1;
    delete[] out_parallel_2;
    delete[] out_parallel_3;
    delete[] out_parallel_4;

    return is_valid;
}

void Linear::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;
    rounds = properties["rounds"].GetInt();
    warmup = properties["warmup"].GetInt();

    size =  properties["size"].GetInt();
    Logger::INFO << VAR(size);

    std::stringstream os;
    os << VAR_(size);
    descriptor = os.str();

    Reinitialize();
}

void Linear::Reinitialize() {
    if(initialized) {
        delete[] input;
        delete[] output;
    }

    input = new float[size];
    output = new float[size];

    FillRandomArray(input, size);
    this->initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<Linear>("Linear");
}

REGISTER_BENCHMARK(Linear, CreateBench);
