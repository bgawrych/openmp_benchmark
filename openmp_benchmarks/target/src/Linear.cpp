#include "Linear.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>
#include <functional>

void Linear::RunParallel() {
    RunParallel_1();
    RunParallel_2();
}


void Linear::RunParallel_1() {
    auto excel = *this->file;

    auto fn = [&]() {
        int _size = size;
        float* raw_input = input;
        float* raw_output = output;
        #if defined(__clang__)
	/*
	 * compiled code from 'else' branch with clang:
	 * Libomptarget fatal error 1: failure of target construct while offloading is mandatory
	 */
	    #pragma omp target teams distribute parallel for \
                map(tofrom:raw_input[0:_size]) map(tofrom:raw_output[0:_size])
            for (int i=0; i < _size; i++) {
                raw_output[i] = raw_input[i]*13 + 2;
            }
       #else
	/* above code produces memory leaks when compiled with gcc
	 *
	 */
            #pragma omp target enter data map(to: raw_input[0:_size]) map(alloc: raw_output[0:_size])
            #pragma omp target //data map(from: raw_output[0:_size])
            {
                #pragma omp teams distribute parallel for
                for (int i=0; i < _size; i++) {
                    raw_output[i] = raw_input[i]*13 + 2;
                }
            }
            #pragma omp target update from(raw_output[0:_size])
            #pragma omp target exit data map(delete: raw_output) map(delete:raw_input)
       #endif

    };

    BenchmarkIt(excel, "PARALLEL_FOR", warmup, rounds, fn);
}


void Linear::RunParallel_2() {
    auto excel = *this->file;

    auto fn = [&]() {
        int _size = size;
        float* raw_input = input;
        float* raw_output = output;
       #if defined(__clang__)
            #pragma omp target teams distribute parallel for simd \
                map(tofrom:raw_input[0:_size]) map(tofrom:raw_output[0:_size])
            for (int i=0; i < _size; i++) {
                raw_output[i] = raw_input[i]*13 + 2;
            }
       #else

            #pragma omp target enter data map(to: raw_input[0:_size]) map(alloc: raw_output[0:_size])
            #pragma omp target //data map(from: raw_output[0:_size])
            {
                #pragma omp teams distribute parallel for simd
                for (int i=0; i < _size; i++) {
                    raw_output[i] = raw_input[i]*13 + 2;
                }
            }
            #pragma omp target update from(raw_output[0:_size])
            #pragma omp target exit data map(delete: raw_output) map(delete:raw_input)
       #endif

    };

    BenchmarkIt(excel, "PARALLEL_FOR_SIMD", warmup, rounds, fn);
}

void Linear::RunSerial() {
    auto excel = *this->file;

    auto fn = [&]() {
        int _size = size;
        float* raw_input = input;
        float* raw_output = output;
        for(int i=0; i < _size; i++) {
            raw_output[i] = raw_input[i]*13 + 2;
        }
    };
    BenchmarkIt(excel, "SERIAL", warmup, rounds, fn);
}

bool Linear::Validate() {
    float* out_serial = new float[size];
    float* out_parallel_1 = new float[size];
    float* out_parallel_2 = new float[size];
    rounds = 1;
    warmup = 0;

    float* tmp = output;

    output = out_serial;
    RunSerial();

    output = out_parallel_1;
    RunParallel_1();

    output = out_parallel_2;
    RunParallel_2();


    bool is_valid = CompareArray(out_serial, out_parallel_1, size);
    is_valid &= CompareArray(out_serial, out_parallel_2, size);

    output = tmp;
    delete[] out_serial;
    delete[] out_parallel_1;
    delete[] out_parallel_2;

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
