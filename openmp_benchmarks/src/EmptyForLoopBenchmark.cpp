#include "EmptyForLoopBenchmark.hpp"
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <thread>

void EmptyForLoopBenchmark::RunParallel() {
    auto excel = *this->file;
    int dummy = 0;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL",
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel for private(dummy))
            for(int i = 0; i < iterations; i++ )
            {
                LOOP_UNOPTIMIZER(dummy);
            }
        }
    )
}

void EmptyForLoopBenchmark::RunSerial() {
    auto excel = *this->file;
    int dummy=0;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "SERIAL",
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            for(int i = 0; i < iterations; i++ )
            {
                LOOP_UNOPTIMIZER(dummy);
            }
        }
    )
}

void EmptyForLoopBenchmark::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;
    rounds = properties["rounds"].GetInt();
    warmup = properties["warmup"].GetInt();
    iterations = properties["iterations"].GetInt();

    std::stringstream os;
    os << VAR_(iterations);
    descriptor = os.str();
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<EmptyForLoopBenchmark>("EmptyForLoopBenchmark");
}

REGISTER_BENCHMARK(EmptyForLoopBenchmark, CreateBench);
