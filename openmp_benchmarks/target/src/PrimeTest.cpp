#include "PrimeTest.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>
#include <functional>

#pragma omp declare target
bool CheckPrime(int value) {
    if (value <= 3) {
        return value > 1;
    }

    if (value % 2 == 0 || value % 3 == 0) {
        return false;
    }

    int i = 5;
    while (i * i <= value) {
        if(value % i == 0 || value % (i + 2) == 0) {
            return false;
        }
        i += 6;
    };
    return true;
}
#pragma omp end declare target

void PrimeTest::RunParallel() {
   // RunParallel_1();
   // RunParallel_2();
   // RunParallel_3();
}


void PrimeTest::RunParallel_1() {
 //   auto excel = *this->file;
 //       
 //   auto fn = [=]() {
 //       int _size = size;
 //       int* raw_input = input;
 //       bool* raw_output = output;
 //       #pragma omp target teams distribute parallel for \
 //               map(to:raw_input[0:_size]) map(tofrom:raw_output[0:_size])
 //           for(int i=0; i < _size; i++) {
 //               raw_output[i] = CheckPrime(raw_input[i]);
 //           }
 //   };

 //   BenchmarkIt(excel, "PARALLEL_NO_DIST_SCHEDULE", warmup, rounds, fn);
}

void PrimeTest::RunParallel_2() {
 //   auto excel = *this->file;
 //       
 //   auto fn = [=]() {
 //       int _size = size;
 //       int* raw_input = input;
 //       bool* raw_output = output;
 //       #pragma omp target teams distribute parallel for dist_schedule(static) \
 //               map(to:raw_input[0:_size]) map(tofrom:raw_output[0:_size])
 //           for(int i=0; i < _size; i++) {
 //               raw_output[i] = CheckPrime(raw_input[i]);
 //           }
 //   };

 //   BenchmarkIt(excel, "PARALLEL_DIST_SCHEDULE", warmup, rounds, fn);
}


void PrimeTest::RunParallel_3() {
   // auto excel = *this->file;
   //     
   // auto fn = [=]() {
   //     int _size = size;
   //     int* raw_input = input;
   //     bool* raw_output = output;
   //     #pragma omp target teams distribute parallel for schedule(static) \
   //             map(to:raw_input[0:_size]) map(tofrom:raw_output[0:_size])
   //         for(int i=0; i < _size; i++) {
   //             raw_output[i] = CheckPrime(raw_input[i]);
   //         }
   // };

   // BenchmarkIt(excel, "PARALLEL_GUIDED", warmup, rounds, fn);
}

void PrimeTest::RunSerial() {
 //   auto excel = *this->file;
 //       
 //   auto fn = [&]() {
 //       int _size = size;
 //       int* raw_input = input;
 //       bool* raw_output = output;

 //       for(int i=0; i < size; i++) {
 //           output[i] = CheckPrime(input[i]);
 //       }
 //   };

 //   BenchmarkIt(excel, "SERIAL", warmup, rounds, fn);
}

bool PrimeTest::Validate() {
   Logger::INFO << " Not applicable for Offloading";
   // auto CompareBoolArray = [](bool* arr1, bool* arr2, int N) {
   //     bool result = true;
   //     int err_cnt = 0;
   //     for(int i=0; i < N; i++) {
   //         if(arr1[i] != arr2[i]) {
   //             Logger::ERROR << "Boolean value not equal: arr1" << IND(i) << " = " << arr1[i]
   //                         << " and arr2" << IND(i) << " = " << arr2[i];
   //             result = false;
   //             if (err_cnt++ >= 10)
   //                 return result;
   //         }
   //     }
   //     return result;
   // };

   // bool* out_serial = new bool[size];
   // bool* out_parallel_1 = new bool[size];
   // bool* out_parallel_2 = new bool[size];
   //// bool* out_parallel_3 = new bool[size];
   // rounds = 1;
   // warmup = 0;

   // bool* tmp = output;

   // output = out_serial;
   // RunSerial();

   // output = out_parallel_1;
   // RunParallel_1();

   //// output = out_parallel_2;
  ////  RunParallel_2();

   //// output = out_parallel_3;
   //// RunParallel_3();


   // bool is_valid = CompareBoolArray(out_serial, out_parallel_1, size);
   // is_valid &= CompareBoolArray(out_serial, out_parallel_2, size);
   //// is_valid &= CompareBoolArray(out_serial, out_parallel_3, size);

   // output = tmp;
   // delete[] out_serial;
   // delete[] out_parallel_1;
   // delete[] out_parallel_2;
   // delete[] out_parallel_3;

    return true;
}

void PrimeTest::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
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

void PrimeTest::Reinitialize() {
    if(initialized) {
        delete[] input;
        delete[] output;
    }

    input = new int[size];
    output = new bool[size];

    FillRandomArray(input, size, 0, 1000000);
    this->initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<PrimeTest>("PrimeTest");
}

REGISTER_BENCHMARK(PrimeTest, CreateBench);
