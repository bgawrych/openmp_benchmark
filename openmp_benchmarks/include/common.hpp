#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdlib.h>
#include <sstream>
#include <unordered_map>
#include "Benchmark.hpp"
#include "Logger.hpp"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <rapidjson/istreamwrapper.h>
#include <omp.h>
#include <random>

#define PUT_BENCHMARK(NAME) {#NAME, std::make_shared<NAME>(#NAME)}
#define VAR(X) #X":" << X << " "
#define VAR_(X) #X":" << X << "_"
#define IND(x) "[" << x << "]"

typedef std::unordered_map<std::string, BenchmarkPtr> BenchmarkMap;
typedef unsigned int uint;
typedef rapidjson::Document Json;
typedef std::shared_ptr<rapidjson::Document> JsonPtr;

template <class T>
using Tensor4D = T****;

template <class T>
using Tensor3D = T***;

template <class T>
using Tensor2D = T**;


inline int CalcAlignedSize(int size, int alignment=128) {
    if (size % alignment == 0)
        return size;
    
    int aligned_size = size + (alignment - (size % alignment));
    return aligned_size;
}

template<class T>
T** Create2DArray(int H, int W) {
    T** array2D = new T*[H];
    int size = CalcAlignedSize(H*W*sizeof(T));
    T* rawData = (T*) aligned_alloc(128, size);

    for(int i=0; i < H; i++) {
        array2D[i] = rawData;
        rawData += W;
    }

    return array2D;
}

template<class T>
T*** Create3DArray(int C, int H, int W) {
    T*** array3D = new T**[C]; 
    T** array2D = new T*[C*H];
    int size = CalcAlignedSize(C*H*W*sizeof(T));
    T* rawData = (T*) aligned_alloc(128, size);

    for(int i=0; i < C; i++) {
        array3D[i] = array2D;
        for(int j=0; j < H; j++) {
            array2D[j] = rawData;   
            rawData += W;
        }
        array2D += H;
    }

    return array3D;
}

template<class T>
T**** Create4DArray(int N, int C, int H, int W) {
    T**** array4D = new T***[N];
    T*** array3D = new T**[N*C]; 
    T** array2D = new T*[N*C*H];
    int size = CalcAlignedSize(N*C*H*W*sizeof(T));
    T* rawData = (T*) aligned_alloc(128, size);


    for(int i=0; i < N; ++i) {
        array4D[i] = array3D;
        for(int j=0; j < C; ++j) {
            array3D[j] = array2D;
            for(int z=0; z < H; ++z) {
                array2D[z] = rawData;   
                rawData += W;
            }
            array2D += H;
        }
        array3D += C;
    }

    return array4D;
}

template <class T> 
void Free2DArray(T** ptr) {
    delete[] ptr[0];
    delete[] ptr;
};

template <class T> 
void Free3DArray(T*** ptr) {
    delete[] ptr[0][0]; 
    delete[] ptr[0];
    delete[] ptr;
};

template <class T> 
void Free4DArray(T**** ptr) {
    delete[] ptr[0][0][0]; 
    delete[] ptr[0][0]; 
    delete[] ptr[0];
    delete[] ptr;
};


inline void FillRandomArray(int* arr, int N, int min=-1, int max=1) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<int> distribution(min, max);
    for(int i=0; i < N; i++) {
        arr[i] = distribution(generator);
    }
}

inline void FillRandomArray(float* arr, int N, float min=-1, float max=1) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<int> distribution(0, RAND_MAX);
    for(int i=0; i < N; i++) {
        arr[i] = (static_cast<float>(distribution(generator)) / (float) RAND_MAX) * (max - min) + min;
    }
}

inline void FillRandom2DArray(float** arr, int N, int M, float min=-1, float max=1) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<int> distribution(0, RAND_MAX);
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++) {
            arr[i][j] = (static_cast<float>(distribution(generator)) / (float) RAND_MAX) * (max - min) + min;
        }
    }
}

inline void FillRandom3DArray(float*** arr, int N, int M, int K, float min=-1, float max=1) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<int> distribution(0, RAND_MAX);
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++) {
            for(int l=0; l < K; l++) {
                arr[i][j][l] = (static_cast<float>(distribution(generator)) / (float) RAND_MAX) * (max - min) + min;
            }
        }
    }
}

inline void FillRandom4DArray(float**** arr, int N, int M, int K, int O, float min=-1, float max=1) {
    std::mt19937 generator(0);
    std::uniform_int_distribution<int> distribution(0, RAND_MAX);
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++) {
            for(int l=0; l < K; l++) {
                 for(int z=0; z < O; z++) {
                    arr[i][j][l][z] = (static_cast<float>(distribution(generator)) / (float) RAND_MAX) * (max - min) + min;
                 }
            }
        }
    }
}


template<class T>
inline bool CompareArray(T* arr1, T* arr2, int N, float rtol=0.01, float atol=0.001) {
    bool result = true;
    int err_cnt = 0;
    for(int i=0; i < N; i++) {
        if(std::abs( (arr1[i] - arr2[i])/arr1[i]) > rtol
           && std::abs(arr1[i] - arr2[i]) > atol) {
            Logger::ERROR << "Error exceeds epsilon: arr1" << IND(i) << " = " << arr1[i]
                          << " and arr2" << IND(i) << " = " << arr2[i];
            result = false;
            if (err_cnt++ >= 10)
                return result;
        }
    }
    return result;
}

template<class T>
inline bool Compare2DArray(Tensor2D<T> arr1, Tensor2D<T> arr2, int N, int M, float rtol=0.01, float atol=0.001) {
    bool result = true;
    int err_cnt = 0;
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++) {
            if(std::abs( (arr1[i][j] - arr2[i][j])/arr1[i][j]) > rtol
               && std::abs(arr1[i][j] - arr2[i][j]) > atol) {
                Logger::ERROR << "Error exceeds epsilon: arr1" << IND(i) << IND(j) << " = " << arr1[i][j]
                              << " and arr2" << IND(i) << IND(j) << " = " << arr2[i][j];
                result = false;
                if (err_cnt++ >= 10)
                    return result;
            }
        }
    }
    return result;
}

template<class T>
inline bool Compare3DArray(Tensor3D<T> arr1, Tensor3D<T> arr2, int N, int M, int K, float rtol=0.01, float atol=0.001) {
    bool result = true;
    int err_cnt = 0;
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++) {
            for(int l=0; l < K; l++) {
               if(std::abs( (arr1[i][j][l] - arr2[i][j][l])/arr1[i][j][l]) > rtol
                 && std::abs(arr1[i][j][l] - arr2[i][j][l]) > atol) {
                    Logger::ERROR << "Error exceeds epsilon: arr1" << IND(i) << IND(j) << IND(l) << " = " << arr1[i][j][l]
                                    << " and arr2" << IND(i) << IND(j) << IND(l) << " = " << arr2[i][j][l];
                    result = false;
                    if (err_cnt++ >= 10)
                        return result;
                }
            }
        }
    }
    return result;
}

template<class T>
inline bool Compare4DArray(Tensor4D<T> arr1, Tensor4D<T> arr2, int N, int M, int K, int O, float rtol=0.01, float atol=0.0004) {
    bool result = true;
    int err_cnt = 0;
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++) {
            for(int l=0; l < K; l++) {
                 for(int z=0; z < O; z++) {
                    if(std::abs( (arr1[i][j][l][z] - arr2[i][j][l][z])/arr1[i][j][l][z]) > rtol
                       && std::abs(arr1[i][j][l][z] - arr2[i][j][l][z]) > atol) {
                        Logger::ERROR << "Error exceeds epsilon: arr1" << IND(i) << IND(j) << IND(l) << IND(z) << " = " << arr1[i][j][l][z]
                                      << " and arr2" << IND(i) << IND(j) << IND(l) << IND(z) << " = " << arr2[i][j][l][z];
                        result = false;
                        if (err_cnt++ >= 10)
                            return result;
                    }
                 }
            }
        }
    }
    return result;
}

template<class T>
inline void Swap2DArray(Tensor2D<T> a, Tensor2D<T> b, int M) {
    for(int i=0; i < M; i++) {
        auto tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
    auto tmp = a;
    a = b;
    b = tmp;
};

template<class T>
inline void Swap3DArray(Tensor3D<T> a, Tensor3D<T> b, int M, int N) {
    for(int i=0; i < M; i++) {
        for(int j=0; j < N; j++) {
            auto tmp = a[i][j];
            a[i][j] = b[i][j];
            b[i][j] = tmp;
        }
        auto tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
    auto tmp = a;
    a = b;
    b = tmp;
};

template<class T>
inline void Swap4DArray(Tensor4D<T> a, Tensor4D<T> b, int M, int N, int K) {
    for(int i=0; i < M; i++) {
        for(int j=0; j < N; j++) {
            for(int z=0; z < K; z++) {
                auto tmp = a[i][j][z];
                a[i][j][z] = b[i][j][z];
                b[i][j][z] = tmp;
            }
            auto tmp = a[i][j];
            a[i][j] = b[i][j];
            b[i][j] = tmp;
        }
        auto tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
    auto tmp = a;
    a = b;
    b = tmp;
};

inline void PrintArray(float* arr, int N) {
    for(int i=0; i < N; i++) {
        std::cout << (float)arr[i] << " ";
    }
}

inline void Print2DArray(float** arr, int N, int M) {
    for(int i=0; i < N; i++) {
        for(int j=0; j < M; j++){
            std::cout << (float)arr[i][j] << " ";
        }
        std::cout << "\n";
    }
}

inline double VecMean(const std::vector<double>& vec) {
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

inline double VecStdDev(const std::vector<double>& vec) {
    double mean = VecMean(vec);
    double square_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    double stdev = std::sqrt(square_sum / vec.size() - mean * mean);
    return stdev;
}

#define LOOP_UNOPTIMIZER(var) __asm__ volatile("" : "+g" (var) : :);

#define mpragma(...)  _Pragma(#__VA_ARGS__)

#define BENCHMARK_STRUCTURE(_Excel, _Mode, _Warmup, _Rounds, _Elapsed, ...)                       \
        std::vector<double> durations(_Rounds);                                                   \
        int unoptimizer = 0;                                                                      \
        for(int warmup_i=0; warmup_i < _Warmup; warmup_i++){                                      \
            {__VA_ARGS__}                                                                         \
            LOOP_UNOPTIMIZER(unoptimizer)                                                         \
        }                                                                                         \
        auto start = omp_get_wtime();                                                             \
        for(int round_i=0; round_i < _Rounds; round_i++){                                         \
            auto start_iter = omp_get_wtime();                                                    \
            {__VA_ARGS__}                                                                         \
            LOOP_UNOPTIMIZER(unoptimizer)                                                         \
            auto end_iter = omp_get_wtime();                                                      \
            durations[round_i] = end_iter - start_iter;                                           \
        }                                                                                         \
        auto end = omp_get_wtime();                                                               \
        double _Elapsed = end - start;                                                            \
        double Mean = VecMean(durations);                                                         \
        double StdDev = VecStdDev(durations);                                                     \
        _Excel << this->name << this->descriptor << _Mode                                         \
               << _Warmup << _Rounds << _Elapsed << Mean << StdDev;                               \
        Logger::INFO << this->descriptor << " " << _Mode << " Warmup:" << _Warmup                 \
                     << " Rounds: " << _Rounds << " Time: " << _Elapsed                           \
                     << " Mean: " << Mean << " StdDev: " << StdDev;


template<typename Func>
inline void BenchmarkIt(Logger::LoggerClass& file, std::string mode, int warmup, int rounds, Func func) {
        std::vector<double> durations(rounds);
        int unoptimizer = 0;
        for(int warmup_i=0; warmup_i < warmup; warmup_i++){
            func();
            LOOP_UNOPTIMIZER(unoptimizer)
        }
        double start = omp_get_wtime();
        for(int round_i=0; round_i < rounds; round_i++){
            double start_iter = omp_get_wtime();
            func();
            LOOP_UNOPTIMIZER(unoptimizer)
            double end_iter = omp_get_wtime();
            durations[round_i] = end_iter - start_iter;
        }
        double end = omp_get_wtime();
        double _Elapsed = end - start;
        double Mean = VecMean(durations);
        double StdDev = VecStdDev(durations);
        file  << mode << warmup << rounds << _Elapsed << Mean << StdDev;
        Logger::INFO << mode << " Warmup:" << warmup
                     << " Rounds: " << rounds << " Time: " << _Elapsed
                     << " Mean: " << Mean << " StdDev: " << StdDev;
}
#endif