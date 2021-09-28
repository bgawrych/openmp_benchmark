#include "BatchNorm.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>
#include <functional>
#include <cmath>

#define EPS 0.00000001f

void BatchNorm::RunParallel() {
    auto excel = *this->file;

    float* norm_divider = new float[C];

    auto fn = [&]() {
        int _N = N; int _C = C; int _H = H; int _W = W;
        int size = N*C*H*W;
        float* raw_input = input_data[0][0][0];
        float* raw_output = output[0][0][0];
        float* _beta = beta;
        float* _gamma = gamma;
        float* _variance = variance;
        float* _norm_divider  = norm_divider;
        float* _mean  = mean;

        #pragma omp target enter data  map(to:raw_input[0:size], _beta[0:_C], _gamma[0:_C]) \
                                        map(to:raw_output[0:size]) \
                                        map(to: _variance[0:_C], _mean[0:_C], _norm_divider[0:_C])
        {

            #pragma omp target teams distribute parallel for
            for(int i=0; i < _C; ++i) {
                _variance[i] = 0;
                _norm_divider[i] = 0;
                _mean[i] = 0;
            }

            //# mini-batch mean
        #if defined(__clang__) 
            //clang error: cannot generate code for reduction on array section, which requires a variable length array
            //              #pragma omp  target teams distribute parallel for reduction(+:_variance[:_C]) schedule(static) collapse(4)
            #pragma omp  target teams distribute parallel for schedule(static)
        #else
            #pragma omp  target teams distribute parallel for reduction(+:_mean[:_C]) schedule(static) collapse(4)
        #endif
            for(int i=0; i < _C; ++i) {
                for(int j=0; j < _N; ++j) {
                    for(int k=0; k < _H; ++k) {
                        for(int p=0; p < _W; ++p) {
                            _mean[i] += raw_input[i* _H * _W + j * _C * _H * _W + k* _W + p] / (1.0f*(_N * _H * _W));
                        }
                    }
                }
            }
        #if defined(__clang__) 
            #pragma omp  target teams distribute parallel for schedule(static)
        #else
            #pragma omp  target teams distribute parallel for reduction(+:_variance[:_C]) schedule(static) collapse(4)
        #endif
            for(int i=0; i < _C; ++i) {
                for(int j=0; j < _N; ++j) {
                    for(int k=0; k < _H; ++k) {
                        for(int p=0; p < _W; ++p) {
                            auto in = raw_input[i * _H * _W + j * _C * _H * _W + k * _W + p];
                            _variance[i] += (in - _mean[i]) * (in - _mean[i]) / (1.0f*(_N * _H * _W));
                        }
                    }
                }
            }

            #pragma omp  target teams distribute parallel for schedule(static) collapse(4)
            for(int i=0; i < _C; ++i) {
                for(int j=0; j < _N; ++j) {
                    for(int k=0; k < _H; ++k) {
                        for(int p=0; p < _W; ++p) {
                            raw_output[i*_H *_W + j*_C*_H*_W + k*_W + p] = _gamma[i] * ((raw_input[i*_H*_W + j*_C*_H*_W + k*_W + p] - _mean[i]) / (sqrt(_variance[i] + EPS))) + _beta[i];
                        }
                    }
                }
            }
        }
        #if defined(__clang__) 
            #pragma omp target exit data map(from:raw_output[0:size]) \
                                         map(from: _variance[0:_C], _mean[0:_C], _norm_divider[0:_C])

        #else
            #pragma omp target update from(raw_output[0:size], _variance[0:_C], _mean[0:_C], _norm_divider[0:_C])
            #pragma omp target exit data map(delete:raw_output, raw_input, _beta, _gamma, _variance, _mean, _norm_divider)
	    #endif
    };

    BenchmarkIt(excel, "RunParallel_opt", warmup, rounds, fn);

   delete[] norm_divider;
}




void BatchNorm::RunSerial() {
    auto excel = *this->file;

    float* norm_divider = new float[C];
     auto fn = [&]() {
        float* input_raw = input_data[0][0][0];
        //# mini-batch mean
        for(int i=0; i < C; ++i) {
            variance[i] = 0;
            norm_divider[i] = 0;
            mean[i] = 0;
        }

        for(int i=0; i < C; ++i) {
            for(int j=0; j < N; ++j) {
                for(int k=0; k < H; ++k) {
                    for(int p=0; p < W; ++p) {
                        mean[i] += input_raw[i*H*W + j*C*H*W + k*W + p];
                    }
                }
            }
        }

        for(int i=0; i < C; ++i) {
            mean[i] = mean[i] / (1.0f*(N*H*W));
        }
        //# mini-batch variance
        //variance = np.mean((arr - mean.reshape((1, 3, 1, 1))) ** 2, axis=(0, 2, 3))
        for(int i=0; i < C; ++i) {
            for(int j=0; j < N; ++j) {
                for(int k=0; k < H; ++k) {
                    for(int p=0; p < W; ++p) {
                        auto in = input_raw[i*H*W + j*C*H*W + k*W + p];
                        variance[i] += (in - mean[i]) * (in - mean[i]);
                    }
                }
            }
        }

        for(int i=0; i < C; ++i) {
            variance[i] = variance[i] / (1.0f*(N*H*W));
            norm_divider[i] = sqrt(variance[i] + EPS);
        }

        for(int i=0; i < C; ++i) {
            for(int j=0; j < N; ++j) {
                for(int k=0; k < H; ++k) {
                    for(int p=0; p < W; ++p) {
                        output[j][i][k][p] = gamma[i] * ((input_data[j][i][k][p] - mean[i]) / norm_divider[i]) + beta[i];
                    }
                }
            }
        }
    };

    BenchmarkIt(excel, "Serial", warmup, rounds, fn);

   delete[] norm_divider;
}

bool BatchNorm::Validate() {
    Tensor4D<float> out_serial = Create4DArray<float>(N, C, H, W);
    Tensor4D<float> out_parallel = Create4DArray<float>(N, C, H, W);
    rounds = 1;
    warmup = 0;


    Swap4DArray(output, out_serial, N, C, H);
    RunSerial();
    Swap4DArray(output, out_serial, N, C, H);

    Swap4DArray(output, out_parallel, N, C, H);
    RunParallel();
    Swap4DArray(output, out_parallel, N, C, H);

    bool is_valid = Compare4DArray(out_serial, out_parallel, N, C, H, W);

    Free4DArray<float>(out_serial);
    Free4DArray<float>(out_parallel);
    return is_valid;
}

void BatchNorm::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;
    
    rounds = properties["rounds"].GetInt();
    warmup = properties["warmup"].GetInt();

    N = properties["N"].GetInt();
    C = properties["C"].GetInt();
    H = properties["H"].GetInt();
    W = properties["W"].GetInt();
    Logger::INFO << VAR(N) << VAR(C) << VAR(H) << VAR(W);

    Reinitialize();
}

void BatchNorm::Reinitialize() {
    if(initialized) {
        Free4DArray(input_data);
        Free4DArray(output);
        delete[] mean;
        delete[] variance;
        delete[] gamma;
        delete[] beta;
    }

    input_data = Create4DArray<float>(N, C, H, W);
    FillRandom4DArray(input_data, N, C, H, W);
    gamma = new float[C];
    beta = new float[C];
    output = Create4DArray<float>(N, C, H, W);
    mean = new float[C];
    variance = new float[C];


    FillRandomArray(gamma, C);
    FillRandomArray(beta, C);
    initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<BatchNorm>("BatchNorm");
}

REGISTER_BENCHMARK(BatchNorm, CreateBench);
