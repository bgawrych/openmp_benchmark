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
    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            float* input_raw = input_data[0][0][0];
            //# mini-batch mean
            mpragma(omp parallel)
            {
                mpragma(omp for)
                for(int i=0; i < C; ++i) {
                    variance[i] = 0;
                    norm_divider[i] = 0;
                    mean[i] = 0;
                }

                mpragma(omp for reduction(+:mean[:C]) collapse(2) schedule(static))
                for(int i=0; i < C; ++i) {
                    for(int j=0; j < N; ++j) {
                        for(int k=0; k < H; ++k) {
                            for(int p=0; p < W; ++p) {
                                mean[i] += input_raw[i*H*W + j*C*H*W + k*W + p];
                            }
                        }
                    }
                }

                mpragma(omp for)
                for(int i=0; i < C; ++i) {
                    mean[i] = mean[i] / (1.0f*(N*H*W));
                }

                mpragma(omp for reduction(+:variance[:C]) collapse(2) schedule(static))
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

                mpragma(omp for)
                for(int i=0; i < C; ++i) {
                    variance[i] = variance[i] / (1.0f*(N*H*W));
                    norm_divider[i] = sqrt(variance[i] + EPS);
                }

                mpragma(omp for schedule(static) collapse(2))
                for(int i=0; i < C; ++i) {
                    for(int j=0; j < N; ++j) {
                        for(int k=0; k < H; ++k) {
                            for(int p=0; p < W; ++p) {
                                output[j][i][k][p] = gamma[i] * ((input_data[j][i][k][p] - mean[i]) / norm_divider[i]) + beta[i];
                            }
                        }
                    }
                }
            }
        }
    )
   delete[] norm_divider;
}




void BatchNorm::RunSerial() {
    auto excel = *this->file;

    float* norm_divider = new float[C];
    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "SERIAL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            float* input_raw = input_data[0][0][0];
            //# mini-batch mean
            for(int i=0; i < C; ++i) {
                variance[i] = 0;
                norm_divider[i] = 0;
                mean[i] = 0;
            }

            for(int j=0; j < N; ++j) {
                for(int i=0; i < C; ++i) {
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
            for(int j=0; j < N; ++j) {
                for(int i=0; i < C; ++i) {
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

            for(int j=0; j < N; ++j) {
                for(int i=0; i < C; ++i) {
                    for(int k=0; k < H; ++k) {
                        for(int p=0; p < W; ++p) {
                            output[j][i][k][p] = gamma[i] * ((input_data[j][i][k][p] - mean[i]) / norm_divider[i]) + beta[i];
                        }
                    }
                }
            }
        }
   )
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

    std::stringstream os;
    os << VAR_(N) << VAR_(C) << VAR_(H) << VAR_(W);
    descriptor = os.str();

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