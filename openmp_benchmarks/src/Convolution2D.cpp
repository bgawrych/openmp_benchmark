#include "Convolution2D.hpp"
#include <iostream>
#include <omp.h>
#include <cstring>

void Convolution2D::RunParallel() {
    RunParallel_1();
    RunParallel_2();
    RunParallel_3();
}

void Convolution2D::RunParallel_1() {
    auto excel = *this->file;

    const int H = this->H;
    const int W = this->W;
    const int kernel = this->kernel;
    const int kernel_center = kernel/2;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_COLLAPSE_3",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED_2,    // variable name to store execution time
        {

            mpragma(omp parallel for collapse(3))
            for(int batch=0; batch < N; ++batch) {

                for(int y=kernel_center; y < H - kernel_center; ++y) {
                    for(int x=kernel_center; x < W - kernel_center; ++x) {
                        result[batch][y-kernel_center][x-kernel_center] = 0.0f;
                        for(int ky=0; ky < kernel; ++ky) {
                            for(int kx=0; kx < kernel; ++kx) {

                                int input_index_y = y + (kernel_center - ky);
                                int input_index_x = x + (kernel_center - kx);
                                result[batch][y-kernel_center][x-kernel_center] += input_data[batch][input_index_y][input_index_x] * kernel_data[ky][kx];
                            }
                        }
                    }
                }
            }
        }
    )
}


void Convolution2D::RunParallel_2() {
    auto excel = *this->file;

    const int H = this->H;
    const int W = this->W;
    const int kernel = this->kernel;
    const int kernel_center = kernel/2;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_COLLAPSE_2",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED_3,    // variable name to store execution time
        {

            mpragma(omp parallel for collapse(2))
            for(int batch=0; batch < N; ++batch) {

                for(int y=kernel_center; y < H - kernel_center; ++y) {
                    for(int x=kernel_center; x < W - kernel_center; ++x) {
                        result[batch][y-kernel_center][x-kernel_center] = 0.0f;
                        for(int ky=0; ky < kernel; ++ky) {
                            for(int kx=0; kx < kernel; ++kx) {

                                int input_index_y = y + (kernel_center - ky);
                                int input_index_x = x + (kernel_center - kx);
                                result[batch][y-kernel_center][x-kernel_center] += input_data[batch][input_index_y][input_index_x] * kernel_data[ky][kx];
                            }
                        }
                    }
                }
            }
        }
    )
}

void Convolution2D::RunParallel_3() {
    auto excel = *this->file;

    const int H = this->H;
    const int W = this->W;
    const int kernel = this->kernel;
    const int kernel_center = kernel/2;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_NO_COLLAPSE",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED_3,    // variable name to store execution time
        {

            mpragma(omp parallel for)
            for(int batch=0; batch < N; ++batch) {
                for(int y=kernel_center; y < H - kernel_center; ++y) {
                    for(int x=kernel_center; x < W - kernel_center; ++x) {
                        result[batch][y-kernel_center][x-kernel_center] = 0.0f;
                        for(int ky=0; ky < kernel; ++ky) {
                            for(int kx=0; kx < kernel; ++kx) {

                                int input_index_y = y + (kernel_center - ky);
                                int input_index_x = x + (kernel_center - kx);
                                result[batch][y-kernel_center][x-kernel_center] += input_data[batch][input_index_y][input_index_x] * kernel_data[ky][kx];
                            }
                        }
                    }
                }
            }
        }
    )
}

void Convolution2D::RunSerial() {
    auto excel = *this->file;

    const int H = this->H;
    const int W = this->W;
    const int kernel = this->kernel;
    const int kernel_center = kernel/2;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "SERIAL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {

            for(int batch=0; batch < N; ++batch) {

                for(int y=kernel_center; y < H - kernel_center; ++y) {
                    for(int x=kernel_center; x < W - kernel_center; ++x) {
                        result[batch][y-kernel_center][x-kernel_center] = 0.0f;
                        for(int ky=0; ky < kernel; ++ky) {
                            for(int kx=0; kx < kernel; ++kx) {

                                int input_index_y = y + (kernel_center - ky);
                                int input_index_x = x + (kernel_center - kx);
                                result[batch][y-kernel_center][x-kernel_center] += input_data[batch][input_index_y][input_index_x] * kernel_data[ky][kx];
                            }
                        }
                    }
                }
            }
       }
   )
}


bool Convolution2D::Validate() {
    const int kernel_center = kernel / 2;
    const int res_rows = H - 2 * kernel_center;
    const int res_cols = W - 2 * kernel_center;
    Tensor3D<float> out_serial = Create3DArray<float>(N, res_rows, res_cols);
    Tensor3D<float> out_parallel_1 = Create3DArray<float>(N, res_rows, res_cols);
    Tensor3D<float> out_parallel_2 = Create3DArray<float>(N, res_rows, res_cols);
    Tensor3D<float> out_parallel_3 = Create3DArray<float>(N, res_rows, res_cols);
    rounds = 1;
    warmup = 0;

    Swap3DArray(result, out_serial, N, res_rows);
    RunSerial();
    Swap3DArray(result, out_serial, N, res_rows);

    Swap3DArray(result, out_parallel_1, N, res_rows);
    RunParallel_1();
    Swap3DArray(result, out_parallel_1, N, res_rows);

    Swap3DArray(result, out_parallel_2, N, res_rows);
    RunParallel_2();
    Swap3DArray(result, out_parallel_2, N, res_rows);

    Swap3DArray(result, out_parallel_3, N, res_rows);
    RunParallel_3();
    Swap3DArray(result, out_parallel_3, N, res_rows);

    bool is_valid = Compare3DArray(out_serial, out_parallel_1, N, res_rows, res_cols);
    is_valid = is_valid && Compare3DArray(out_serial, out_parallel_2, N, res_rows, res_cols);
    is_valid = is_valid && Compare3DArray(out_serial, out_parallel_3, N, res_rows, res_cols);

    Free3DArray<float>(out_serial);
    Free3DArray<float>(out_parallel_1);
    Free3DArray<float>(out_parallel_2);
    Free3DArray<float>(out_parallel_3);
    return is_valid;
}


void Convolution2D::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;

    rounds = properties["rounds"].GetInt();
    warmup = properties["warmup"].GetInt();

    N = properties["N"].GetInt();
    H = properties["H"].GetInt();
    W = properties["W"].GetInt();
    kernel = properties["kernel"].GetInt();

    Logger::INFO << VAR(N) << VAR(H) << VAR(W) << VAR(kernel);

    std::stringstream os;
    os << VAR_(N) << VAR_(H) << VAR_(W) << VAR_(kernel);
    descriptor = os.str();

    Reinitialize();
}

void Convolution2D::Reinitialize() {
    if(initialized) {
        Free3DArray<float>(input_data);
        Free2DArray<float>(kernel_data);
        Free3DArray<float>(result);
    }

    int kernel_center = kernel / 2;
    int res_rows = H - 2 * kernel_center;
    int res_cols = W - 2 * kernel_center;

    input_data = Create3DArray<float>(N, H, W);
    kernel_data = Create2DArray<float>(kernel, kernel);
    result  = Create3DArray<float>(N, res_rows, res_cols);

    FillRandom3DArray(input_data, N, H, W);
    FillRandom2DArray(kernel_data, kernel, kernel);
    float* raw_result_ptr = result[0][0];
    std::memset(raw_result_ptr, 0, N * res_rows * res_cols * sizeof(float));

    this->initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<Convolution2D>("Convolution2D");
}

REGISTER_BENCHMARK(Convolution2D, CreateBench);
