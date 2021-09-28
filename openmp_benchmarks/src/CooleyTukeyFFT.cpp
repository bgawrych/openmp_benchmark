#include "CooleyTukeyFFT.hpp"
#include <iostream>
#include <omp.h>

#include <math.h>

//TODO add final clause benchmark

#define PI 3.14159265358979323846
//using namespace std::literals::complex_literals;

void CTFFT(double* x, std::complex<double>* out, int N, int s){
    unsigned int k;
    std::complex<double> t;
    if (N == 1) {
        out[0] = x[0];
        return;
    }

    CTFFT(x, out, N/2, 2*s);
    CTFFT(x+s, out + N/2, N/2, 2*s);

    for (k = 0; k < N/2; k++) {
        t = out[k];
        out[k] = t + exp(std::complex<double>(0,-2 * PI * k / N)) * out[k + N/2];
        out[k + N/2] = t - exp(std::complex<double>(0,-2 * PI  * k / N)) * out[k + N/2];
    }
}


void CTFFT_Parallel(double* x, std::complex<double>* out, int N, int s){
    if (N == 1) {
        out[0] = x[0];
        return;
    }
    
    mpragma(omp task default(none) shared(x, out, N, s)) {
        CTFFT_Parallel(x, out, N/2, 2*s);
    }

    CTFFT_Parallel  (x+s, out + N/2, N/2, 2*s);

    mpragma(omp taskwait)

    std::complex<double> t;
    for (unsigned int k = 0; k < N/2; k++) {
        t = out[k];
        out[k] = t + exp(std::complex<double>(0,-2 * PI * k / N)) * out[k + N/2];
        out[k + N/2] = t - exp(std::complex<double>(0,-2 * PI  * k / N)) * out[k + N/2];
    }
}




void CTFFT_Parallel_final(double* x, std::complex<double>* out, int N, int s, int nodes){
    if (N == 1) {
        out[0] = x[0];
        return;
    }

    // Cooley-Tukey: recursively split in two, then combine beneath.
    int nthreads = omp_get_num_threads();

    mpragma(omp task default(none) firstprivate(x, out, N, s, nodes) final(nodes >= nthreads)) {
        CTFFT_Parallel_final(x, out, N/2, 2*s, 2*nodes);
    }
    
    mpragma(omp task default(none) firstprivate(x, out, N, s, nodes) final(nodes >= nthreads)) {
        CTFFT_Parallel_final(x+s, out + N/2, N/2, 2*s, 2*nodes + 1);
    }
    mpragma(omp taskwait)

    std::complex<double> t;
    for (unsigned int k = 0; k < N/2; k++) {
        t = out[k];
        out[k] = t + exp(std::complex<double>(0,-2 * PI * k / N)) * out[k + N/2];
        out[k + N/2] = t - exp(std::complex<double>(0,-2 * PI  * k / N)) * out[k + N/2];
    }
}


void CTFFT_Parallel_single(double* x, std::complex<double>* out, int N, int s, int nodes){
    if (N == 1) {
        out[0] = x[0];
        return;
    }

    int nthreads = omp_get_num_threads();


    mpragma(omp task default(none) firstprivate(x, out, N, s, nodes) final(nodes >= nthreads)) {
        CTFFT_Parallel_single(x, out, N/2, 2*s, 2*nodes);
    }
    CTFFT_Parallel_single(x+s, out + N/2, N/2, 2*s, 2*nodes + 1);
    
    mpragma(omp taskwait)

    std::complex<double> t;
    for (unsigned int k = 0; k < N/2; k++) {
        t = out[k];
        out[k] = t + exp(std::complex<double>(0,-2 * PI * k / N)) * out[k + N/2];
        out[k + N/2] = t - exp(std::complex<double>(0,-2 * PI  * k / N)) * out[k + N/2];
    }
}

void CooleyTukeyFFT::RunParallel() {
    //RunParallel_Bad();
    RunParallel_Final();
    RunParallel_Single();
}

void CooleyTukeyFFT::RunParallel_Bad() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_BAD",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel shared(input, output, size)) {
                mpragma(omp single) {
                     CTFFT_Parallel(input, output, size, 1);
                }
            }
        }
   )
}

void CooleyTukeyFFT::RunParallel_Final() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_FINAL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel shared(input, output, size)) {
                mpragma(omp single) {
                     CTFFT_Parallel_final(input, output, size, 1, 1);
                }
            }
        }
   )
}


void CooleyTukeyFFT::RunParallel_Single() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "PARALLEL_SINGLE_FINAL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            mpragma(omp parallel shared(input, output, size)) {
                mpragma(omp single) {
                     CTFFT_Parallel_single(input, output, size, 1, 1);
                }
            }
        }
   )
}

void CooleyTukeyFFT::RunSerial() {
    auto excel = *this->file;

    BENCHMARK_STRUCTURE(
        excel,      // name of csv logger
        "SERIAL",       // name of benchmark
        warmup,     // name of warmup rounds variable
        rounds,     // name of benchmark rounds variable
        ELAPSED,    // variable name to store execution time
        {
            CTFFT(input, output, size, 1);
        }
   )
}


bool CooleyTukeyFFT::Validate() {
    
    auto ComplexArrayCmp = [](std::complex<double>* arr1, std::complex<double>* arr2, int N) {
        bool result = true;
        for(int i=0; i < N; i++) {
            if(std::abs( (arr1[i] - arr2[i])/arr1[i]) > 0.01
               && std::abs(arr1[i] - arr2[i]) > 0.004) {
                Logger::ERROR << "Error exceeds epsilon: arr1" << IND(i) << " = " << arr1[i]
                            << " and arr2" << IND(i) << " = " << arr2[i];
                result = false;
            }
        }
        return result;
    };

    std::complex<double>* out_serial = new std::complex<double>[size];
    std::complex<double>* out_parallel_1 = new std::complex<double>[size];
    std::complex<double>* out_parallel_2 = new std::complex<double>[size];
    std::complex<double>* out_parallel_3 = new std::complex<double>[size];

    rounds = 1;
    warmup = 0;

    std::complex<double>* tmp = output;

    output = out_serial;
    RunSerial();

    output = out_parallel_1;
    RunParallel_Single();

    output = out_parallel_2;
    RunParallel_Final();

    output = out_parallel_3;
    RunParallel_Bad();

    bool is_valid = ComplexArrayCmp(out_serial, out_parallel_1, size);
    is_valid &= ComplexArrayCmp(out_serial, out_parallel_2, size);
    is_valid &= ComplexArrayCmp(out_serial, out_parallel_3, size);

    output = tmp;
    delete[] out_serial;
    delete[] out_parallel_1;
    delete[] out_parallel_2;
    delete[] out_parallel_3;

    return is_valid;
}

void CooleyTukeyFFT::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;
    
    rounds = properties["rounds"].GetInt();
    warmup = properties["warmup"].GetInt();
    size = properties["size"].GetInt();//128;

    Logger::INFO << VAR(size);

    std::stringstream os;
    os << VAR_(size);
    descriptor = os.str();

    Reinitialize();
}


void CooleyTukeyFFT::Reinitialize() {
    if(initialized) {
        delete[] input;
        delete[] output;
    }

    input = new double[size];
    for(int i=0;i < size;i++) {
        input[i] = 2*sin(i) + cos(i*2);
    }
    output = new std::complex<double>[size];

    this->initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<CooleyTukeyFFT>("CooleyTukeyFFT");
}

REGISTER_BENCHMARK(CooleyTukeyFFT, CreateBench);
