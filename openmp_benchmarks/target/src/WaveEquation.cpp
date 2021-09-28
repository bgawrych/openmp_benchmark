#include "WaveEquation.hpp"
#include <iostream>
#include <omp.h>
#include <tuple>
#include <math.h>
#include <cstring>

void WaveEquation::RunParallel() {
    RunParallel_1();
    //RunParallel_2();
}
void WaveEquation::RunParallel_1() {

    auto excel = *this->file;


    auto fn = [&]() {
        int _M = M; int _N = N; int _K = K;
        int size = 3 * M * N;
        int _px = px; int _py = py;
        double* raw_src = waves[0][0];
        double* src_2;// = waves[k % 3][0]; //
        double* src_1;// = waves[(k + 1) % 3][0];
        double* dst;//   = waves[(k + 2) % 3][0];
        #pragma omp target enter data map(to:raw_src[0:size])
        {
            for(int t=0; t < _K-1; t++) {
                src_2 = raw_src + (t % 3) * _M * _N; //
                src_1 = raw_src +((t + 1) % 3) * _M * _N;
                dst  = raw_src + ((t + 2) % 3) * _M * _N;
                #pragma omp target teams distribute parallel for schedule(static) collapse(2)
                for(int i=1; i < _M-1; i++) {
                    for(int j=1; j < _N-1; j++) {
                        // z tlumieniem
                        // dst[i][j] = 2.0 / q * (1 - px - py)* src[k][i][j]      // 2/q*(1-px-py)*f(2:M-1,2:N-1,k)
                        //                 + px * (src[k][i+1][j] + src[k][i-1][j])/q  // px*(f(3:M,2:N-1,k)+f(1:M-2,2:N-1,k))/q
                        //                 + py * (src[k][i][j+1] + src[k][i][j-1])/q  // py*( f(2:M-1,3:N,k)+f(2:M-1,1:N-2,k))/q
                        //                 - w*wave[k-1][i][j]/q;                        // w*f(2:M-1,2:N-1,k-1)/q
                        // bez tlumienia
                        dst[i*(_N) + j] = 2.0 * (1 - _px - _py) * src_1[i*(_N) +j]             // 2/q*(1-px-py)*f(2:M-1,2:N-1,k)
                                            + _px * (src_1[(i+1)*(_N) + j] + src_1[(i-1)*(_N) + j])  // px*(f(3:M,2:N-1,k)+f(1:M-2,2:N-1,k))
                                            + _py * (src_1[i*(_N) + j+1] + src_1[i*(_N) + j-1])  // py*( f(2:M-1,3:N,k)+f(2:M-1,1:N-2,k)
                                            - src_2[i*(_N) +j];                        // f(2:M-1,2:N-1,k-1)
                    }
                }
            }
        }
        #if defined(__clang__) 
            #pragma omp target exit data map(from:raw_src[0:size])
        #else
            #pragma omp target update from(raw_src[0:size])
            #pragma omp target exit data map(delete:raw_src)
	    #endif

    };

    BenchmarkIt(excel, "Offload", warmup, rounds, fn);


}


void WaveEquation::RunParallel_2() {

// Brak zastosowania dla offloadingu

}

void WaveEquation::RunSerial() {

    auto excel = *this->file;

    auto fn = [&]() {
        Tensor2D<double> src_2;
        Tensor2D<double> src_1;
        Tensor2D<double> dst;

        for(int t=0; t < K-1; t++) {
                src_2 = waves[t % 3]; //
                src_1 = waves[(t + 1) % 3];
                dst  = waves[(t + 2) % 3];

                for(int i=1; i < M-1; i++) {
                    for(int j=1; j < N-1; j++) {
                        // z tlumieniem
                        // dst[i][j] = 2.0 / q * (1 - px - py)* src[k][i][j]      // 2/q*(1-px-py)*f(2:M-1,2:N-1,k)
                        //                 + px * (src[k][i+1][j] + src[k][i-1][j])/q  // px*(f(3:M,2:N-1,k)+f(1:M-2,2:N-1,k))/q
                        //                 + py * (src[k][i][j+1] + src[k][i][j-1])/q  // py*( f(2:M-1,3:N,k)+f(2:M-1,1:N-2,k))/q
                        //                 - w*wave[k-1][i][j]/q;                        // w*f(2:M-1,2:N-1,k-1)/q
                        // bez tlumienia
                        dst[i][j] = 2.0 * (1 - px - py)* src_1[i][j]             // 2/q*(1-px-py)*f(2:M-1,2:N-1,k)
                                        + px * (src_1[i+1][j] + src_1[i-1][j])  // px*(f(3:M,2:N-1,k)+f(1:M-2,2:N-1,k))
                                        + py * (src_1[i][j+1] + src_1[i][j-1])  // py*( f(2:M-1,3:N,k)+f(2:M-1,1:N-2,k)
                                        - src_2[i][j];                        // f(2:M-1,2:N-1,k-1)
                    }
                }
            }

    };

    BenchmarkIt(excel, "Serial", warmup, rounds, fn);
}


bool WaveEquation::Validate() {
    Tensor3D<double> out_serial = Create3DArray<double>(3, M, N);
    Tensor3D<double> out_parallel_1 = Create3DArray<double>(3, M, N);

    //copy input to out tensors as this operation is inplace
    memcpy(**out_serial, **waves, 3*M*N*sizeof(double));
    memcpy(**out_parallel_1, **waves, 3*M*N*sizeof(double));
    
    rounds = 1;
    warmup = 0;


    Swap3DArray(waves, out_serial, 3, M);
    RunSerial();
    Swap3DArray(waves, out_serial, 3, M);

    Swap3DArray(waves, out_parallel_1, 3, M);
    RunParallel_1();
    Swap3DArray(waves, out_parallel_1, 3, N);



    bool is_valid = Compare3DArray(out_serial, out_parallel_1, 3, M, N);

    Free3DArray<double>(out_serial);
    Free3DArray<double>(out_parallel_1);

    return is_valid;
}

void WaveEquation::Init(Logger::LoggerClass* file, const rapidjson::Value& properties) {
    this->file = file;

    rounds = properties["rounds"].GetInt();//128;
    warmup = properties["warmup"].GetInt();//128;

    v = 100;
    a = 1.2;
    b = 0.8;

    M = properties["M"].GetInt();
    N = properties["N"].GetInt();
    K = properties["K"].GetInt();
    Logger::INFO << VAR(M) << VAR(N) << VAR(K);

    Reinitialize();
}


void WaveEquation::Reinitialize() {
    if(initialized) {
        Free3DArray<double>(waves);
    }


    double* x = new double[M]; //linspace(0,a,M); // utworz M elementów od 0 do a z równym odstępem
    double* y = new double[N]; //linspace(0,b,N); // utworz N elementów od 0 do b z równym odstępem

    auto linspace = [=](double* arr, double l, double r, int size) {
        const double delta = (r - l) / (size - 1);
        for(int i=0; i < size; ++i) {
            arr[i] = l + delta * i;
        }
    };

    linspace(x, 0, a, M);
    linspace(y, 0, b, N);



    dx = x[2]-x[1];
    dy = y[2]-y[1];
    dt = dx*dy / v / (dx+dy);
    px = (v*dt/dx) * (v*dt/dx);
    py = (v*dt/dy) * (v*dt/dy);

    beta=20;
    q=1+beta*dt;
    w=1-beta*dt;

    auto meshgrid = [=](double* arr_1, int size_arr_1, double* arr_2, int size_arr_2) {

        //Tensor2D<double> ret_1 = Create2DArray<double>(size_arr_2, size_arr_1);
        //Tensor2D<double> ret_2 = Create2DArray<double>(size_arr_2, size_arr_1);
        Tensor2D<double> ret_1 = Create2DArray<double>(size_arr_1, size_arr_2); // transpose /\'
        Tensor2D<double> ret_2 = Create2DArray<double>(size_arr_1, size_arr_2); //transpose /\'
        for(int i=0; i < size_arr_1; ++i) {
            for(int j=0; j < size_arr_2; ++j) {
                ret_1[i][j] = arr_1[i];
                ret_2[i][j] = arr_2[j];
            }
        }
        return std::tuple<Tensor2D<double>, Tensor2D<double>>(ret_1, ret_2);
    };


    //[X,Y]=meshgrid(x,y); // z vectorów x i y stworz kombinacje: wektor X powtorzony len(y) razy w pionie - wektor y powtorzony len(x) razy w poziomie
    //X=X'; // transpozycja -- zawartwa funkcji meshgrid
    //Y=Y'; // transpozycja -- zawarta w funkcji meshgrid
    auto meshgrid_XY = meshgrid(x, M, y, N);
    auto X = std::get<0>(meshgrid_XY); // MxN
    auto Y = std::get<1>(meshgrid_XY); // MxN

    
    //s=X.^2.*(a-X).*Y.^2.*(b-Y);
    Tensor2D<double> s = Create2DArray<double>(M, N);
    std::memset(s[0], 0, M * N *sizeof(double));

    for(int i=0; i < M; ++i) {
        for(int j=0; j < N; ++j) { //elemwise
            // s    =        X.^2.       *     (a-X).     *         Y.^2.       *     (b-Y);
            s[i][j] = (X[i][j] * X[i][j]) * (a - X[i][j]) * (Y[i][j] * Y[i][j]) * (b - Y[i][j]);
        }
    }


    // warunki początkowe - nie potrzebne - wszędzie 0
    // g = Create2DArray(M, N);// zeros(M,N);
    // d = Create3DArray(1, N, K);//  d=zeros(1,N,K);
    // r = Create3DArray(M, 1, K);//  r=zeros(M,1,K);
    // u = Create3DArray(1, N, K);//  u=zeros(1,N,K);
    // l = Create3DArray(M, 1, K);//  l=zeros(M,1,K);;
    // memset(g[0],    0, M * K *sizeof(double));
    // memset(d[0][0], 0, N * K *sizeof(double));
    // memset(r[0][0], 0, M * K *sizeof(double));
    // memset(u[0][0], 0, N * K *sizeof(double));
    // memset(l[0][0], 0, M * K *sizeof(double));
    // f(M,:,:)=d(1,:,:); f(:,N,:)=r(:,1,:); f(1,:,:)=u(1,:,:); f(:,1,:)=l(:,1,:);

    waves = Create3DArray<double>(3, M, N);//  f=zeros(M,N) // two previous timesteps needed

    for(int i=0; i < M; i++) {
        for(int j=0; j < N; j++) {
            waves[0][i][j] = s[i][j];
        }
    }

    Free2DArray(s);
    Free2DArray(Y);
    Free2DArray(X);
    delete[] x;
    delete[] y;

    for(int i=1; i < M-1; i++) {
        for(int j=1; j < N-1; j++) {
            waves[1][i][j] = (1 - px - py) * waves[0][i][j]               // (1-px-py)*f(2:M-1,2:N-1,1)
                          + px * (waves[0][i+1][j] + waves[0][i-1][j])/2  // px*(f(3:M,2:N-1,1)+f(1:M-2,2:N-1,1))/2
                          + py * (waves[0][i][j+1] + waves[0][i][j-1])/2; // py*(f(2:M-1,3:N,1)+f(2:M-1,1:N-2,1))/2
        }
    }

    initialized = true;
}

static std::shared_ptr<Benchmark> CreateBench() {
    return std::make_shared<WaveEquation>("WaveEquation");
}

REGISTER_BENCHMARK(WaveEquation, CreateBench);
