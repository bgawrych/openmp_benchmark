#ifndef _WAVEEQUATION_HPP
#define _WAVEEQUATION_HPP
#include "Benchmark.hpp"
#include "common.hpp"
#include <complex>

class WaveEquation : public Benchmark {
    public:
        WaveEquation(std::string name) {
            this->name = name;
        };
        virtual void RunSerial() override;
        virtual void RunParallel_1();
        virtual void RunParallel_2();
        virtual void RunParallel() override;
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual bool Validate() override;

        virtual ~WaveEquation() {
            if(initialized) {
                Free3DArray<double>(waves);
            }
        }
    private:
        void Reinitialize();

        Logger::LoggerClass* file;
        int rounds;
        int warmup;
        bool initialized = false;
        
        double v, a, b, q, w;
        int M, N, K, beta;
        double dx, dy, dt, px, py;

        Tensor3D<double> waves;
};

#endif