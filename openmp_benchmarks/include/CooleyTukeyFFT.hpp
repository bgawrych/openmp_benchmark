#ifndef _COOLEYTUKEYFFT_HPP
#define _COOLEYTUKEYFFT_HPP
#include "Benchmark.hpp"
#include "common.hpp"
#include <complex>

class CooleyTukeyFFT : public Benchmark {
    public:
        CooleyTukeyFFT(std::string name) {
            this->name = name;
        };
        virtual void RunSerial() override;
        virtual void RunParallel() override;
        void RunParallel_Bad();
        void RunParallel_Final();
        void RunParallel_Single();
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual bool Validate() override;

        virtual ~CooleyTukeyFFT() {
            if(initialized) {
                delete[] input;
                delete[] output;
            }
        }
    private:
        void Reinitialize();

        Logger::LoggerClass* file;
        int rounds;
        int warmup; 
        bool initialized = false;
        int size;
        double* input;
        std::complex<double>* output;
};

#endif