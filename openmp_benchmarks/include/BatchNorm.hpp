#ifndef _BATCHNORM_HPP
#define _BATCHNORM_HPP
#include "Benchmark.hpp"
#include "common.hpp"

class BatchNorm : public Benchmark {
    public:
        BatchNorm(std::string name) {
            this->name = name;
        };
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual void RunSerial() override;
        virtual void RunParallel() override;
        virtual bool Validate() override;
        virtual ~BatchNorm() {
            if(initialized) {
                Free4DArray(input_data);
                Free4DArray(output);
                delete[] mean;
                delete[] variance;
                delete[] gamma;
                delete[] beta;
            }
        }
    private:
        void Reinitialize();

        Logger::LoggerClass* file;
        bool initialized = false;
        int rounds;
        int warmup;
        int N,C,H,W;
        Tensor4D<float> input_data;
        Tensor4D<float> output;
        float* mean;
        float* variance;
        float* gamma;
        float* beta;
};

#endif