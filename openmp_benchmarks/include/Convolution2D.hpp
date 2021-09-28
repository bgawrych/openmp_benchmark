#ifndef _CONVOLUTION2D_HPP
#define _CONVOLUTION2D_HPP
#include "Benchmark.hpp"
#include "common.hpp"

class Convolution2D : public Benchmark {
    public:
        Convolution2D(std::string name) {
            this->name = name;
        };
        virtual void RunSerial() override;
        virtual void RunParallel() override;
        void RunParallel_1();
        void RunParallel_2();
        void RunParallel_3();
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual bool Validate() override;
        virtual ~Convolution2D() {
            if(initialized) {
                Free3DArray<float>(input_data);
                Free2DArray<float>(kernel_data);
                Free3DArray<float>(result);
            }
        }
    private:
        void Reinitialize();

        Logger::LoggerClass* file;
        int rounds;
        int warmup; 
        bool initialized = false;
        int N, H, W, kernel;
        float*** input_data;
        float** kernel_data;
        float*** result;
};

#endif