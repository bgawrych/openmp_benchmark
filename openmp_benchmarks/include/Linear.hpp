#ifndef _LINEAR_HPP
#define _LINEAR_HPP
#include "Benchmark.hpp"
#include "common.hpp"

class Linear : public Benchmark {
    public:
        Linear(std::string name) {
            this->name = name;
        };
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual void RunSerial() override;
        virtual void RunParallel() override;
        void RunParallel_1();
        void RunParallel_2();
        void RunParallel_3();
        void RunParallel_4();
        virtual bool Validate() override;
        virtual ~Linear() {
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
        float* input;
        float* output;
};

#endif
