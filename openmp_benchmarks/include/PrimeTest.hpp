#ifndef _PRIMETEST_HPP
#define _PRIMETEST_HPP
#include "Benchmark.hpp"
#include "common.hpp"

class PrimeTest : public Benchmark {
    public:
        PrimeTest(std::string name) {
            this->name = name;
        };
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual void RunSerial() override;
        virtual void RunParallel() override;
        void RunParallel_1();
        void RunParallel_2();
        void RunParallel_3();
        virtual bool Validate() override;
        virtual ~PrimeTest() {
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
        int* input;
        bool* output;
};

#endif