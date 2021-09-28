#ifndef _QUICKSORT_HPP
#define _QUICKSORT_HPP
#include "Benchmark.hpp"
#include "common.hpp"

class QuickSort : public Benchmark {
    public:
        QuickSort(std::string name) {
            this->name = name;
        };
        virtual void RunSerial() override;
        virtual void RunParallel() override;
        void RunParallelSingle();
        void RunParallelDouble();
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) override;
        virtual bool Validate() override;
        virtual ~QuickSort() {
            if(initialized) {
                delete[] input_data;
                delete[] output_data;
            }
        }
    private:
        void Reinitialize();

        Logger::LoggerClass* file;
        int rounds;
        int warmup;
        bool initialized = false;
        int size;
        float* input_data;  // random data
        float* output_data;        // copy input data into new memory as quicksort is inplace
};

#endif
