#ifndef _BENCHMARK_HPP
#define _BENCHMARK_HPP

#include <functional>
#include <memory>
#include <unordered_map>
#include "Logger.hpp"
#include "rapidjson/document.h"

class Benchmark {
    public:
        virtual void RunSerial() = 0;
        virtual void RunParallel() = 0;
        virtual bool Validate() {
            return true;
        };
        virtual void Init(Logger::LoggerClass* file, const rapidjson::Value& properties) = 0;
    protected:
        std::string name = "Benchmark";
        std::string descriptor = "";
};


typedef std::shared_ptr<Benchmark> BenchmarkPtr;
typedef std::function<std::shared_ptr<Benchmark>()> create_bench_fn_t;
typedef std::unordered_map<std::string, create_bench_fn_t> benchmark_creators_map_t;

struct BenchmarkRegister
{
private:
    std::shared_ptr<benchmark_creators_map_t> reg ;
    BenchmarkRegister() {
        reg = std::make_shared<benchmark_creators_map_t>();
    }
    BenchmarkRegister(const BenchmarkRegister & source) {}
    BenchmarkRegister(BenchmarkRegister && source) {}

public:
    static BenchmarkRegister& Get() {
        static BenchmarkRegister instance;
        return instance;
    }

    void Add(std::string name, create_bench_fn_t fn) {
        reg->insert({name, fn});
    }

    BenchmarkPtr Find(std::string name) {
        auto benchmark = reg->find(name);

        if(benchmark == reg->end()) {
            throw std::invalid_argument("No such benchmark as " + name);
            return nullptr;
        }

        return benchmark->second();
    }
};


static bool RegisterBenchmark(std::string name, create_bench_fn_t callback) {
    BenchmarkRegister::Get().Add(name, callback);
    return true;
}

static std::shared_ptr<Benchmark> GetBenchmark(std::string name) {
    return BenchmarkRegister::Get().Find(name);;
}

#define REGISTER_BENCHMARK(name, create_fn) \
        static bool name ## __LINE__ = RegisterBenchmark(#name, create_fn)

#endif