#ifndef _BENCHENGINE_HPP
#define _BENCHENGINE_HPP
#include "common.hpp"
#include "Benchmark.hpp"
 
class BenchEngine {
    public:
        static void Start(JsonPtr json, std::string out_log_path, bool skip_serial=false);
    private:

};
#endif