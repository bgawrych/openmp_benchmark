#include <iostream>
#include "BenchEngine.hpp"

#include <stdexcept>

using JsonBenchmark = rapidjson::Value::ConstValueIterator;

void BenchEngine::Start(JsonPtr json, std::string out_log_path, bool skip_serial) {
    std::fstream excelStream(out_log_path, std::ios::out);
    auto excel = Logger::EXCEL(excelStream);

    Logger::INFO << "Starting benchmark engine!";
    Logger::INFO << "Executing following benchmarks:";
    
    for (JsonBenchmark itr = json->Begin(); itr != json->End(); ++itr) {
         Logger::INFO << " \t- " << itr->GetObject()["name"].GetString();
    }

    // Create Excel Columns Titles
    excel << "Name" << "Descriptor" << "Mode" << "Warmup" << "Iterations" << "Duration" << "Mean" << "StdDev";
    excel.newLine();

    for (JsonBenchmark itr = json->Begin(); itr != json->End(); ++itr) {
        auto benchDescriptor = itr->GetObject();
        auto name = benchDescriptor["name"].GetString();
        
        bool is_validation = false;
        if (benchDescriptor.FindMember("validate") != benchDescriptor.MemberEnd())
            is_validation = benchDescriptor["validate"].GetBool();

        auto bench = GetBenchmark(name);
        const rapidjson::Value& properties = benchDescriptor["properties"];

        Logger::INFO << "[" << name << "] " << "Initializing benchmark";
        bench->Init(&excel, properties);

        if(is_validation) {
            Logger::INFO << "[" << name << "] " << "Starting benchmark validation";
            if(bench->Validate()) {
                Logger::INFO << "[" << name << "] " << "Successfully validated";
            } else {
                Logger::ERROR << "[" << name << "] " << "Validation failed";
            }
            Logger::INFO << "==============================================================";
        } else {
            Logger::INFO << "[" << name << "] " << "Starting benchmark";
            /* Benchmark Parallel Function With OpenMP */
            bench->RunParallel();
            Logger::INFO << "[" << name << "] " << "Finished parallel benchmark";
            if(!skip_serial) {
                /* Benchmark Serial Function */
                bench->RunSerial();
                Logger::INFO << "[" << name << "] " << "Finished serial benchmark";
            } else {
                Logger::INFO << "[" << name << "] " << "Skipping serial benchmark";
            }
            excel.newLine();
            Logger::INFO << "==============================================================";
        }
     }
}
