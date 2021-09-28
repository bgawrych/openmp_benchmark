#include<iostream>
#include "BenchEngine.hpp"
#include <memory>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <rapidjson/istreamwrapper.h>

using namespace rapidjson;

int main(int argc, char *argv[]) {
   if (argc < 3) { //argument passed
      std::cerr << "Provide descriptors and output log filename for benchmarks!";
      return -1;
   }
   auto file = argv[1];
   std::ifstream ifs {file};
   if (!ifs.is_open()) {
      std::cerr << "Could not open descriptor for reading!\n";
      return -1;
   }
   
   IStreamWrapper isw {ifs};
   std::shared_ptr<Document> doc = std::make_shared<Document>();
   doc->ParseStream( isw );

   std::string out_log_path = argv[2];

   
   bool skip_serial = false;
   if (argc > 3) {
      std::string skip_string(argv[3]);
      if (skip_string == "skipserial") {
         skip_serial = true;
      }
   }

   BenchEngine::Start(doc, out_log_path, skip_serial);

   return 0;
}