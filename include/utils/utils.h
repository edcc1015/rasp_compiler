#ifndef RASP_INCLUDE_UTILS_UTILS_H_
#define RASP_INCLUDE_UTILS_UTILS_H_

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "log.h"


namespace rasp {

#define RPC_CHECK(x)                         \
  do {                                       \
    int ret = (x);                           \
    if (ret != rasp::kRpcSuccess) {           \
      if (ret == rasp::kRpcShowInfoEnd)       \
        ret = 0;                             \
      return ret; /* or throw or whatever */ \
    }                                        \
  } while (0)

struct InputArgument {
  std::string json_path;
  std::string output_path  = "model_gen.cpp";  /* .cpp or .so */
  int         opt_level    = 2;
  bool        dump_ir      = false;
  std::string dump_dir     = "./ir_dump";
  /* Concrete input shape parsed from --input-shape, e.g. {1,3,224,224}. */
  std::vector<int64_t> input_shape;
  bool        emit_so      = false;
  std::string cc           = "aarch64-linux-gnu-g++";
};

enum RpcErrCode  {
  kRpcSuccess = 0,
  kRpcError = 1,
  kRpcCmdLineErr = 2,
  kRpcShowInfoEnd = 3
};


} /* namespace rasp */

#endif /* RASP_INCLUDE_UTILS_UTILS_H_ */
