#ifndef RASP_INCLUDE_UTILS_UTILS_H_
#define RASP_INCLUDE_UTILS_UTILS_H_

#pragma once

#include <sstream>

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
  bool dump_ir = false;
  std::string dump_dir = "./ir_dump";
  int opt_level = 2;
  std::ostringstream input_ir_str_buf;
};

enum RpcErrCode  {
  kRpcSuccess = 0,
  kRpcError = 1,
  kRpcCmdLineErr = 2,
  kRpcShowInfoEnd = 3
};


} /* namespace rasp */

#endif /* RASP_INCLUDE_UTILS_UTILS_H_ */
