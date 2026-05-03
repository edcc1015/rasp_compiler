#pragma once

#include "log.h"


namespace rasp {

enum RpcErrCode  {
  kRpcSuccess = 0,
  kRpcError = 1,
  kRpcCmdLineErr = 2,
  kRpcShowInfoEnd = 3
};

#define RPC_CHECK(x)                         \
  do {                                       \
    int ret = (x);                           \
    if (ret != kRpcSuccess) {           \
      if (ret == kRpcShowInfoEnd)       \
        ret = 0;                             \
      return ret; /* or throw or whatever */ \
    }                                        \
  } while (0)

} /* namespace rasp */