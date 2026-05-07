#ifndef RASP_INCLUDE_PASS_PASS_PIPELINE_H_
#define RASP_INCLUDE_PASS_PASS_PIPELINE_H_

#pragma once

#include "pass/opt/opt_pass.h"
#include "pass/pass_manager.h"

static rasp::PassManager build_hlir_pass_pipeline(int opt_level) {
  rasp::PassManager pm;
  if (opt_level >= 1)
    pm.add_pass(std::make_shared<rasp::ConstantFolding>());
  if (opt_level >= 2)
    pm.add_pass(std::make_shared<rasp::OperatorFusion>());
  return pm;
}


#endif /* RASP_INCLUDE_PASS_PASS_PIPELINE_H_ */