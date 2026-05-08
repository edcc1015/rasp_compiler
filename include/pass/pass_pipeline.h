#ifndef RASP_INCLUDE_PASS_PASS_PIPELINE_H_
#define RASP_INCLUDE_PASS_PASS_PIPELINE_H_

#pragma once

#include "pass/opt/opt_pass.h"
#include "pass/pass_manager.h"
#include "utils/log.h"

static rasp::PassManager build_hlir_pass_pipeline(int opt_level) {
  LOG_I("level = %d" + std::to_string(opt_level));
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::ConstantFolding>());
  pm.add_pass(std::make_shared<rasp::OperatorFusion>());
  pm.add_pass(std::make_shared<rasp::DeadCodeElimination>());
  return pm;
}


#endif /* RASP_INCLUDE_PASS_PASS_PIPELINE_H_ */