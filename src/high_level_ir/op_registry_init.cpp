/*
 * op_registry_init.cpp — registers all built-in CNN operators at startup.
 *
 * Each registration is performed inside a static-initializer block so that
 * the ops are available before main() runs.  Op names follow the nn.* and
 * plain namespace conventions used by the Python frontend.
 */

#include "high_level_ir/op.h"

namespace rasp {

namespace {

/* Helper: run a lambda in static-init order. */
struct StaticInit {
  template <typename F>
  explicit StaticInit(F&& f) { f(); }
};

static const StaticInit REGISTER_OPS([]() {
  /* ── 2D convolution ────────────────────────────────────────────────── */
  OpRegistry::register_op("nn.conv2d", {
    {"strides",      "ints",   false, "[1,1]"},
    {"padding",      "ints",   false, "[0,0,0,0]"},
    {"dilation",     "ints",   false, "[1,1]"},
    {"groups",       "int",    false, "1"},
    {"data_layout",  "string", false, "NCHW"},
    {"kernel_layout","string", false, "OIHW"},
  }, "2D convolution");

  /* ── Batch normalization ───────────────────────────────────────────── */
  OpRegistry::register_op("nn.batch_norm", {
    {"axis",     "int",   false, "1"},
    {"epsilon",  "float", false, "1e-5"},
    {"momentum", "float", false, "0.1"},
  }, "Batch normalization (inference mode returns normalized tensor only)");

  /* ── Activations ───────────────────────────────────────────────────── */
  OpRegistry::register_op("nn.relu", {}, "Rectified linear unit");

  OpRegistry::register_op("nn.clip", {
    {"a_min", "float", true,  "0"},
    {"a_max", "float", true,  "6"},
  }, "Clip values to [a_min, a_max] (also used for ReLU6)");

  OpRegistry::register_op("nn.sigmoid", {}, "Sigmoid activation: 1/(1+exp(-x))");

  OpRegistry::register_op("nn.softmax", {
    {"axis", "int", false, "-1"},
  }, "Softmax along the specified axis");

  /* ── Pooling ────────────────────────────────────────────────────────── */
  OpRegistry::register_op("nn.max_pool2d", {
    {"pool_size",  "ints",   true,  "[1,1]"},
    {"strides",    "ints",   false, "[1,1]"},
    {"padding",    "ints",   false, "[0,0,0,0]"},
    {"ceil_mode",  "int",    false, "0"},
    {"layout",     "string", false, "NCHW"},
  }, "2D max pooling");

  OpRegistry::register_op("nn.avg_pool2d", {
    {"pool_size",       "ints",   true,  "[1,1]"},
    {"strides",         "ints",   false, "[1,1]"},
    {"padding",         "ints",   false, "[0,0,0,0]"},
    {"count_include_pad","int",   false, "0"},
    {"layout",          "string", false, "NCHW"},
  }, "2D average pooling");

  OpRegistry::register_op("nn.global_avg_pool2d", {
    {"layout", "string", false, "NCHW"},
  }, "Global average pooling (spatial dims → 1×1)");

  /* ── Linear / matmul ───────────────────────────────────────────────── */
  OpRegistry::register_op("nn.dense", {
    {"units",         "int",    false, "0"},
    {"out_dtype",     "string", false, ""},
  }, "Fully connected layer: out = input @ weight.T + bias");

  OpRegistry::register_op("nn.matmul", {
    {"transpose_a", "int", false, "0"},
    {"transpose_b", "int", false, "0"},
  }, "General matrix multiplication");

  /* ── Element-wise / reduction ─────────────────────────────────────── */
  OpRegistry::register_op("nn.add",  {}, "Element-wise addition (broadcast)");

  /* ── Shape manipulation ─────────────────────────────────────────────── */
  OpRegistry::register_op("nn.concatenate", {
    {"axis", "int", true, "0"},
  }, "Concatenate tensors along the given axis");

  OpRegistry::register_op("nn.flatten", {
    {"start_dim", "int", false, "1"},
    {"end_dim",   "int", false, "-1"},
  }, "Flatten dimensions [start_dim, end_dim] into one");

  OpRegistry::register_op("nn.reshape", {
    {"newshape", "ints", true, ""},
  }, "Reshape tensor to the given shape");
});

} /* anonymous namespace */

} /* namespace rasp */
