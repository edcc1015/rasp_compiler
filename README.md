# RASP Compiler

> 一个面向树莓派 4B（ARMv8-A + NEON SIMD）的精简深度学习编译器，支持 ONNX/PyTorch CNN 模型的解析、优化与代码生成。

深度学习推理部署在边缘设备上面临性能瓶颈。TVM 等工业级编译器虽然功能全面，但体量庞大、二次开发门槛高。RASP Compiler 以 TVM 为参考原型，实现一个功能完整、架构清晰的精简版深度学习编译器，专为树莓派 4B（Cortex-A72，ARMv8-A，NEON SIMD）优化。

## 单元测试方法
1. 新建测试源文件
在 `tests/<模块名>/test_<模块名>.cpp` 里写：

```cpp
#include "你的头文件.h"

static bool test_xxx() {
/* 构造输入，运行代码，检查结果 */
bool ok = (result == expected);
LOG_I(ok ? "test_xxx PASS" : "test_xxx FAIL");
return ok;
}

int main() {
bool ok = true;
ok &= test_xxx();
/* ... 更多测试 ... */
return ok ? 0 : 1;  /* 返回非 0 → CTest 标记失败 */
}
```

2. 新建 `tests/<模块名>/CMakeLists.txt`
```cmake
add_executable(test_<模块名> test_<模块名>.cpp)
target_link_libraries(test_<模块名>
PRIVATE rasp_ir rasp_compile_flags
)
add_test(NAME <模块名> COMMAND test_<模块名>)
```

3. 在 `tests/CMakeLists.txt` 里注册
```cmake
add_subdirectory(opt_pass)
add_subdirectory(<模块名>)   # ← 加这一行

4. 之后运行 `./build.sh --test` 就会自动构建并跑所有测试。