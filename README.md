# RASP Compiler

> 一个面向树莓派 4B（ARMv8-A + NEON SIMD）的精简深度学习编译器，支持 ONNX/PyTorch CNN 模型的解析、优化与代码生成。

深度学习推理部署在边缘设备上面临性能瓶颈。TVM 等工业级编译器虽然功能全面，但体量庞大、二次开发门槛高。RASP Compiler 以 TVM 为参考原型，实现一个功能完整、架构清晰的精简版深度学习编译器，专为树莓派 4B（Cortex-A72，ARMv8-A，NEON SIMD）优化。