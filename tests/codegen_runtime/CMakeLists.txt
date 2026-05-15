add_executable(test_codegen_runtime test_codegen_runtime.cpp)
target_link_libraries(test_codegen_runtime
  PRIVATE rasp_ir rasp_compile_flags
)

add_test(NAME codegen_runtime COMMAND test_codegen_runtime)
