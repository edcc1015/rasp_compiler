add_executable(test_type_inference test_type_inference.cpp)
target_link_libraries(test_type_inference
  PRIVATE rasp_ir rasp_compile_flags
)

add_test(NAME type_inference COMMAND test_type_inference)
