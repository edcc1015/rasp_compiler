add_executable(test_lowering test_lowering.cpp)
target_link_libraries(test_lowering
  PRIVATE rasp_ir rasp_compile_flags
)

add_test(NAME lowering COMMAND test_lowering)
