add_executable(test_opt_pass test_opt_pass.cpp)
target_link_libraries(test_opt_pass
  PRIVATE rasp_ir rasp_compile_flags
)

add_test(NAME opt_pass COMMAND test_opt_pass)
