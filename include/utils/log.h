#pragma once
#include <string>
#include <iostream>

#define ANSI_RED    "\033[31m"
#define ANSI_GREEN  "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_RESET  "\033[0m"

#define LOG_E(msg) \
    std::cerr << ANSI_RED    "[Error] " ANSI_RESET << (msg) << "\n"

#define LOG_I(msg) \
    std::cout << ANSI_GREEN  "[Info]  " ANSI_RESET << (msg) << "\n"

#define LOG_D(msg) \
    std::cout << ANSI_YELLOW "[Debug] " ANSI_RESET << (msg) << "\n"
