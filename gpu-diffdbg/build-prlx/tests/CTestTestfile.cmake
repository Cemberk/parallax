# CMake generated Testfile for 
# Source directory: /home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/tests
# Build directory: /home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[manual_trace_test]=] "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/tests/manual_trace_test")
set_tests_properties([=[manual_trace_test]=] PROPERTIES  ENVIRONMENT "PRLX_TRACE=/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/tests/manual_trace.prlx" _BACKTRACE_TRIPLES "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/tests/CMakeLists.txt;25;add_test;/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/tests/CMakeLists.txt;0;")
