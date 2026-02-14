# CMake generated Testfile for 
# Source directory: /home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples
# Build directory: /home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[example_loop_divergence]=] "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples/loop_divergence" "10")
set_tests_properties([=[example_loop_divergence]=] PROPERTIES  ENVIRONMENT "PRLX_TRACE=/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples/loop_test.prlx" _BACKTRACE_TRIPLES "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples/CMakeLists.txt;76;add_test;/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples/CMakeLists.txt;0;")
add_test([=[example_occupancy_light]=] "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples/occupancy_test" "light")
set_tests_properties([=[example_occupancy_light]=] PROPERTIES  ENVIRONMENT "PRLX_TRACE=/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples/occupancy_test.prlx" _BACKTRACE_TRIPLES "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples/CMakeLists.txt;81;add_test;/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples/CMakeLists.txt;0;")
add_test([=[example_matmul_correct]=] "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples/matmul_divergence" "correct")
set_tests_properties([=[example_matmul_correct]=] PROPERTIES  ENVIRONMENT "PRLX_TRACE=/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/build-prlx/examples/matmul_test.prlx" _BACKTRACE_TRIPLES "/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples/CMakeLists.txt;86;add_test;/home/khushiyant/Develop/experiments/parallax/gpu-diffdbg/examples/CMakeLists.txt;0;")
