# Host-only tests for NVBit tool components (no GPU required).
# These test site_table and trace_writer logic on the CPU.

# Site table tests
add_executable(test_nvbit_site_table
    test_site_table.cpp
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool/site_table.cu
)
target_include_directories(test_nvbit_site_table PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool
    ${CMAKE_SOURCE_DIR}/lib/common
)
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool/site_table.cu
    PROPERTIES LANGUAGE CXX
)
set_target_properties(test_nvbit_site_table PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME nvbit_site_table COMMAND test_nvbit_site_table)

# Trace writer tests
add_executable(test_nvbit_trace_writer
    test_trace_writer.cpp
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool/trace_writer.cu
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool/site_table.cu
)
target_include_directories(test_nvbit_trace_writer PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool
    ${CMAKE_SOURCE_DIR}/lib/common
)
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool/trace_writer.cu
    ${CMAKE_SOURCE_DIR}/lib/nvbit_tool/site_table.cu
    PROPERTIES LANGUAGE CXX
)
set_target_properties(test_nvbit_trace_writer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME nvbit_trace_writer COMMAND test_nvbit_trace_writer)
