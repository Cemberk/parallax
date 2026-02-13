# Trace analysis tools

add_executable(dump_trace dump_trace.cpp)

target_include_directories(dump_trace PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/common
)

install(TARGETS dump_trace
    RUNTIME DESTINATION bin
)
