add_definitions(-DEXAMPLE_CMAKE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

include_directories(../cylon)

# find CUDA
find_package(CUDAToolkit REQUIRED)

#build a binary with cuda
macro(cylon_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} gcylon)
    install(TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(cylon_add_exe)

cylon_add_exe(cudf_test)
cylon_add_exe(a2a_cudf_test)
cylon_add_exe(gtable_test)
