cmake_minimum_required(VERSION 3.24)
project(megakernel LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Reproducible-build discipline (greenfield, .host-software): no embedded
# absolute paths, no timestamps. repro-check builds twice and compares.
# --objdir-as-tempdir keeps nvcc's tmpxft_<pid> temp names (which leak into
# the fatbin module id) deterministic.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>")

# ptxas resource accounting for the G11 residency gate: every CUDA target
# compiles with -v so tools/check-residency.sh can gate the build.
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>")

# Rig precheck: device props, cooperative-launch support, SM clock lock.
add_executable(mk-devinfo tests/devinfo.cu)

# The interpreter spine: persistent kernel + host upload/run library.
find_package(CUDAToolkit REQUIRED)
add_library(mk-core STATIC core/interp.cu core/host.cpp)
target_link_libraries(mk-core PUBLIC CUDA::cudart)

# Every tests/test_*.cu is a standalone test executable (op unit tests add
# a file, not a CMake edit).
file(GLOB MK_TEST_SOURCES CONFIGURE_DEPENDS tests/test_*.cu)
foreach(src ${MK_TEST_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    string(REPLACE "_" "-" tgt "mk-${name}")
    add_executable(${tgt} ${src})
    target_link_libraries(${tgt} PRIVATE mk-core)
endforeach()

# G14 litmus family: positive + the two negatives (which must be CAUGHT).
add_executable(mk-litmus-boundary tests/litmus_boundary.cu)
add_executable(mk-litmus-boundary-plain tests/litmus_boundary.cu)
target_compile_definitions(mk-litmus-boundary-plain PRIVATE LITMUS_PLAIN_READS)
add_executable(mk-litmus-boundary-norelease tests/litmus_boundary.cu)
target_compile_definitions(mk-litmus-boundary-norelease PRIVATE LITMUS_NO_RELEASE)

add_custom_target(repro-check
    COMMAND ${CMAKE_SOURCE_DIR}/tools/repro-check.sh ${CMAKE_SOURCE_DIR}
    COMMENT "double-build byte-identity check")

# Intra-GPU MP probe: does the local stale-L1 read exist at all?
add_executable(mk-litmus-mp tests/litmus_boundary.cu)
target_compile_definitions(mk-litmus-mp PRIVATE LITMUS_MP)
add_executable(mk-litmus-mp-cg tests/litmus_boundary.cu)
target_compile_definitions(mk-litmus-mp-cg PRIVATE LITMUS_MP LITMUS_MP_CG)

# mk-harness: the standalone parity/bench harness (k0/harness.cpp + the
# GGUF mmap reader k0/gguf.h). Compiled as CUDA (nvcc) because the packer
# includes the op Args structs from k0/ops/*.cuh (half types, __device__
# wrappers); links mk-core so the persistent interpreter kernel is launchable.
find_package(CUDAToolkit REQUIRED)
set_source_files_properties(k0/harness.cpp PROPERTIES LANGUAGE CUDA)
add_executable(mk-harness k0/harness.cpp)
target_link_libraries(mk-harness PRIVATE mk-core CUDA::cudart)
