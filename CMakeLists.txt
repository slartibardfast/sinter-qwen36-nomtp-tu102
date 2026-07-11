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

add_custom_target(repro-check
    COMMAND ${CMAKE_SOURCE_DIR}/tools/repro-check.sh ${CMAKE_SOURCE_DIR}
    COMMENT "double-build byte-identity check")
