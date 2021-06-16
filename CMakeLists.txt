cmake_minimum_required(VERSION 3.5)

project(CellInstancing LANGUAGES CXX)

#set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_NAME "CellInstancing")
set(homeDir "/home/geotyper")


ADD_DEFINITIONS(-DENTRY_CONFIG_USE_SDL=2)
ADD_DEFINITIONS(-DRUNTIME_DIR="/home/geotyper/CPP_bgfx_shaders/runtime")
ADD_DEFINITIONS(-DBGFX_DIR="/home/geotyper/CPP_bgfx_shaders/runtime/")

ADD_DEFINITIONS(-DBGFX_BUILDDIR="/home/geotyper/CPP_bgfx_shaders/runtime/")
message(BGFX_BUILDDIR)

set(core_math  ${homeDir}/CPP_lib/Core_math)
include_directories(${core_math})

set(BGFX_INCLUDE  ${homeDir}/CPP_lib/bgfx/include)
set(BX_INCLUDE  ${homeDir}/CPP_lib/bx/include)
set(BIMG_INCLUDE  ${homeDir}/CPP_lib/bimg/include)
set(BGFX_COMMON_INCLUDE  ${homeDir}/CPP_bgfx_shaders/common)
set(BGFX_THIRDPARTY_INCLUDE  ${homeDir}/CPP_lib/bgfx/3rdparty)


include_directories(${BGFX_INCLUDE})
include_directories(${BX_INCLUDE})
include_directories(${BIMG_INCLUDE})
include_directories(${BGFX_THIRDPARTY_INCLUDE})
include_directories(${BGFX_COMMON_INCLUDE})


set(BGFX_LIB  ${homeDir}/CPP_lib/bgfx/.build/linux64_gcc/bin/)
#set(BGFX_LIB  "/home/geotyper/CPP_lib/bgfx/.build/linux64_gcc/bin/")

file(GLOB HEADERSE
    "common/entry/*.h"
    )
file(GLOB SRCE
    "common/entry/*.cpp"
    )


file(GLOB HEADERSB
   "common/bgfx_utils.h"
    )
file(GLOB SRCB
    "common/bgfx_utils.cpp"
    )
# Local source files here

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

find_package(CUDA REQUIRED)


add_library(Common_bgfx #STATIC
    ${HEADERSE} ${SRCE}   ${HEADERSB} ${SRCB} )

target_include_directories(Common_bgfx  PRIVATE)# ${BGFX_INCLUDE} ${BX_INCLUDE} ${BIMG_INCLUDE} ${BGFX_THIRDPARTY_INCLUDE})
target_link_libraries(Common_bgfx  ${SDL2_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode")
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbgfxDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbxDebug.a ${BGFX_LIB}libfcppDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libglslangDebug.a ${BGFX_LIB}libglsl-optimizerDebug.a ${BGFX_LIB}libspirv-crossDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbgfx-shared-libDebug.so)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libspirv-optDebug.a)
    #target_link_libraries(Common_bgfx ${BGFX_LIB}libexample-glueDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbimgDebug.a ${BGFX_LIB}libbimg_decodeDebug.a ${BGFX_LIB}libbimg_encodeDebug.a)
    target_link_libraries(Common_bgfx -lGL -lX11 -ldl -lpthread -lrt)
else()
    message("release mode")
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbgfxRelease.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbxRelease.a ${BGFX_LIB}libfcppRelease.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libglslangRelease.a ${BGFX_LIB}libglsl-optimizerRelease.a ${BGFX_LIB}libspirv-crossRelease.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbgfx-shared-libRelease.so)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libspirv-optRelease.a)

    target_link_libraries(Common_bgfx ${BGFX_LIB}libbimgRelease.a ${BGFX_LIB}libbimg_decodeRelease.a ${BGFX_LIB}libbimg_encodeRelease.a)
    target_link_libraries(Common_bgfx -lGL -lX11 -ldl -lpthread -lrt)

endif()

SET(TARGET_H
    mainloop.hpp

    particle/body.hpp
    particle/constraint.hpp
    environment.hpp
    SupportBGFX.hpp
    GraphModule.hpp
    #utils.hpp
    Common_eigen.hpp
    constants.hpp


    )

SET(TARGET_SRC

    environment.cpp
    mainloop.cpp
    main.cpp
    )


#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_61,code=sm_61
#    )

# Current arch's: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
set(BUILD_CUDA_ARCH "62;70;72;75" CACHE STRING
    "CUDA Architectures to build, default: 62;70;72;75")

# Add the compilation trajectories for different architectures.
# See: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-examples
# I have no specific need for complicated build trajectories.
foreach(arch ${BUILD_CUDA_ARCH})
  # virtually real
  list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_${arch},code=sm_${arch}")
  # virtually virtual
  list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_${arch},code=compute_${arch}")
endforeach()

set_source_files_properties(${HEADERSE} ${SRCE} PROPERTIES COMPILE_FLAGS "-fno-rtti")

#add_executable(${EXECUTABLE_NAME} ${TARGET_SRC} ${TARGET_H})
cuda_add_executable(${EXECUTABLE_NAME} ${TARGET_SRC} ${TARGET_H})

target_link_libraries(${EXECUTABLE_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME} Common_bgfx)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode")
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfxDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbxDebug.a ${BGFX_LIB}libfcppDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libglslangDebug.a ${BGFX_LIB}libglsl-optimizerDebug.a ${BGFX_LIB}libspirv-crossDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfx-shared-libDebug.so)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-commonDebug.a)
    #target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-glueDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libspirv-optDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbimgDebug.a ${BGFX_LIB}libbimg_decodeDebug.a ${BGFX_LIB}libbimg_encodeDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} -lGL -lX11 -ldl -lpthread -lrt)
else()
    message("release mode")
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfxRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbxRelease.a ${BGFX_LIB}libfcppRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libglslangRelease.a ${BGFX_LIB}libglsl-optimizerRelease.a ${BGFX_LIB}libspirv-crossRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfx-shared-libRelease.so)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-commonRelease.a)
    #target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-glueRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libspirv-optRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbimgRelease.a ${BGFX_LIB}libbimg_decodeRelease.a ${BGFX_LIB}libbimg_encodeRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} -lGL -lX11 -ldl -lpthread -lrt)

endif()

#target_include_directories(${EXECUTABLE_NAME} Common_bgfx)
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${INCLUDES} ${core_math} )
