cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
  message(CMAKE_CUDA_ARCHITECTURES)
endif()

project(Cuda_particles LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(EXECUTABLE_NAME "particles")
set(homeDir "/home/geotyper")

# BGFX Section
ADD_DEFINITIONS(-DENTRY_CONFIG_USE_SDL=1)
#ADD_DEFINITIONS(-DBX_CONFIG_SUPPORTS_THREADING=no)
ADD_DEFINITIONS(-DRUNTIME_DIR="/home/geotyper/CPP_bgfx_shaders/runtime")
ADD_DEFINITIONS(-DBGFX_DIR="/home/geotyper/CPP_bgfx_shaders/runtime/")

ADD_DEFINITIONS(-DBGFX_BUILDDIR="/home/geotyper/CPP_bgfx_shaders/runtime/")
message(BGFX_BUILDDIR)

#set(core_math  ${homeDir}/CPP_lib/Core_math)
#include_directories(${core_math})


set(stb_lib  ${homeDir}/CPP_lib/stb)
include_directories(${stb_lib})


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
    "common/*.h"
    "common/debugdraw/*.h"
    "common/imgui/*.h"
    "common/font/*.h"

    # "3dparty/dear-imgui/*.h"
    )
file(GLOB SRCE
    "common/entry/*.cpp"
    "common/*.cpp"
    "common/debugdraw/*.cpp"
    "common/debugdraw/*.sc"

    "common/imgui/*.cpp"
    "common/imgui/*.sc"

    "common/font/*.cpp"
    "common/font/*.sc"

    "3rdparty/dear-imgui/*.cpp"
    "3rdparty/meshoptimizer/src/*.cpp"

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

add_library(Common_bgfx STATIC ${HEADERSE} ${SRCE}   ${HEADERSB} ${SRCB} )

target_include_directories(Common_bgfx  PRIVATE ${BGFX_INCLUDE} ${BX_INCLUDE} ${BIMG_INCLUDE} ${BGFX_THIRDPARTY_INCLUDE} )
target_link_libraries(Common_bgfx  ${SDL2_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode")
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbgfxDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbxDebug.a ${BGFX_LIB}libfcppDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libglslangDebug.a ${BGFX_LIB}libglsl-optimizerDebug.a ${BGFX_LIB}libspirv-crossDebug.a)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libbgfx-shared-libDebug.so)
    target_link_libraries(Common_bgfx ${BGFX_LIB}libspirv-optDebug.a)
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
    environment.hpp
    SupportBGFX.hpp
    SupportBGFX.hpp
    constants.hpp
    cuda_exp.hpp
    bgfxappl.hpp

    CubeMapLib.h
    VL234f.h

    )

SET(TARGET_SRC
    mainloop.cpp
    main.cpp
    cuda_exp.cpp
    environment.cpp
    bgfxappl.cpp

    CubeMapLib.cpp

    )

SET(TARGET_CUDA_H
    particles_kernel.cuh
    particles_kernel_impl.cuh
    particleSystem.cuh
    CUDA/Array.cuh
    CUDA/CudaMemoryManager.cuh
    CUDA/Swap.cuh
    CUDA/Base.cuh
    CUDA/Macros.cuh
   # CUDA/Definitions.cuh
    CUDA/CudaConstants.h
    CUDA/tinycolormap.hpp
    CUDA/checkCudaCall.hpp
)

SET(TARGET_CUDA_SRC
    particles_kernel.cu
    particleSystem_cuda.cu
)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

#set(OpenGL_GL_PREFERENCE LEGACY)


set(Eigen_include "/home/geotyper/CPP_lib/Eigen/Eigen")
include_directories(${Eigen_include})

set(Cuda_common_include "/usr/local/cuda/samples/common/inc")
include_directories(${Cuda_common_include})

message(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

#add_definitions(-DHAVE_THRUST)
### ViperLib
#find_package (Eigen3 3.3.7 REQUIRED)

add_executable(${EXECUTABLE_NAME}  ${TARGET_CUDA_H} ${TARGET_CUDA_SRC} ${TARGET_SRC} ${TARGET_H})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_ARCHITECTURES "61")
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES})
#target_link_libraries (${EXECUTABLE_NAME} Eigen3::Eigen)

target_link_libraries(${EXECUTABLE_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME} ${GLUT_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME} Common_bgfx)


set_source_files_properties(${HEADERSE} ${SRCE} PROPERTIES COMPILE_FLAGS "-fno-rtti")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode")
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfxDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbxDebug.a ${BGFX_LIB}libfcppDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libglslangDebug.a ${BGFX_LIB}libglsl-optimizerDebug.a ${BGFX_LIB}libspirv-crossDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfx-shared-libDebug.so)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB} Common_bgfx)
    #target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-glueDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libspirv-optDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbimgDebug.a ${BGFX_LIB}libbimg_decodeDebug.a ${BGFX_LIB}libbimg_encodeDebug.a)
    target_link_libraries(${EXECUTABLE_NAME} -lX11 -ldl -lpthread -lrt)
else()
    message("release mode")
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfxRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbxRelease.a ${BGFX_LIB}libfcppRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libglslangRelease.a ${BGFX_LIB}libglsl-optimizerRelease.a ${BGFX_LIB}libspirv-crossRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbgfx-shared-libRelease.so)
    #target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-commonRelease.a)
    #target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libexample-glueRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libspirv-optRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} ${BGFX_LIB}libbimgRelease.a ${BGFX_LIB}libbimg_decodeRelease.a ${BGFX_LIB}libbimg_encodeRelease.a)
    target_link_libraries(${EXECUTABLE_NAME} -lX11 -ldl -lpthread -lrt)

endif()

#target_include_directories(${EXECUTABLE_NAME} Common_bgfx)
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${INCLUDES} ${core_math} )
target_link_libraries(${EXECUTABLE_NAME} ${OPENGL_LIBRARIES})
