cmake_minimum_required(VERSION 3.24)
project(trellis2_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(FA ${CMAKE_SOURCE_DIR}/third_party/flash-attention)
# flash-attn kernels built with the exact flags of its setup.py (bitwise-identical SASS to the official wheel build)
add_library(t2flash STATIC src/flash.cu ${FA}/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu)
target_include_directories(t2flash PUBLIC src ${CMAKE_SOURCE_DIR}/third_party/shim ${FA}/csrc/flash_attn ${FA}/csrc/flash_attn/src ${FA}/csrc/cutlass/include)
target_compile_options(t2flash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
# FlashAttention-3 (hopper/) forward for the fast build, its setup.py flags; FA2 symbol names renamed away
add_library(t2fa3 STATIC src/fa3.cu ${FA}/hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu)
set_target_properties(t2fa3 PROPERTIES CUDA_ARCHITECTURES OFF)
target_include_directories(t2fa3 PRIVATE ${FA}/hopper ${FA}/csrc/cutlass/include)
target_compile_definitions(t2fa3 PRIVATE flash=flash3 Flash_fwd_params=Flash3_fwd_params Flash_bwd_params=Flash3_bwd_params Qkv_params=Qkv3_params
  CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED CUTLASS_DEBUG_TRACE_LEVEL=0 NDEBUG FLASHATTENTION_DISABLE_BACKWARD FLASHATTENTION_DISABLE_FP8
  FLASHATTENTION_DISABLE_FP16 FLASHATTENTION_DISABLE_SM8x)
target_compile_options(t2fa3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
  --ftemplate-backtrace-limit=0 -gencode arch=compute_90a,code=sm_90a>)

# Everything else: no fast-math (ATen is built without it), default FMA contraction like ATen.
# Link the same cuDNN torch loads (pip nvidia-cudnn-cu12) when present
execute_process(COMMAND python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))"
                OUTPUT_VARIABLE PIP_CUDNN OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
find_path(CUDNN_INC cudnn.h HINTS ${CUDNN_ROOT}/include ${PIP_CUDNN}/include /usr/include)
find_library(CUDNN_LIB NAMES cudnn libcudnn.so.9 HINTS ${CUDNN_ROOT}/lib ${PIP_CUDNN}/lib /usr/lib/x86_64-linux-gnu)
# ATen's vendored mem-efficient attention (fp32 SDPA) against torch v2.6.0's pinned cutlass
set(PT ${CMAKE_SOURCE_DIR}/third_party/pytorch)
add_library(t2mea STATIC src/mea.cu)
target_include_directories(t2mea PRIVATE src ${CMAKE_SOURCE_DIR}/third_party/shim ${PT}/aten/src ${CMAKE_SOURCE_DIR}/third_party/cutlass-pt/include)
target_compile_options(t2mea PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda>)

# CuMesh (mesh post-processing) with its setup.py flags; torch API served by third_party/shim
set(CM ${CMAKE_SOURCE_DIR}/third_party/CuMesh/src)
add_library(t2cumesh STATIC ${CM}/hash/hash.cu ${CM}/atlas.cu ${CM}/clean_up.cu ${CM}/cumesh.cu ${CM}/connectivity.cu ${CM}/geometry.cu
  ${CM}/io.cu ${CM}/simplify.cu ${CM}/shared.cu ${CM}/remesh/simple_dual_contour.cu ${CM}/remesh/svox2vert.cu
  ${CM}/../third_party/cubvh/src/bvh.cu ${CM}/../third_party/xatlas/xatlas.cpp)
set_source_files_properties(${CM}/../third_party/cubvh/src/bvh.cu PROPERTIES COMPILE_OPTIONS
  "--extended-lambda;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__")
target_include_directories(t2cumesh PUBLIC ${CMAKE_SOURCE_DIR}/third_party/shim ${CMAKE_SOURCE_DIR}/src ${CM}
  ${CM}/../third_party/cubvh/include ${CM}/../third_party/cubvh/third_party/eigen ${CM}/../third_party/xatlas)
target_compile_options(t2cumesh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# FlexGEMM grid_sample neighbor/weight kernels with its setup.py flags (fast-math)
set(FGK ${CMAKE_SOURCE_DIR}/third_party/FlexGEMM/flex_gemm/kernels/cuda)
add_library(t2flexgemm STATIC ${FGK}/grid_sample/grid_sample.cu ${FGK}/hash/hash.cu)
target_include_directories(t2flexgemm PRIVATE ${CMAKE_SOURCE_DIR}/third_party/shim ${CMAKE_SOURCE_DIR}/src)
target_compile_options(t2flexgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>)

# nvdiffrast v0.4.0 rasterizer core with its setup.py flags; torch glue replaced by src/nvdr.cpp
set(NV ${CMAKE_SOURCE_DIR}/third_party/nvdiffrast/csrc)
add_library(t2nvdr STATIC src/nvdr.cpp ${NV}/common/rasterize.cu ${NV}/common/interpolate.cu ${NV}/common/common.cpp
  ${NV}/common/cudaraster/impl/Buffer.cpp ${NV}/common/cudaraster/impl/CudaRaster.cpp ${NV}/common/cudaraster/impl/RasterImpl.cpp
  ${NV}/common/cudaraster/impl/RasterImpl_kernel.cu)
target_include_directories(t2nvdr PRIVATE ${NV} ${CMAKE_SOURCE_DIR}/third_party/shim)
target_compile_options(t2nvdr PRIVATE -include nvdr_check.h $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(t2nvdr PUBLIC CUDA::cudart CUDA::cuda_driver)

add_library(t2 STATIC src/core.cu src/gemm.cu src/ops.cu src/dit.cu src/conv.cu src/ssdec.cu src/dino.cu src/image.cu src/spconv.cu src/sdec.cu src/mesh.cu src/pipeline.cu src/rng.cpp src/glb.cu src/inpaint.cpp src/gltf.cpp)
target_include_directories(t2 PUBLIC src ${CUDNN_INC} ${CMAKE_SOURCE_DIR}/third_party/cudnn-frontend/include ${CMAKE_SOURCE_DIR}/third_party/stb)
target_compile_options(t2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-ffp-contract=off>)
set_source_files_properties(src/rng.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=fast")
set_source_files_properties(src/inpaint.cpp PROPERTIES COMPILE_OPTIONS "-O3;-ffp-contract=off")
target_include_directories(t2 PRIVATE ${CMAKE_SOURCE_DIR}/third_party/torch-cpu)
# libwebp v1.4.0 = the version bundled in the reference Pillow 11.0.0 wheel (GLB texture encoding)
foreach(o ANIM_UTILS CWEBP DWEBP GIF2WEBP IMG2WEBP VWEBP WEBPINFO WEBPMUX EXTRAS)
  set(WEBP_BUILD_${o} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(third_party/libwebp EXCLUDE_FROM_ALL)
target_compile_definitions(t2 PRIVATE T2_WEBP T2_FLEXGEMM_CACHE_DEFAULT="${CMAKE_SOURCE_DIR}/assets/flexgemm_autotune.json")
target_link_libraries(t2 PUBLIC webp libwebpmux)
target_include_directories(t2 PRIVATE ${CMAKE_SOURCE_DIR}/third_party/libwebp/src)
target_link_libraries(t2 PUBLIC t2flash t2fa3 t2mea t2cumesh t2flexgemm t2nvdr Threads::Threads CUDA::cudart CUDA::cublas CUDA::cublasLt ${CUDNN_LIB})

add_executable(parity apps/parity.cu)
target_link_libraries(parity t2)

add_executable(t2cli apps/t2.cu)
set_target_properties(t2cli PROPERTIES OUTPUT_NAME t2)
target_link_libraries(t2cli t2)
