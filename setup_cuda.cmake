message("## Setup NVIDIA CUDA for GPU(s) with CMake in setup_cuda.cmake...")
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message("-- Found NVIDIA cuda-compiler nvcc: " ${CMAKE_CUDA_COMPILER})
else()
  message(STATUS "No NVIDIA cuda-compiler found by CMake! Cannot compile without CUDA support. Try [nvidia-smi] and [nvcc --version] in a terminal to check for a usable GPU, NVIDIA Drivers, NVIDIA Toolkit, and CUDA.")
endif()
set(CUDA_FOUND ${CMAKE_CUDA_COMPILER})

#  --------- READ ME IMPORTANT --------- Justin Bonus
# Set GPU architecture(s) to compile code for. Very important!
# Use the TARGET_CUDA_ARCH appropriate for the Compute Capability of your GPU below
# To determine CC of your GPU, try 'nvidia-smi' in console to get GPU model, then Google search
# Recommended minimum of CC 6.1, some backward compatibility available
# You can compile for one GPU architecture efficiently or a wide-range less efficiently
# Just-in-time (JIT) compilation can affect software performance but is convenient.
# Learn the difference between -arch, -gencode, and -code to improve compile and run-times.
# Reference: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# https://stackoverflow.com/questions/48274058/how-do-i-set-cuda-architecture-to-compute-50-and-sm-50-from-cmake-3-10-version

# % --------- CHOSE ONE OF THE BELOW OPTIONS THAT FIT YOUR GPU --------- %
# % Here you will choose what GPU(s) that ClaymoreUW compiles for, e.g. CUDA GPU binaries (fast, GPU arch. specific) and/or JIT compilation (slow first-run that is then cached on disk as binaries, but forward-compatible and covers more systems).
# % ClaymoreUW does not support mixed GPUs, i.e. all GPUs should be the same architecture sm_XY and ideally the same model.
# % Note that something like sm_${native} or sm_${keyword_you_set} can automate this, - Justin Bonus
# set(TARGET_CUDA_ARCH -arch=sm_${native})

# --------- Dell G7 Laptop - GTX 1060 Max-Q GPU (Arch = Pascal CC = 6.1) - Univ. of Wash. - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_61,code=sm_61)
# set(TARGET_CUDA_ARCH -gencode arch=compute_61,code=sm_61)
# set(TARGET_CUDA_ARCH --gpu-architecture=compute_61 --gpu-code=compute_61,sm_61)
# set(TARGET_CUDA_ARCH -arch=sm_61)

# --------- Klone Hyak HPC - RTX 2080ti GPUs (Arch = Turing, CC = 7.5) - Univ. of Wash. - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_75,code=compute_75)
#set(TARGET_CUDA_ARCH -arch=sm_75)

# --------- Frontera TACC HPC - RTX Quadro 5000 GPUs (Arch = Turing, CC = 7.5) - Univ. of Texas Austin - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_75,code=compute_75)
#set(TARGET_CUDA_ARCH -arch=sm_75)

# ---------Lonestar6 TACC HPC - NVIDIA A100 40GB (Arch = Amper, CC = 8.0) - Univ. of Texas Austin - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_80,code=sm_80)
#set(TARGET_CUDA_ARCH -arch=sm_80)

# --------- Desktop PC - RTX 4060 ti 16GB GPU (Arch = Ada, CC = 8.9) - UW Seattle / UC Berkeley - Justin Bonus
# set(TARGET_CUDA_ARCH --gpu-architecture=compute_89 --gpu-code=compute_89,sm_89)
#set(TARGET_CUDA_ARCH -arch=sm_89)
set(TARGET_CUDA_ARCH "-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_75,code=sm_75;-gencode=arch=compute_86,code=sm_86;-gencode=arch=compute_89,code=[sm_89,compute_89]")

# --------- ACCESS ACES - NVIDIA H100 80GB (Arch = Hopper, CC = 9.0) - Texas A&M University - Justin Bonus
# set(TARGET_CUDA_ARCH --gpu-architecture=compute_90 --gpu-code=compute_90,sm_90)
# set(TARGET_CUDA_ARCH -arch=sm_90)

# % --------- END GPU TARGET ARCH OPTIONS --------- %


# Set CMake CUDA Architectures 
# The native keyword auto detects the GPU arch. XY value, won't work if compiling without active GPU (e.g. an HPC's non-GPU login node).
# I recommend setting CMAKE_CUDA_ARCHITECTURES manually as XY, which is the compute capability number without the decimal point.
# 1060 Max-Q = 61, RTX 2080 ti / Quadro 5000 = 75, A100 = 80, RTX 4060 ti = 89, H100 = 90
#set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_ARCHITECTURES 61-real;61-virtual;75-real;75-virtual;86-real;86-virtual;89-real;89-virtual)

# set(CMAKE_CUDA_ARCHITECTURES native)
message("-- NVIDIA CUDA GPU detected [" ${CMAKE_CUDA_ARCHITECTURES} ", ${TARGET_CUDA_ARCH}]\t compute-capability number and architecture label with CMake.")



# --------- Set NVCC compiler flags ---------
# NOTE: I removed compiler flags, e.g. --use_fast_math, to improve accuracy - Justin Bonus
# Reference: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
#     get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
#     if(NOT "${old_flags}" STREQUAL "")
#         string(REPLACE ";" "," CUDA_flags "${old_flags}")
#         set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
#             "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
#             )
#     endif()
# endfunction()

# reference: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
	get_property(old_flags
		TARGET ${EXISTING_TARGET}
		PROPERTY INTERFACE_COMPILE_OPTIONS
	)
	if(NOT "${old_flags}" STREQUAL "")
		string(REPLACE ";" "," CUDA_flags "${old_flags}")
		set_property(
			TARGET ${EXISTING_TARGET}
			PROPERTY INTERFACE_COMPILE_OPTIONS
			"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
		)
	endif()
endfunction()

# Suggested by Destranix May 2023 on upstream Claymore branch
#Cmake does not handle linking correctly for separable compilation: https://gitlab.kitware.com/cmake/cmake/-/issues/22788
function(GET_DEVICE_LINK_PATH TARGET_NAME ret)
	cmake_path(SET DEVICE_LINK_PATH ${CMAKE_BINARY_DIR})
	cmake_path(APPEND DEVICE_LINK_PATH "CMakeFiles")
	cmake_path(APPEND DEVICE_LINK_PATH ${TARGET_NAME}.dir)
	cmake_path(APPEND DEVICE_LINK_PATH ${CMAKE_BUILD_TYPE})
	cmake_path(APPEND DEVICE_LINK_PATH "cmake_device_link.obj")
	set(${ret} ${DEVICE_LINK_PATH} PARENT_SCOPE)
endfunction()

# --------- Set NVCC compiler flags ---------
function(add_cuda_executable binary)
  if(CUDA_FOUND)
    message("-- [${binary}]\tNVIDIA CUDA executable build config started...")
    add_executable(${binary} ${ARGN})
    # seems not working

    # target_compile_options(${binary} 
    #   PRIVATE     $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --fmad=true -lineinfo --ptxas-options=-allow-expensive-optimizations=true --maxrregcount=96> 
    #   # disabled fast math flag because it can cause numerical errors for engineering
    #   # --maxrregcount=128
    # )

    # -ccbin=icc

		# Debug flags <- Slower to compile / run but more debugging info
		target_compile_options(${binary} 
			PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --resource-usage -Xptxas -v --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --fmad=true -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
		)

    # Release flags <- Faster to compile / run but less info for debugging
		target_compile_options(${binary} 
			PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --fmad=true -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
		)

    target_compile_features(${binary} PRIVATE cuda_std_14)
    set_target_properties(${binary}
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION OFF
                  #LINKER_LANGUAGE CUDA
                  # RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Destranix
		GET_DEVICE_LINK_PATH(${binary} DEVICE_LINK_PATH)
    message("-- [${DEVICE_LINK_PATH}]\tNVIDIA CUDA DEVICE_LINK_PATH")

    target_link_libraries(${binary}
        PRIVATE mncuda
    )

    # --- WINDOWS ONLY---
    # May help you if you have default library linking problems with MSVC on Windows
    # https://stackoverflow.com/questions/11512795/ignoring-unknown-option-nodefaultliblibcmtd
		# https://learn.microsoft.com/en-us/cpp/build/reference/nodefaultlib-ignore-libraries?view=msvc-170
    # https://learn.microsoft.com/en-us/cpp/build/reference/md-mt-ld-use-run-time-library?view=msvc-170&source=recommendations
    # target_link_options(${binary}
		# 	PRIVATE /NODEFAULTLIB:libcmt.lib
		# )
    # --- 

    # We are telling CMake to install the executable in the build directory, e.g. build/Projects/EXAMPLE/example. This is fine for development, but not for production. We need to tell CMake to install the executable in the bin directory of the project, e.g. bin/Projects/EXAMPLE/example. We can do this by adding the following line to the CMakeLists.txt file:
    # install(TARGETS ${binary} DESTINATION bin)
    install(TARGETS ${binary}  )

    message("-- [${binary}]\tNVIDIA CUDA executable build config finished.")
  endif()
endfunction(add_cuda_executable)

# --------- Set NVCC compiler flags ---------
function(add_cuda_library library)
  if(CUDA_FOUND)
  message("-- [${library}]\tNVIDIA CUDA library build config started...")
    add_library(${library} ${ARGN})
    # Seems to not be working
    # target_compile_options(${library} 
    #   PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --fmad=true --ptxas-options=-allow-expensive-optimizations=true --maxrregcount=96> #--maxrregcount=128
    # )

    # -ccbin=icc

		# Debug flags <- Slower to compile / run but more debugging info
    target_compile_options(${library} 
		PUBLIC        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --resource-usage -Xptxas -v --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --fmad=true --ptxas-options=-allow-expensive-optimizations=true>
	  )

    # Release flags <- Faster to compile / run but less info for debugging
    target_compile_options(${library} 
    PUBLIC        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --fmad=true --ptxas-options=-allow-expensive-optimizations=true>
    )


  # target_link_options(${library} 
  #    PRIVATE       $<$<LINKER_LANGUAGE:CUDA>: ${TARGET_CUDA_ARCH}>
  #   )
    target_compile_features(${library} PRIVATE cuda_std_14)
    set_target_properties(${library}
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION OFF # CUDA is buggy with separable compilation of files
                  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
                  POSITION_INDEPENDENT_CODE ON # CUDA can be a bit weird with position independent code
                  #LINKER_LANGUAGE CUDA # Pretty sure this is not needed
    )
    target_compile_definitions(${library} 
      PUBLIC        CMAKE_GENERATOR_PLATFORM=x64 # Assuming you are using x64 system (common). May not work on experimental HPC systems with fancy hardware
    )
    message("-- [${library}]\tNVIDIA CUDA library build config finished.")
  endif()
endfunction(add_cuda_library)
