if (CUDA_FOUND)

add_cuda_executable(osu_lwf)
target_sources(osu_lwf
    PRIVATE     osu_lwf.cu
)
# GCC versions less than 8.1, e.g. version 7.5, did not fully support C++ 17 standard. Because of this we have to use <experimental/filesystem> and link to stdc++fs here instead of just using <filesystem>. Required for parsing JSON inputs in Claymore currently.
# NOTE: GCC may want -lstdc++fs flag passed in as a CMAKE_CXX_FLAGS compiler flag to link C++17 filesystem implementation. MINGW may not like this.
# e.g. something like: g++-7 -std=c++14 -lstdc++fs  -c main.cpp -o main.o
# e.g. something like: g++-7 -o main main.o /usr/lib/gcc/x86_64-linux-gnu/7/libstdc++fs.a
# if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MINGW) 
# if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.1)
# 	target_link_libraries(fbar
# 	    PRIVATE     mnio
# 		        cxxopts
# 		        spdlog
# 		        fmt
# 		        rapidjson
# 		        range-v3
# 		        filesystem
# 		        stdc++fs # Adds C++ <experimental/filsystem> for some compilers, e.g. GCC/G++ 7.5. This flag must be last to work
# 	)
# else()
# 	target_link_libraries(fbar
# 	    PRIVATE     mnio
# 		        cxxopts
# 		        spdlog
# 		        fmt
# 		        rapidjson
# 		        range-v3
# 		        filesystem
# 		        # stdc++fs
# 	)
# endif()
# else()
# 	target_link_libraries(fbar
# 	    PRIVATE     mnio
# 		        cxxopts
# 		        spdlog
# 		        fmt
# 		        rapidjson
# 		        range-v3
# 		        filesystem
# 		        # stdc++fs
# 	)
# endif()
target_link_libraries(osu_lwf
	PRIVATE     mnio
			cxxopts
			spdlog
			fmt
			rapidjson
			range-v3
			filesystem
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(osu_lwf PRIVATE stdc++fs)  # only GCC 7/8 need this
  endif()
endif()

endif()
