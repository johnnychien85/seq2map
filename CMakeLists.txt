cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Build Option(s)
set(MONOLITHIC TRUE CACHE BOOL "Build big standalone executables statically linked against thirt-party libraries")

# Executables Configuration
project(seq2map CXX)
add_library   (common      includes/seq2map/common.hpp includes/seq2map/disparity.hpp includes/seq2map/features.hpp includes/seq2map/geometry.hpp includes/seq2map/sequence.hpp includes/seq2map/solve.hpp sources/common/common.cpp sources/common/disparity.cpp sources/common/features.cpp sources/common/geometry.cpp sources/common/sequence.cpp sources/common/solve.cpp)
add_executable(im2dispmap  sources/im2dispmap/im2dispmap.cpp)   # stereo matcher-based disparity map computatiion
add_executable(im2flow     sources/im2flow/im2flow.cpp)         # optical flow computation
add_executable(im2features sources/im2features/im2features.cpp) # image feature detection and extraction
add_executable(raw2seq     sources/raw2seq/scanner.hpp sources/raw2seq/scanner.cpp sources/raw2seq/raw2seq.cpp) # structured raw data folder to sequence profile
add_executable(monovo      sources/monovo/monovo.cpp)     # monocular visual odometry
add_executable(dispmap2pts sources/miscs/dispmap2pts.cpp) # convert a disparity map to a point cloud
add_executable(imrect      sources/miscs/imrect.cpp)      # image rectification utility
add_executable(imcvt       sources/miscs/imcvt.cpp)       # image conversion utility
add_executable(seqchk      sources/miscs/seqchk.cpp)      # sequence file examining utility

target_link_libraries(im2dispmap  common)
target_link_libraries(im2features common)
target_link_libraries(im2flow     common)
target_link_libraries(raw2seq     common)
target_link_libraries(monovo      common)
target_link_libraries(dispmap2pts common)
target_link_libraries(imrect      common)
target_link_libraries(imcvt       common)
target_link_libraries(seqchk      common)

include_directories(includes)

if(MONOLITHIC)
	set(BUILD_SHARED_LIBS OFF)
	set(Boost_USE_STATIC_LIBS ON)
	set(Boost_USE_STATIC_RUNTIME ON)
else()
	set(BUILD_SHARED_LIBS ON)
	set(Boost_USE_STATIC_LIBS OFF)
	set(Boost_USE_STATIC_RUNTIME OFF)
endif()

# Thirty-party Libraries
mark_as_advanced(FORCE OPENCV_FOUND OpenCV_FOUND)
find_package(OpenCV REQUIRED core highgui calib3d features2d xfeatures2d)

if(OpenCV_VERSION AND NOT OpenCV_VERSION_MAJOR LESS 3)
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(im2dispmap  ${OpenCV_LIBS})
	target_link_libraries(im2features ${OpenCV_LIBS})
	target_link_libraries(im2flow     ${OpenCV_LIBS})
	target_link_libraries(raw2seq     ${OpenCV_LIBS})
	target_link_libraries(monovo      ${OpenCV_LIBS})
	target_link_libraries(dispmap2pts ${OpenCV_LIBS})
	target_link_libraries(imrect      ${OpenCV_LIBS})
	target_link_libraries(imcvt       ${OpenCV_LIBS})
	target_link_libraries(seqchk      ${OpenCV_LIBS})
else()
	message(SEND_ERROR "OpenCV not found")
endif()

find_package(Boost REQUIRED COMPONENTS chrono date_time filesystem thread timer log log_setup program_options system)
mark_as_advanced(CLEAR BOOST_INCLUDEDIR BOOST_LIBRARYDIR)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	target_link_libraries(im2dispmap  ${Boost_LIBRARIES})
	target_link_libraries(im2features ${Boost_LIBRARIES})
	target_link_libraries(im2flow     ${Boost_LIBRARIES})
	target_link_libraries(raw2seq     ${Boost_LIBRARIES})
	target_link_libraries(monovo      ${Boost_LIBRARIES})
	target_link_libraries(dispmap2pts ${Boost_LIBRARIES})
	target_link_libraries(imrect      ${Boost_LIBRARIES})
	target_link_libraries(imcvt       ${Boost_LIBRARIES})
	target_link_libraries(seqchk      ${Boost_LIBRARIES})
else()
	set(BOOST_INCLUDEDIR ${Boost_INCLUDE_DIRS} CACHE PATH "The directory containing boost headers folder 'boost'." FORCE)
	set(BOOST_LIBRARYDIR ${Boost_LIBRARY_DIRS} CACHE PATH "The directory containing boost libraries." FORCE)
	message(SEND_ERROR "boost not found")
endif()

# MUST use static standard c runtime for monolithic build
if(MONOLITHIC AND MSVC)
	foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
		if(${flag_var} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		endif(${flag_var} MATCHES "/MD")
		if(${flag_var} MATCHES "/MDd")
			string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
		endif(${flag_var} MATCHES "/MDd")
	endforeach(flag_var)
endif()
