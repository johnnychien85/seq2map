cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Build Option(s)
set(MONOLITHIC TRUE  CACHE BOOL "Build big standalone executables statically linked against thirt-party libraries")
set(WITH_EBUS  FALSE CACHE BOOL "Build the grabber with the support of Pleora's cameras")
set(WITH_PYLON FALSE CACHE BOOL "Build the grabber with the support of Basler's cameras")

# Basler's Pylon library
if(WITH_EBUS)
    set(eBUS_INCLUDE_DIR ${eBUS_INCLUDE_DIR} CACHE PATH "The directory containing eBUS header files." FORCE)
    set(eBUS_LIBRARY_DIR ${eBUS_LIBRARY_DIR} CACHE PATH "The directory containing eBUS libraries."    FORCE)
    
    if(NOT EXISTS "${eBUS_INCLUDE_DIR}/PvBase.h")
        message(SEND_ERROR "eBUS includes missing")
    elseif(NOT EXISTS "${eBUS_LIBRARY_DIR}")
        message(SEND_ERROR "eBUS library folder missing")
    else()
        include_directories(${eBUS_INCLUDE_DIR})
        link_directories   (${eBUS_LIBRARY_DIR})
        add_definitions    (-DWITH_EBUS)
        add_definitions    (-DPV_GUI_NOT_AVAILABLE) # disable GUI functions
    endif()
endif()

# Basler's Pylon library
if(WITH_PYLON)
    set(Pylon_INCLUDE_DIR ${Pylon_INCLUDE_DIR} CACHE PATH "The directory containing Pylon header files." FORCE)
    set(Pylon_LIBRARY_DIR ${Pylon_LIBRARY_DIR} CACHE PATH "The directory containing Pylon libraries."    FORCE)
    
    if(NOT EXISTS "${Pylon_INCLUDE_DIR}/pylon/PylonIncludes.h")
        message(SEND_ERROR "Pylon includes missing")
    elseif(NOT EXISTS "${Pylon_LIBRARY_DIR}")
        message(SEND_ERROR "Pylon library folder missing")
    else()
        include_directories(${Pylon_INCLUDE_DIR})
        link_directories   (${Pylon_LIBRARY_DIR})
        add_definitions    (-DWITH_PYLON)
    endif()
endif()

# Executables Configuration
project(seq2map CXX)
add_library   (common      includes/seq2map/common.hpp           # Shared components
                           includes/seq2map/disparity.hpp        # 
                           includes/seq2map/features.hpp         # 
                           includes/seq2map/geometry.hpp         # 
                           includes/seq2map/sequence.hpp         # 
                           includes/seq2map/solve.hpp            # 
                           sources/common/common.cpp             # 
                           sources/common/disparity.cpp          # 
                           sources/common/features.cpp           # 
                           sources/common/geometry.cpp           # 
                           sources/common/sequence.cpp           # 
                           sources/common/solve.cpp)             # 
add_executable(calibn      sources/calibn/args.hpp               # Multi-camera calibration utility
                           sources/calibn/calibgraph.hpp         # 
                           sources/calibn/calibgraphbuilder.hpp  # 
                           sources/calibn/calibgraphbundler.hpp  # 
                           sources/calibn/args.cpp               # 
                           sources/calibn/calibn.cpp             # 
                           sources/calibn/calibgraph.cpp         # 
                           sources/calibn/calibgraphbuilder.cpp  # 
                           sources/calibn/calibgraphbundler.cpp) # 
add_executable(im2dpmap    sources/im2dpmap/im2dpmap.cpp)        # Stereo matcher-based disparity map computatiion
add_executable(im2flow     sources/im2flow/im2flow.cpp)          # Optical flow computation
add_executable(im2features sources/im2features/im2features.cpp)  # Image feature detection and extraction
add_executable(raw2seq     sources/raw2seq/scanner.hpp           # Process raw data to the structured sequence profile
                           sources/raw2seq/scanner.cpp           # 
                           sources/raw2seq/raw2seq.cpp)          # 
add_executable(monovo      sources/monovo/monovo.cpp)            # Monocular visual odometry
add_executable(vggrab      sources/vggrab/recorder.hpp           # Software synchronised multi-camera frame grabber
                           sources/vggrab/recorder.cpp           # 
                           sources/vggrab/grabber.hpp            # 
                           sources/vggrab/grabber.cpp            # 
                           sources/vggrab/grabber_dummy_impl.hpp # 
                           sources/vggrab/grabber_ebus_impl.hpp  # 
                           sources/vggrab/grabber_pylon_impl.hpp # 
                           sources/vggrab/storage.hpp            # 
                           sources/vggrab/storage.cpp            # 
                           sources/vggrab/syncbuf.hpp            # 
                           sources/vggrab/syncbuf.cpp            # 
                           sources/vggrab/uirenderer.hpp         # 
                           sources/vggrab/uirenderer.cpp         # 
                           sources/vggrab/vggrab.cpp)            # 
add_executable(dpmap2pts   sources/miscs/dpmap2pts.cpp)          # Converion from disparity map to point clouds
add_executable(imrect      sources/miscs/imrect.cpp)             # Image rectification utility
add_executable(imcvt       sources/miscs/imcvt.cpp)              # Image conversion utility
add_executable(seqchk      sources/miscs/seqchk.cpp)             # Sequence file examining utility

target_link_libraries(calibn      common)
target_link_libraries(im2dpmap    common)
target_link_libraries(im2features common)
target_link_libraries(im2flow     common)
target_link_libraries(raw2seq     common)
target_link_libraries(monovo      common)
target_link_libraries(vggrab      common)
target_link_libraries(dpmap2pts   common)
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

if(WITH_EBUS AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(vggrab "PvBase" "PvDevice" "PvBuffer" "PvGenICam" "PvStream")
endif()

# Thirty-party Libraries
mark_as_advanced(FORCE OPENCV_FOUND OpenCV_FOUND)
find_package(OpenCV REQUIRED COMPONENTS core highgui calib3d features2d)
find_package(OpenCV COMPONENTS xfeatures2d)

if(OpenCV_VERSION AND NOT OpenCV_VERSION_MAJOR LESS 3)
	include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(calibn      ${OpenCV_LIBS})
	target_link_libraries(im2dpmap    ${OpenCV_LIBS})
	target_link_libraries(im2features ${OpenCV_LIBS})
	target_link_libraries(im2flow     ${OpenCV_LIBS})
	target_link_libraries(raw2seq     ${OpenCV_LIBS})
	target_link_libraries(monovo      ${OpenCV_LIBS})
	target_link_libraries(vggrab      ${OpenCV_LIBS})
	target_link_libraries(dpmap2pts   ${OpenCV_LIBS})
	target_link_libraries(imrect      ${OpenCV_LIBS})
	target_link_libraries(imcvt       ${OpenCV_LIBS})
	target_link_libraries(seqchk      ${OpenCV_LIBS})

    if(OPENCV_XFEATURES2D_FOUND)
        add_definitions(-DWITH_XFEATURES2D)
    else()
        message(STATUS "extended features2d module not found")
        message(STATUS "it might be located in the build of opencv-contrib")
    endif()
else()
	message(SEND_ERROR "OpenCV not found")
endif()

find_package(Boost REQUIRED COMPONENTS log_setup log filesystem thread timer date_time chrono program_options system)
mark_as_advanced(CLEAR BOOST_INCLUDEDIR BOOST_LIBRARYDIR)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(calibn      ${Boost_LIBRARIES})
	target_link_libraries(im2dpmap    ${Boost_LIBRARIES})
	target_link_libraries(im2features ${Boost_LIBRARIES})
	target_link_libraries(im2flow     ${Boost_LIBRARIES})
	target_link_libraries(raw2seq     ${Boost_LIBRARIES})
	target_link_libraries(monovo      ${Boost_LIBRARIES})
	target_link_libraries(vggrab      ${Boost_LIBRARIES})
	target_link_libraries(dpmap2pts   ${Boost_LIBRARIES})
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
