cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Build Option(s)
set(MONOLITHIC  TRUE  CACHE BOOL "Build big standalone executables statically linked against thirt-party libraries")
set(WITH_EBUS   FALSE CACHE BOOL "Build the grabber with the support of Pleora's cameras")
set(WITH_PYLON  FALSE CACHE BOOL "Build the grabber with the support of Basler's cameras")
set(WITH_STCAM  FALSE CACHE BOOL "Build the grabber with the support of Sentech's cameras")
set(BUILD_DOCS  FALSE CACHE BOOL "Build documentation (requires Doxygen)")
set(BUILD_TESTS FALSE CACHE BOOL "Build unit tests")

# The project version number.
set(VERSION_MAJOR 0 CACHE STRING "Project major version number.")
set(VERSION_MINOR 0 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 1 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

# Pleora's eBUS library
if(WITH_EBUS)
    set(eBUS_INCLUDE_DIR ${eBUS_INCLUDE_DIR} CACHE PATH "The directory containing eBUS header files.")
    set(eBUS_LIBRARY_DIR ${eBUS_LIBRARY_DIR} CACHE PATH "The directory containing eBUS libraries."   )
    
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
    set(Pylon_INCLUDE_DIR ${Pylon_INCLUDE_DIR} CACHE PATH "The directory containing Pylon header files.")
    set(Pylon_LIBRARY_DIR ${Pylon_LIBRARY_DIR} CACHE PATH "The directory containing Pylon libraries."   )
    
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

# SenTech's StCamUSB library
if(WITH_STCAM)
    set(StCam_INCLUDE_DIR ${StCam_INCLUDE_DIR} CACHE PATH "The directory containing StCamUSB header files.")
    set(StCam_LIBRARY_DIR ${StCam_LIBRARY_DIR} CACHE PATH "The directory containing StCamUSB library."     )

    if(NOT EXISTS "${StCam_INCLUDE_DIR}/StCamD.h")
        message(SEND_ERROR "StCamUSB includes missing")
    elseif(NOT EXISTS "${StCam_LIBRARY_DIR}")
        message(SEND_ERROR "StCamUSB library folder missing")
    else()
        include_directories(${StCam_INCLUDE_DIR})
        link_directories   (${StCam_LIBRARY_DIR})
        add_definitions    (-DWITH_STCAM)
    endif()
endif()

# Executables Configuration
project(seq2map CXX)
add_library   (base        includes/seq2map/common.hpp           # Shared components
                           includes/seq2map/common_impl.hpp      # 
                           includes/seq2map/disparity.hpp        # 
                           includes/seq2map/features.hpp         # 
                           includes/seq2map/features_opencv.hpp  # 
                           includes/seq2map/geometry.hpp         # 
                           includes/seq2map/geometry_problems.hpp# 
                           includes/seq2map/mapping.hpp          # 
                           includes/seq2map/seq_file_store.hpp   # 
                           includes/seq2map/sequence.hpp         # 
                           includes/seq2map/solve.hpp            # 
                           includes/seq2map/sparse_node.hpp      # 
                           sources/base/common.cpp               # 
                           sources/base/disparity.cpp            # 
                           sources/base/features.cpp             # 
                           sources/base/features_opencv.cpp      #
                           sources/base/geometry.cpp             # 
                           sources/base/geometry_problems.cpp    # 
                           sources/base/mapping.cpp              # 
                           sources/base/sequence.cpp             # 
                           sources/base/solve.cpp               )# 
add_executable(calibn      sources/calibn/args.hpp               # Multi-camera calibration utility
                           sources/calibn/calibgraph.hpp         # 
                           sources/calibn/calibgraphbuilder.hpp  # 
                           sources/calibn/calibgraphbundler.hpp  # 
                           sources/calibn/args.cpp               # 
                           sources/calibn/calibn.cpp             # 
                           sources/calibn/calibgraph.cpp         # 
                           sources/calibn/calibgraph_report.cpp  # 
                           sources/calibn/calibgraphbuilder.cpp  # 
                           sources/calibn/calibgraphbundler.cpp )# 
add_executable(im2disp     sources/im2disp/im2disp.cpp          )# Stereo matcher-based disparity map computation
add_executable(im2flow     sources/im2flow/im2flow.cpp          )# Optical flow computation
add_executable(im2kpts     sources/im2kpts/im2kpts.cpp          )# Image feature detection and extraction
add_executable(raw2seq     sources/raw2seq/builder.hpp           # Process raw data to build structured sequence profiles
                           sources/raw2seq/builder.cpp           # 
                           sources/raw2seq/raw2seq.cpp          )# 
add_executable(seq2map     sources/seq2map/seq2map.cpp          )# Motion recovery and mapping
add_executable(vggrab      sources/vggrab/recorder.hpp           # Software synchronised multi-camera frame grabber
                           sources/vggrab/recorder.cpp           # 
                           sources/vggrab/grabber.hpp            # 
                           sources/vggrab/grabber.cpp            # 
                           sources/vggrab/grabber_dummy_impl.hpp # 
                           sources/vggrab/grabber_ebus_impl.hpp  # 
                           sources/vggrab/grabber_pylon_impl.hpp # 
                           sources/vggrab/grabber_stcam_impl.hpp # 
                           sources/vggrab/storage.hpp            # 
                           sources/vggrab/storage.cpp            # 
                           sources/vggrab/syncbuf.hpp            # 
                           sources/vggrab/syncbuf.cpp            # 
                           sources/vggrab/uirenderer.hpp         # 
                           sources/vggrab/uirenderer.cpp         # 
                           sources/vggrab/vggrab.cpp            )# 
add_executable(chkseq      sources/miscs/chkseq.cpp             )# Sequence file examining utility
add_executable(disp2pts    sources/miscs/disp2pts.cpp           )# Converion from disparity map to point cloud
add_executable(imcvt       sources/miscs/imcvt.cpp              )# Image conversion utility
add_executable(imfuse      sources/miscs/imfuse.cpp             )# Multi-camera image fusion utility
add_executable(imrect      sources/miscs/imrect.cpp             )# Image rectification utility

set_target_properties(chkseq   PROPERTIES FOLDER "miscs")
set_target_properties(disp2pts PROPERTIES FOLDER "miscs")
set_target_properties(imcvt    PROPERTIES FOLDER "miscs")
set_target_properties(imfuse   PROPERTIES FOLDER "miscs")
set_target_properties(imrect   PROPERTIES FOLDER "miscs")

target_link_libraries(calibn   base)
target_link_libraries(im2kpts  base)
target_link_libraries(im2disp  base)
target_link_libraries(im2flow  base)
target_link_libraries(raw2seq  base)
target_link_libraries(seq2map  base)
target_link_libraries(vggrab   base)
target_link_libraries(disp2pts base)
target_link_libraries(imcvt    base)
target_link_libraries(imfuse   base)
target_link_libraries(imrect   base)
target_link_libraries(chkseq   base)

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

if(WITH_STCAM)
    target_link_libraries(vggrab "StCamD")
endif()

# Thirty-party Libraries
mark_as_advanced(FORCE OPENCV_FOUND OpenCV_FOUND)
find_package(OpenCV REQUIRED COMPONENTS core highgui calib3d features2d)
find_package(OpenCV COMPONENTS xfeatures2d cudastereo cudafeatures2d)

if(OpenCV_VERSION AND NOT OpenCV_VERSION_MAJOR LESS 3)
	include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(calibn   ${OpenCV_LIBS})
	target_link_libraries(im2disp  ${OpenCV_LIBS})
	target_link_libraries(im2kpts  ${OpenCV_LIBS})
	target_link_libraries(im2flow  ${OpenCV_LIBS})
	target_link_libraries(raw2seq  ${OpenCV_LIBS})
    target_link_libraries(seq2map  ${OpenCV_LIBS})
	target_link_libraries(vggrab   ${OpenCV_LIBS})
	target_link_libraries(disp2pts ${OpenCV_LIBS})
	target_link_libraries(imcvt    ${OpenCV_LIBS})
	target_link_libraries(imfuse   ${OpenCV_LIBS})
	target_link_libraries(imrect   ${OpenCV_LIBS})
	target_link_libraries(chkseq   ${OpenCV_LIBS})

    if(OPENCV_XFEATURES2D_FOUND)
        add_definitions(-DWITH_XFEATURES2D)
    else()
        message(STATUS "extended features2d module not found")
        message(STATUS "it might be located in the build of opencv-contrib")
    endif()
else()
	message(SEND_ERROR "OpenCV not found")
endif()

if(BUILD_TESTS)
    find_package(Boost REQUIRED COMPONENTS log_setup log filesystem thread timer date_time chrono program_options system unit_test_framework)
else()
    find_package(Boost REQUIRED COMPONENTS log_setup log filesystem thread timer date_time chrono program_options system)
endif()

mark_as_advanced(CLEAR BOOST_INCLUDEDIR BOOST_LIBRARYDIR)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(calibn      ${Boost_LIBRARIES})
	target_link_libraries(im2kpts     ${Boost_LIBRARIES})
	target_link_libraries(im2disp     ${Boost_LIBRARIES})
	target_link_libraries(im2flow     ${Boost_LIBRARIES})
	target_link_libraries(raw2seq     ${Boost_LIBRARIES})
	target_link_libraries(seq2map     ${Boost_LIBRARIES})
	target_link_libraries(vggrab      ${Boost_LIBRARIES})
	target_link_libraries(disp2pts    ${Boost_LIBRARIES})
	target_link_libraries(imcvt       ${Boost_LIBRARIES})
	target_link_libraries(imfuse      ${Boost_LIBRARIES})
	target_link_libraries(imrect      ${Boost_LIBRARIES})
	target_link_libraries(chkseq      ${Boost_LIBRARIES})
else()
	set(BOOST_INCLUDEDIR ${Boost_INCLUDE_DIRS} CACHE PATH "The directory containing boost headers folder 'boost'." FORCE)
	set(BOOST_LIBRARYDIR ${Boost_LIBRARY_DIRS} CACHE PATH "The directory containing boost libraries." FORCE)
	message(SEND_ERROR "boost not found")
endif()

# Unit test
if(BUILD_TESTS)
    enable_testing()
    
    set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "The directory containing test data")
    file(GLOB TEST_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} sources/tests/*.cpp)
    
    foreach(src ${TEST_SOURCES})
        get_filename_component(test ${src} NAME_WE)
        add_executable(${test} ${src})
        set_target_properties(${test} PROPERTIES FOLDER "tests")
        target_link_libraries(${test} base ${OpenCV_LIBS} ${Boost_LIBRARIES})
        add_test(
            NAME ${test}
            WORKING_DIRECTORY ${TEST_DATA_DIR}
            COMMAND ${test}
        )
        message(STATUS "Test added: ${test}")
    endforeach(src)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen)
    
    if(NOT DOXYGEN_FOUND)
        message(SEND_ERROR "Doxygen is required to build documentation")
    endif()
    
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYFILE    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYFILE_IN} ${DOXYFILE} @ONLY)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION docs)
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
