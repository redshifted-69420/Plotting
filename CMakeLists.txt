cmake_minimum_required(VERSION 3.30)
project(Plot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(PNG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Threads REQUIRED)
find_package(Freetype REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)

add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)

# Find Pango and Cairo
pkg_check_modules(PANGO REQUIRED pango)
pkg_check_modules(CAIRO REQUIRED cairo)

# Set include and library directories explicitly
set(PANGO_INCLUDE_DIRS "/opt/homebrew/include/pango-1.0")
set(PANGO_LIBRARY_DIRS "/opt/homebrew/lib")
set(PANGO_LIBRARIES "-lpango-1.0 -lgobject-2.0 -lglib-2.0 -lintl -lharfbuzz")

set(CAIRO_INCLUDE_DIRS "/opt/homebrew/include/cairo")
set(CAIRO_LIBRARY_DIRS "/opt/homebrew/lib")
set(CAIRO_LIBRARIES "-lcairo")

set(GLIB_INCLUDE_DIRS "/opt/homebrew/include/glib-2.0" "/opt/homebrew/lib/glib-2.0/include")
set(GLIB_LIBRARY_DIRS "/opt/homebrew/lib")
set(GLIB_LIBRARIES "-lglib-2.0 -lgobject-2.0 -lintl")

include_directories(${GLIB_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS})

set(HARFBUZZ_INCLUDE_DIRS "/opt/homebrew/include/harfbuzz")
set(HARFBUZZ_LIBRARY_DIRS "/opt/homebrew/lib")
set(HARFBUZZ_LIBRARIES "-lharfbuzz")

include_directories(${HARFBUZZ_INCLUDE_DIRS})
link_directories(${HARFBUZZ_LIBRARY_DIRS})

set(OPENSSL_ROOT_DIR /opt/homebrew/opt/openssl@3)
set(OPENSSL_INCLUDE_DIR /opt/homebrew/opt/openssl@3/include)
set(OPENSSL_LIBRARIES /opt/homebrew/opt/openssl@3/lib)

find_package(OpenSSL REQUIRED)

# Debugging output to verify detection
message(STATUS "Pango include directories: ${PANGO_INCLUDE_DIRS}")
message(STATUS "Pango library directories: ${PANGO_LIBRARY_DIRS}")
message(STATUS "Cairo include directories: ${CAIRO_INCLUDE_DIRS}")
message(STATUS "Cairo library directories: ${CAIRO_LIBRARY_DIRS}")

# Add executable
add_executable(Plot
    src/main.cpp
    src/MathLib/mathlib.cpp
    src/PlotLib/graph.cpp
    src/MathLib/Matrix.mm
)

# Include directories
target_include_directories(Plot PRIVATE
    ${PNG_INCLUDE_DIRS}
    ${FREETYPE_INCLUDE_DIRS}
    ${CAIRO_INCLUDE_DIRS}
    ${PANGO_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# Link directories
link_directories(
    ${PANGO_LIBRARY_DIRS}
    ${CAIRO_LIBRARY_DIRS}
)

# Link libraries
target_link_libraries(Plot PRIVATE
    Boost::filesystem
    PNG::PNG
    Freetype::Freetype
    ZLIB::ZLIB
    Threads::Threads
    ${CAIRO_LIBRARIES}
    ${PANGO_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${HARFBUZZ_LIBRARIES}
    ${OPENSSL_LIBRARIES} OpenSSL::Crypto
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
)

# Add Accelerate framework for macOS arm64
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    target_link_libraries(Plot PRIVATE ${ACCELERATE_FRAMEWORK})
  else()
    message(FATAL_ERROR "Accelerate framework not found")
  endif()
endif()

# Set output directory
set_target_properties(Plot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Compile the Metal shader into a library
if(APPLE)
    # Path to the Metal shader file
    set(METAL_SHADER_SOURCE "${CMAKE_SOURCE_DIR}/src/MathLib/matrix_add.metal")

    # Output files for the Metal shader compilation
    set(METAL_SHADER_AIR "${CMAKE_BINARY_DIR}/matrix_add.air")
    set(METAL_SHADER_METALLIB "${CMAKE_BINARY_DIR}/default.metallib")

    # Compile the Metal shader into an AIR file
    add_custom_command(
        OUTPUT ${METAL_SHADER_AIR}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SOURCE} -o ${METAL_SHADER_AIR}
        DEPENDS ${METAL_SHADER_SOURCE}
        COMMENT "Compiling Metal shader to AIR"
    )

    # Link the AIR file into a Metal library
    add_custom_command(
        OUTPUT ${METAL_SHADER_METALLIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_SHADER_AIR} -o ${METAL_SHADER_METALLIB}
        DEPENDS ${METAL_SHADER_AIR}
        COMMENT "Linking Metal shader into a library"
    )

    # Add the Metal library as a dependency to the Plot target
    add_custom_target(MetalShader ALL DEPENDS ${METAL_SHADER_METALLIB})
    add_dependencies(Plot MetalShader)

    # Copy the Metal library to the output directory
    add_custom_command(
        TARGET Plot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${METAL_SHADER_METALLIB} ${CMAKE_BINARY_DIR}/bin/default.metallib
        COMMENT "Copying Metal library to output directory"
    )
endif()