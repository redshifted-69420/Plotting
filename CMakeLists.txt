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
    src/MathLib/Matrix.cpp
)

# Include directories
target_include_directories(Plot PRIVATE
    ${PNG_INCLUDE_DIRS}
    ${FREETYPE_INCLUDE_DIRS}
    ${CAIRO_INCLUDE_DIRS}
    ${PANGO_INCLUDE_DIRS}
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
)


# Set output directory
set_target_properties(Plot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
