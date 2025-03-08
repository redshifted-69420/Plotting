# Plot Library

A modern C++ plotting library for creating high-quality scientific visualizations with SVG and PNG export capabilities.

## Overview

Plot Library is a powerful and flexible graphing toolkit that allows you to create publication-quality plots and visualizations. Built with performance in mind, it features a robust mathematical foundation, multi-threading support, and advanced rendering capabilities including LaTeX support.

## Features

- **High-Quality Rendering**: Create beautiful graphs with anti-aliased lines, custom colors, and various plot styles
- **Multiple Export Formats**: Save your visualizations as SVG or PNG files
- **Mathematical Foundation**: Built on a comprehensive math library with matrix operations and statistical functions
- **Vector Graphics Support**: First-class SVG output for scalable, publication-ready figures
- **Customizable Styling**: Control colors, line widths, markers, grid properties, and more
- **Annotation Support**: Add LaTeX annotations to your plots for mathematical expressions
- **Multi-Threading**: Thread pool implementation for performance optimization

## Dependencies

- CMake (3.30 or higher)
- PNG library
- Boost (filesystem component)
- Freetype
- ZLIB
- Pango
- Cairo
- GLib
- HarfBuzz
- OpenSSL

## Installation

### Prerequisites

Make sure you have all the required dependencies installed. On macOS with Homebrew:

```bash
brew install cmake libpng boost freetype zlib pkg-config pango cairo glib harfbuzz openssl@3
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/plot-library.git
cd plot-library

# Create a build directory
mkdir build && cd build

# Configure and build the project
cmake ..
make

# Run the executable
./bin/Plot
```
### Example Renders

![One of the out put](cmake-build-debug/bin/damped_oscillations.png)
![One of the out put](cmake-build-debug/bin/plot2.png)
![One of the out put](cmake-build-debug/bin/gaussian_plot.png)

## Usage Examples

### Basic Line Plot

```c++
#include "PlotLib/graph.hpp"
#include "MathLib/mathlib.hpp"

int main() {
    // Create a figure with dimensions 800x600
    Plot::Figure figure(800, 600);
    
    // Generate x values from 0 to 10 with 100 points
    auto x = Math::linspace(0.0f, 10.0f, 100);
    
    // Generate y values as sin(x)
    auto y = Math::Sin(x);
    
    // Set plot ranges
    figure.setXRange(0.0f, 10.0f);
    figure.setYRange(-1.2f, 1.2f);
    
    // Create a plot style
    Plot::PlotStyle style;
    style.color = Plot::Pixel::Blue;
    style.lineWidth = 2.0f;
    
    // Add the plot to the figure
    figure.addPlot(x, y, style);
    
    // Save the figure as PNG
    figure.render("sine_wave.png");
    
    return 0;
}
```

### Multiple Plots with Custom Styling

```c++
#include "PlotLib/graph.hpp"
#include "MathLib/mathlib.hpp"

int main() {
    Plot::Figure figure(800, 600);
    
    auto x = Math::linspace(0.0f, 10.0f, 100);
    auto sin_y = Math::Sin(x);
    auto cos_y = Math::Cos(x);
    
    figure.setXRange(0.0f, 10.0f);
    figure.setYRange(-1.2f, 1.2f);
    
    // Sine wave plot
    Plot::PlotStyle sinStyle;
    sinStyle.color = Plot::Pixel::Blue;
    sinStyle.lineWidth = 2.0f;
    sinStyle.plotType = Plot::PlotStyle::Type::Both;
    sinStyle.marker = Plot::PlotStyle::Marker::Circle;
    sinStyle.markerSize = 3.0f;
    
    // Cosine wave plot
    Plot::PlotStyle cosStyle;
    cosStyle.color = Plot::Pixel::Red;
    cosStyle.lineWidth = 2.0f;
    
    // Add the plots
    figure.addPlot(x, sin_y, sinStyle);
    figure.addPlot(x, cos_y, cosStyle);
    
    // Add LaTeX annotation
    figure.addLatexAnnotation("$f(x) = \\sin(x)$", 2.0f, 0.8f, 1.2f);
    
    // Configure axes
    Plot::AxisProperties xAxis, yAxis;
    xAxis.label = U"x-axis";
    yAxis.label = U"y-axis";
    figure.setAxisProperties(xAxis, yAxis);
    
    // Save as SVG for publication
    figure.render("trigonometric_functions.svg");
    
    return 0;
}
```

## Matrix Operations

The library includes a comprehensive matrix operations module:

```c++
#include "MathLib/Matrix.hpp"

int main() {
    // Create matrices
    Math::Matrix A(3, 3, Math::MatrixType::Identity);
    Math::Matrix B({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    
    // Matrix operations
    Math::Matrix C = A * B;  // Matrix multiplication
    Math::Matrix D = B.transpose();  // Transpose
    float det = B.determinant();  // Determinant
    
    // Decompositions
    auto [Q, R] = B.qrDecomposition();  // QR decomposition
    auto [L, U] = B.luDecomposition();  // LU decomposition
    
    // Print matrices
    C.print();
    
    return 0;
}
```

## Mathematical Functions

The library provides a wide range of mathematical functions for vector operations:

```c++
#include "MathLib/mathlib.hpp"
#include <iostream>

int main() {
    // Generate data
    auto x = Math::linspace(0.0f, 2.0f * M_PI, 10);
    
    // Apply mathematical functions
    auto y_sin = Math::Sin(x);
    auto y_exp = Math::Exp(x);
    auto y_log = Math::Log(x);
    
    // Statistical functions
    float mean = Math::Mean(y_sin);
    float variance = Math::Variance(y_sin);
    float stddev = Math::StdDev(y_sin);
    
    // Print some results
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard Deviation: " << stddev << std::endl;
    
    return 0;
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the IISER-B License - see the LICENSE file for details.