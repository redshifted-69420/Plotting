#include <chrono>
#include <iostream>
#include <vector>
#include "MathLib/Matrix.hpp"
#include "MathLib/Tensor.hpp"
#include "MathLib/mathlib.hpp"
#include "PlotLib/graph.hpp"


int main() {
  try {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing
    const std::vector<float> x = Math::linspace(-20.0f, 20.0f, 1000);
    const std::vector<float> y = 0.5f * Math::Sin(x);

    Plot::Figure fig(3189, 2037);

    // Set custom padding
    Plot::Figure::Padding padding;
    padding.left = 80; // More space for y-axis labels
    padding.right = 40; // Some space on right
    padding.top = 40; // Space for title if needed
    padding.bottom = 60; // Space for x-axis labels
    fig.setPadding(padding);

    fig.setXRange(-20.0f, 20.0f);
    fig.setYRange(-1.5f, 1.5f);

    // Configure X axis with improved aesthetics
    Plot::AxisProperties xAxis;
    xAxis.visible = true;
    xAxis.color = Plot::Pixel{40, 40, 40, 255}; // Darker gray for better contrast
    xAxis.thickness = 1.5f;
    xAxis.showTicks = true;
    xAxis.tickSpacing = 5.0f; // Fewer ticks for cleaner look
    xAxis.tickLength = 8.0f; // Shorter ticks
    xAxis.label = U"x"; // Add axis label

    // Configure Y axis
    Plot::AxisProperties yAxis = xAxis;
    yAxis.tickSpacing = 0.5f; // Adjusted for better spacing
    yAxis.label = U"sin(x)"; // Add axis label

    fig.setAxisProperties(xAxis, yAxis);

    // Configure grid with subtle appearance
    Plot::GridProperties grid;
    grid.visible = true;
    grid.color = Plot::Pixel{230, 230, 230, 255}; // Light gray for subtle grid
    grid.spacing = 300; // Adjusted for better density
    grid.lineThickness = 2.5f;
    fig.setGridProperties(grid);

    // Configure plot style with better aesthetics
    Plot::PlotStyle style;
    style.color = Plot::Pixel{41, 128, 185, 255}; // Appealing blue color
    style.lineWidth = 8.0f; // Thicker line for better visibility
    fig.addPlot(x, y, style);

    // White background is already default
    const auto canvas = fig.render("plot.png");
    fig.render("plot.svg");


    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> duration = end - start;

    std::cout << "Plot saved as plot.png" << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  try {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    const std::vector<float> x = Math::linspace(0.0f, 5.0f, 10);
    const std::vector<float> y = 0.2f * x * x + 1.0f;

    Plot::Figure fig(3127, 2021);

    // Set custom padding
    Plot::Figure::Padding padding;
    padding.left = 120; // More space for y-axis labels
    padding.right = 120; // Some space on right
    padding.top = 120; // Space for title if needed
    padding.bottom = 120; // Space for x-axis labels
    fig.setPadding(padding);

    fig.setXRange(-0.5f, 6.0f);
    fig.setYRange(-0.5f, 10.5f);

    // Configure X axis with improved aesthetics
    Plot::AxisProperties xAxis;
    xAxis.visible = true;
    xAxis.color = Plot::Pixel{40, 40, 40, 255}; // Darker gray for better contrast
    xAxis.thickness = 1.5f;
    xAxis.showTicks = true;
    xAxis.tickSpacing = 0.5f; // Fewer ticks for cleaner look
    xAxis.tickLength = 8.0f; // Shorter ticks
    xAxis.label = U"x"; // Add axis label

    // Configure Y axis
    Plot::AxisProperties yAxis = xAxis;
    yAxis.tickSpacing = 1.0f; // Adjusted for better spacing
    yAxis.label = U"y(x)"; // Add axis label

    fig.setAxisProperties(xAxis, yAxis);

    // Configure grid with subtle appearance
    Plot::GridProperties grid;
    grid.visible = true;
    grid.color = Plot::Pixel{230, 230, 230, 255}; // Light gray for subtle grid
    grid.spacing = 350; // Adjusted for better density
    grid.lineThickness = 2.5f;
    fig.setGridProperties(grid);

    // Configure plot style with better aesthetics
    Plot::PlotStyle style;
    style.color = Plot::Pixel{41, 128, 185, 255}; // Appealing blue color
    style.lineWidth = 8.0f; // Thicker line for better visibility
    style.plotType = Plot::PlotStyle::Type::Both;
    style.pointSize = 45.0f;
    fig.addPlot(x, y, style);

    // White background is already default
    const auto canvas = fig.render("plot2.png");

    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> duration = end - start;

    std::cout << "Plot saved as plot2.png" << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  try {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    const std::vector<float> x = Math::linspace(-2.0f, 2.0f, 1000);
    std::vector<float> y = Math::Exp((-1.0f * x * x));

    Plot::Figure fig(3000, 2000);

    // Set custom padding
    Plot::Figure::Padding padding;
    padding.left = 200; // More space for y-axis labels
    padding.right = 200; // Some space on right
    padding.top = 200; // Space for title and equations
    padding.bottom = 200; // Space for x-axis labels
    fig.setPadding(padding);

    fig.setXRange(-2.5f, 2.5f);
    fig.setYRange(-0.2f, 1.2f);

    // Configure axes, grid, etc. as in your other examples
    Plot::AxisProperties xAxis;
    xAxis.visible = true;
    xAxis.color = Plot::Pixel{40, 40, 40, 255};
    xAxis.thickness = 1.5f;
    xAxis.showTicks = true;
    xAxis.tickSpacing = 0.5f;
    xAxis.tickLength = 8.0f;
    xAxis.label = U"x";

    Plot::AxisProperties yAxis = xAxis;
    yAxis.tickSpacing = 0.2f;
    yAxis.label = U"f(x)";

    fig.setAxisProperties(xAxis, yAxis);

    Plot::GridProperties grid;
    grid.visible = true;
    grid.color = Plot::Pixel{230, 230, 230, 255};
    grid.spacing = 350;
    grid.lineThickness = 2.5f;
    fig.setGridProperties(grid);

    Plot::PlotStyle style;
    style.color = Plot::Pixel{41, 128, 185, 255};
    style.lineWidth = 8.0f;
    fig.addPlot(x, y, style);

    // Add LaTeX formula at the top of the plot
    fig.addLatexAnnotation("f(x) = e^{-x^2}", 0.0f, 1.1f, 1.2f, Plot::Pixel(0, 0, 0, 255));

    // Add another LaTeX formula explaining the properties
    fig.addLatexAnnotation("\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}", -2.0f, -0.1f, 1.0f,
                           Plot::Pixel(180, 0, 0, 255));

    const auto canvas = fig.render("gaussian_plot.png");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Plot saved as gaussian_plot.png" << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  try {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing
    // Generate some interesting data
    const std::vector<float> x = Math::linspace(-10.0f, 10.0f, 1000);
    const std::vector<float> y1 = Math::Sin(x) * Math::Exp(-1.0f * Math::Abs(x) / 5.0f);
    const std::vector<float> y2 = Math::Cos(x) * 0.5f * Math::Exp(-1.0f * Math::Abs(x) / 7.0f);
    const std::vector<float> y3 = Math::Sin(x * 2.0f) * Math::Exp(-1.0f * Math::Abs(x) / 3.0f);

    Plot::Figure figure(3000, 2000);

    // Set custom padding for the plot
    Plot::Figure::Padding padding;
    padding.left = 100; // More space for y-axis labels
    padding.right = 100; // Space on the right
    padding.top = 100; // Space above the plot
    padding.bottom = 100; // Space for x-axis labels
    figure.setPadding(padding);

    // Configure axis properties for better visualization
    Plot::AxisProperties xAxisProps, yAxisProps;

    // X-axis configuration
    xAxisProps.label = U"Time (s)";
    xAxisProps.thickness = 3.0f;
    xAxisProps.tickLength = 6.0f;

    // Y-axis configuration
    yAxisProps.label = U"Amplitude";
    yAxisProps.thickness = 3.0f;
    yAxisProps.tickLength = 6.0f;
    yAxisProps.showTicks = true;
    yAxisProps.tickSpacing = 0.2f;

    figure.setAxisProperties(xAxisProps, yAxisProps);

    Plot::GridProperties grid;
    grid.visible = true;
    grid.color = Plot::Pixel(220, 220, 220, 200);
    grid.lineThickness = 2.5f;
    grid.spacing = 200;

    figure.setGridProperties(grid);

    Plot::PlotStyle style1, style2, style3;
    style1.color = Plot::Pixel::Blue;
    style1.lineWidth = 5.0f;
    style2.color = Plot::Pixel::Red;
    style2.lineWidth = 5.0f;
    style3.color = Plot::Pixel::Orange;
    style3.lineWidth = 5.0f;

    figure.setXRange(-10.5f, 10.5f);
    figure.setYRange(-0.75f, 0.75f);

    figure.addPlot(x, y1, style1);
    figure.addPlot(x, y2, style2);
    figure.addPlot(x, y3, style3);

    figure.addLatexAnnotation("y = \\sin(x) \\cdot e^{-|x|/5}", 2.25f, 0.6f, 1.5f);

    Plot::Canvas canvas = figure.render("damped_oscillations.png");
    figure.render("damped_oscillations.svg");
    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }

  try {
    Matrix M = Matrix::random(4096, 4096);
    M.print();
    auto start = std::chrono::high_resolution_clock::now();
    Matrix M_ = M * M;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to multiply matrix: " << duration.count() << " ms\n";
    M_.print();
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }

  try {
    Matrix M1 = Matrix::random(4096, 4096);
    Matrix M2 = Matrix::random(4096, 4096);
    auto start = std::chrono::high_resolution_clock::now();
    Matrix M_ = M1 + M2;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to add matrix: " << duration.count() << " ms\n";
    M_.print();
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }

  try {
    Matrix M1 = Matrix::random(4096, 4096);
    Matrix M2 = Matrix::random(4096, 4096);
    auto start = std::chrono::high_resolution_clock::now();
    Matrix M_ = M1 - M2;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to sub matrix: " << duration.count() << " ms\n";
    M_.print();
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }

  return 0;
}
