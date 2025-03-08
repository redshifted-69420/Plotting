#include "graph.hpp"
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/process/child.hpp>
#include <boost/process/io.hpp>
#include <boost/scope_exit.hpp>
#include <cairo.h>
#include <cairo/cairo.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <pango/pangocairo.h>
#include <png.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>
#include "SvgGenerator.hpp"
#include "ThreadPool.hpp"

namespace Plot {

  namespace fs = std::filesystem;

  class LaTeXCache {
public:
    static LaTeXCache &getInstance() {
      static LaTeXCache instance;
      return instance;
    }

    std::optional<std::string> getCachedFilePath(const std::string &latexCode, float scale, const Pixel &textColor) {
      std::string key = generateHash(latexCode, scale, textColor);
      std::string filePath = cacheDir + "/" + key + ".png";

      if (fs::exists(filePath)) {
        return filePath;
      }
      return std::nullopt;
    }

    void store(const std::string &latexCode, float scale, const Pixel &textColor, const std::string &filePath) {
      std::string key = generateHash(latexCode, scale, textColor);
      fs::copy(filePath, cacheDir + "/" + key + ".png", fs::copy_options::overwrite_existing);
    }
    void cleanupCache(size_t maxCacheSize);

private:
    LaTeXCache() {
      cacheDir = fs::temp_directory_path().string() + "/LaTeXCache";
      fs::create_directories(cacheDir);
    }

    std::string generateHash(const std::string &latexCode, float scale, const Pixel &textColor) {
      std::string input = latexCode + std::to_string(scale) + std::to_string(textColor.r) +
                          std::to_string(textColor.g) + std::to_string(textColor.b);
      unsigned char hash[SHA256_DIGEST_LENGTH];
      SHA256(reinterpret_cast<const unsigned char *>(input.c_str()), input.size(), hash);

      std::stringstream ss;
      for (unsigned char c: hash) {
        ss << std::hex << (int) c;
      }
      return ss.str();
    }

    std::string cacheDir;
  };


  // Optimized PNG loading using RAII
  class PNGLoader {
public:
    PNGLoader(const std::string &filePath) : fp(nullptr), png(nullptr), info(nullptr) {
      fp = fopen(filePath.c_str(), "rb");
      if (!fp) {
        throw std::runtime_error("Failed to open PNG file: " + filePath);
      }

      png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
      if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG read structure");
      }

      info = png_create_info_struct(png);
      if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info structure");
      }

      if (setjmp(png_jmpbuf(png))) {
        throw std::runtime_error("Error during PNG read");
      }

      png_init_io(png, fp);
      png_read_info(png, info);

      width = png_get_image_width(png, info);
      height = png_get_image_height(png, info);
      color_type = png_get_color_type(png, info);
      bit_depth = png_get_bit_depth(png, info);

      configureTransforms();
      png_read_update_info(png, info);

      row_size = png_get_rowbytes(png, info);
      has_alpha = (png_get_color_type(png, info) & PNG_COLOR_MASK_ALPHA) || png_get_valid(png, info, PNG_INFO_tRNS);
    }

    ~PNGLoader() {
      if (png && info) {
        png_destroy_read_struct(&png, &info, nullptr);
      }
      if (fp) {
        fclose(fp);
      }
    }

    std::vector<Pixel> readImageData(const Pixel &textColor) {
      std::vector<std::vector<uint8_t>> row_pointers(height, std::vector<uint8_t>(row_size));
      std::vector<png_bytep> row_ptrs;
      row_ptrs.reserve(height);

      for (auto &row: row_pointers) {
        row_ptrs.push_back(row.data());
      }

      png_read_image(png, row_ptrs.data());

      std::vector<Pixel> blendedPixels(width * height);
      const float invScale = 1.0f / 255.0f;

// SIMD-friendly loop with minimal branches
#pragma omp parallel for
      for (int y = 0; y < height; y++) {
        const uint8_t *row = row_pointers[y].data();
        for (int x = 0; x < width; x++) {
          const uint8_t *px = &(row[x * 4]);
          const uint8_t r = px[0], g = px[1], b = px[2], a = px[3];
          const float alphaFactor = a * invScale;

          // Pre-compute color blending
          blendedPixels[y * width + x] = Pixel(static_cast<uint8_t>(textColor.r * (r * invScale) * alphaFactor),
                                               static_cast<uint8_t>(textColor.g * (g * invScale) * alphaFactor),
                                               static_cast<uint8_t>(textColor.b * (b * invScale) * alphaFactor), a);
        }
      }

      return blendedPixels;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    void configureTransforms() {
      if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
      if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
      if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
      if (bit_depth == 16)
        png_set_strip_16(png);
      if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    }

    FILE *fp;
    png_structp png;
    png_infop info;
    int width;
    int height;
    png_byte color_type;
    png_byte bit_depth;
    size_t row_size;
    bool has_alpha;
  };

  // Optimized command execution
  class CommandExecutor {
public:
    static std::string execute(const std::string_view command, const std::string_view workingDir = "") {
      namespace bp = boost::process;

      bp::ipstream pipe_stream;
      std::string output;
      std::string cmd_string = std::string(command); // Convert to std::string
      std::string working_dir_string = std::string(workingDir); // Convert to std::string

      try {
        bp::child c;
        if (workingDir.empty()) {
          c = bp::child(cmd_string, bp::std_out > pipe_stream, bp::std_err > bp::null);
        } else {
          c = bp::child(cmd_string, bp::std_out > pipe_stream, bp::std_err > bp::null,
                        bp::start_dir = working_dir_string);
        }

        std::string line;
        while (pipe_stream && std::getline(pipe_stream, line)) {
          output += line + "\n";
        }

        c.wait();
        if (c.exit_code() != 0) {
          throw std::runtime_error("Command failed with exit code " + std::to_string(c.exit_code()) + ": " +
                                   cmd_string);
        }
      } catch (const boost::process::process_error &e) {
        throw std::runtime_error("Failed to execute command: " + cmd_string + " Error: " + e.what());
      }

      return output;
    }

    // Asynchronous execution for non-blocking operations
    static std::future<std::string> executeAsync(const std::string_view command) {
      return std::async(std::launch::async, [command]() { return execute(command); });
    }
  };

  const Pixel Pixel::Black = {0, 0, 0, 255};
  const Pixel Pixel::White = {255, 255, 255, 255};
  const Pixel Pixel::Red = {255, 0, 0, 255};
  const Pixel Pixel::Green = {0, 255, 0, 255};
  const Pixel Pixel::Blue = {0, 0, 255, 255};
  const Pixel Pixel::Gray = {128, 128, 128, 255};
  const Pixel Pixel::DarkGray = {64, 64, 64, 255};
  const Pixel Pixel::LightGray = {192, 192, 192, 255};
  const Pixel Pixel::Cyan = {0, 255, 255, 255};
  const Pixel Pixel::Magenta = {255, 0, 255, 255};
  const Pixel Pixel::Yellow = {255, 255, 0, 255};
  const Pixel Pixel::Orange = {255, 165, 0, 255};
  const Pixel Pixel::Purple = {128, 0, 128, 255};
  const Pixel Pixel::Pink = {255, 192, 203, 255};
  const Pixel Pixel::Brown = {165, 42, 42, 255};
  const Pixel Pixel::Lime = {0, 255, 0, 255};
  const Pixel Pixel::Olive = {128, 128, 0, 255};
  const Pixel Pixel::Teal = {0, 128, 128, 255};
  const Pixel Pixel::Navy = {0, 0, 128, 255};
  const Pixel Pixel::Maroon = {128, 0, 0, 255};
  const Pixel Pixel::Gold = {255, 215, 0, 255};
  const Pixel Pixel::Beige = {245, 245, 220, 255};
  const Pixel Pixel::Salmon = {250, 128, 114, 255};
  const Pixel Pixel::Turquoise = {64, 224, 208, 255};
  const Pixel Pixel::Lavender = {230, 230, 250, 255};
  const Pixel Pixel::Indigo = {75, 0, 130, 255};
  const Pixel Pixel::Violet = {238, 130, 238, 255};
  const Pixel Pixel::SkyBlue = {135, 206, 235, 255};
  const Pixel Pixel::Coral = {255, 127, 80, 255};
  const Pixel Pixel::Mint = {189, 252, 201, 255};
  const Pixel Pixel::Transparent = {0, 0, 0, 0};

  void blendPixel(Pixel &dst, const Pixel &src, const float alpha) {
    const float a = alpha * (static_cast<float>(src.a) / 255.0f);
    const float inv_a = 1.0f - a;
    dst.r = static_cast<uint8_t>(static_cast<float>(src.r) * a + static_cast<float>(dst.r) * inv_a);
    dst.g = static_cast<uint8_t>(static_cast<float>(src.g) * a + static_cast<float>(dst.g) * inv_a);
    dst.b = static_cast<uint8_t>(static_cast<float>(src.b) * a + static_cast<float>(dst.b) * inv_a);
  }

  Canvas::Canvas(const int width, const int height, const Pixel &background) :
      width_(width), height_(height), pixels_(width * height, background) {
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument("Canvas dimensions must be positive");
    }
  }

  void Figure::setAxisProperties(const AxisProperties &xProps, const AxisProperties &yProps) {
    xAxisProps_ = xProps;
    yAxisProps_ = yProps;
  }

  void Canvas::drawRect(int x, int y, int width, int height, const Pixel &color, float thickness) {
    for (int i = 0; i < thickness; i++) {
      drawLine(x + i, y + i, x + width - 1 - i, y + i, color, 1.0f); // Top
      drawLine(x + i, y + i, x + i, y + height - 1 - i, color, 1.0f); // Left
      drawLine(x + i, y + height - 1 - i, x + width - 1 - i, y + height - 1 - i, color, 1.0f); // Bottom
      drawLine(x + width - 1 - i, y + i, x + width - 1 - i, y + height - 1 - i, color, 1.0f); // Right
    }
  }

  void Figure::drawAxes(Canvas &canvas) const {
    if (!xAxisProps_.visible && !yAxisProps_.visible) {
      return;
    }

    int plotX = getPlotX();
    int plotY = getPlotY();
    int plotWidth = getPlotWidth();
    int plotHeight = getPlotHeight();

    // --- X-Axis ---
    if (xAxisProps_.visible) {
      canvas.drawLine(plotX, plotY + plotHeight, plotX + plotWidth, plotY + plotHeight, xAxisProps_.color,
                      xAxisProps_.thickness);

      canvas.addSvgLine(plotX, plotY + plotHeight, plotX + plotWidth, plotY + plotHeight, xAxisProps_.color,
                        xAxisProps_.thickness);

      for (float x = xRange_.first; x <= xRange_.second; x += xAxisProps_.tickSpacing) {
        int xPixel = worldToPixelX(x);

        canvas.drawLine(xPixel, plotY + plotHeight, xPixel, plotY + plotHeight + xAxisProps_.tickLength,
                        xAxisProps_.color, xAxisProps_.thickness);

        canvas.addSvgLine(xPixel, plotY + plotHeight, xPixel, plotY + plotHeight + xAxisProps_.tickLength,
                          xAxisProps_.color, xAxisProps_.thickness);

        if (xAxisProps_.showLabels) {
          std::u32string label = formatNumber(x, xAxisProps_.labelFormat);
          auto textSize = TextRenderer::getTextSize(label);
          int textX = xPixel - textSize.first / 2;
          int textY = plotY + plotHeight + xAxisProps_.tickLength + 35; // Adjust label position

          TextRenderer::renderText(canvas, label, textX, textY, xAxisProps_.color);

          // Store labels as SVG text
          canvas.addSvgText(textX, textY, std::string(label.begin(), label.end()), xAxisProps_.color, 20);
        }
      }

      if (!xAxisProps_.label.empty()) {
        auto xLabelSize = TextRenderer::getTextSize(xAxisProps_.label);
        int xLabelX = plotX + plotWidth / 2 - xLabelSize.first / 2;
        int xLabelY = plotY + plotHeight + xAxisProps_.tickLength + xLabelSize.second + 65;

        TextRenderer::renderText(canvas, xAxisProps_.label, xLabelX, xLabelY, xAxisProps_.color);

        // Store axis label in SVG
        canvas.addSvgText(xLabelX, xLabelY, std::string(xAxisProps_.label.begin(), xAxisProps_.label.end()),
                          xAxisProps_.color, 24);
      }
    }

    // --- Y-Axis ---
    if (yAxisProps_.visible) {
      canvas.drawLine(plotX, plotY, plotX, plotY + plotHeight, yAxisProps_.color, yAxisProps_.thickness);

      canvas.addSvgLine(plotX, plotY, plotX, plotY + plotHeight, yAxisProps_.color, yAxisProps_.thickness);

      for (float y = yRange_.first; y <= yRange_.second; y += yAxisProps_.tickSpacing) {
        int yPixel = worldToPixelY(y);

        canvas.drawLine(plotX, yPixel, plotX - yAxisProps_.tickLength, yPixel, yAxisProps_.color,
                        yAxisProps_.thickness);

        canvas.addSvgLine(plotX, yPixel, plotX - yAxisProps_.tickLength, yPixel, yAxisProps_.color,
                          yAxisProps_.thickness);

        if (yAxisProps_.showLabels) {
          std::u32string label = formatNumber(y, yAxisProps_.labelFormat);
          auto textSize = TextRenderer::getTextSize(label);

          int textX = plotX - yAxisProps_.tickLength - textSize.first - 25;
          int textY = yPixel + textSize.second / 2;

          TextRenderer::renderText(canvas, label, textX, textY, yAxisProps_.color);

          // Store labels as SVG text
          canvas.addSvgText(textX, textY, std::string(label.begin(), label.end()), yAxisProps_.color, 20);
        }
      }

      if (!yAxisProps_.label.empty()) {
        auto yLabelSize = TextRenderer::getTextSize(yAxisProps_.label);
        int yLabelX = plotX - yAxisProps_.tickLength - yLabelSize.second - 50;
        int yLabelY = plotY + plotHeight / 2 + yLabelSize.first / 2;

        TextRenderer::renderText(canvas, yAxisProps_.label, yLabelX, yLabelY, yAxisProps_.color, 1.0f, 90);

        // Store axis label in SVG
        canvas.addSvgText(yLabelX, yLabelY, std::string(yAxisProps_.label.begin(), yAxisProps_.label.end()),
                          yAxisProps_.color, 24);
      }
    }
  }


  inline std::u32string utf8_to_u32(const std::string &utf8_str) {
    std::u32string utf32_str;
    size_t len = std::mbstowcs(nullptr, utf8_str.c_str(), 0); // Get length
    if (len == static_cast<size_t>(-1))
      return U""; // Conversion failed

    utf32_str.resize(len);
    std::mbstowcs(reinterpret_cast<wchar_t *>(&utf32_str[0]), utf8_str.c_str(), len);
    return utf32_str;
  }

  std::u32string Figure::formatNumber(float value, const std::u32string &format) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value; // Convert float to string
    return utf8_to_u32(ss.str()); // Convert to UTF-32
  }

  void Canvas::drawLine(float x1, float y1, float x2, float y2, const Pixel &color, const float thickness) {
    // Fast path for single-pixel lines
    if (thickness <= 1.0f) {
      const float dx = x2 - x1;

      // Determine if line is more vertical than horizontal
      if (const float dy = y2 - y1; std::abs(dx) < std::abs(dy)) {
        if (y2 < y1) {
          std::swap(x1, x2);
          std::swap(y1, y2);
        }
        const float gradient = dx / dy;
        const int y_start = static_cast<int>(y1);
        const int y_end = static_cast<int>(y2);
        float xpos = x1;

#pragma omp parallel for if (y_end - y_start > 1000)
        for (int y = y_start; y <= y_end; y++) {
          if (y >= 0 && y < height_) {
            const int xint = static_cast<int>(xpos);
            if (xint >= -1 && xint < width_) {
              const float fract = xpos - static_cast<float>(xint);
              Pixel &dst1 = pixels_[y * width_ + xint];
              blendPixel(dst1, color, 1.0f - fract);
              if (xint + 1 < width_) {
                Pixel &dst2 = pixels_[y * width_ + xint + 1];
                blendPixel(dst2, color, fract);
              }
            }
          }
          xpos += gradient;
        }
      } else {
        if (x2 < x1) {
          std::swap(x1, x2);
          std::swap(y1, y2);
        }
        const float gradient = dy / dx;
        const int x_start = static_cast<int>(x1);
        const int x_end = static_cast<int>(x2);
        float ypos = y1;

#pragma omp parallel for if (x_end - x_start > 1000)
        for (int x = x_start; x <= x_end; x++) {
          if (x >= 0 && x < width_) {
            const int yint = static_cast<int>(ypos);
            if (yint >= -1 && yint < height_) {
              const float fract = ypos - static_cast<float>(yint);
              Pixel &dst1 = pixels_[yint * width_ + x];
              blendPixel(dst1, color, 1.0f - fract);
              if (yint + 1 < height_) {
                Pixel &dst2 = pixels_[(yint + 1) * width_ + x];
                blendPixel(dst2, color, fract);
              }
            }
          }
          ypos += gradient;
        }
      }
    } else {
      // Thick line drawing using parallel lines
      const float dx = x2 - x1;
      const float dy = y2 - y1;
      const float len = std::sqrt(dx * dx + dy * dy);
      if (len < 0.0001f)
        return;

      const float nx = -dy / len;
      const float ny = dx / len;
      const int steps = static_cast<int>(thickness * 2);
      const float step_size = thickness / static_cast<float>(steps);

#pragma omp parallel for if (steps > 8)
      for (int i = 0; i <= steps; ++i) {
        const float offset = -thickness / 2 + static_cast<float>(i) * step_size;
        const float ox1 = x1 + nx * offset;
        const float oy1 = y1 + ny * offset;
        const float ox2 = x2 + nx * offset;
        const float oy2 = y2 + ny * offset;

        // Draw single-pixel width line
        drawLine(ox1, oy1, ox2, oy2, color, 1.0f);
      }
    }
  }

  void Canvas::setPixel(const int x, const int y, const Pixel &color) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
      pixels_[y * width_ + x] = color;
    }
  }

  void Canvas::clear(const Pixel &color) { std::ranges::fill(pixels_, color); }

  Figure::Figure(const int width, const int height) : width_(width), height_(height) {
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument("Figure dimensions must be positive");
    }
  }

  void Figure::setXRange(float min, float max) {
    if (min >= max) {
      throw std::invalid_argument("X range min must be less than max");
    }
    xRange_ = {min, max};
  }

  void Figure::setYRange(float min, float max) {
    if (min >= max) {
      throw std::invalid_argument("Y range min must be less than max");
    }
    yRange_ = {min, max};
  }

  void Figure::setGridProperties(const GridProperties &props) { gridProps_ = props; }

  void Figure::addPlot(const std::vector<float> &x, const std::vector<float> &y, const PlotStyle &style) {
    if (x.size() != y.size() || x.empty()) {
      throw std::invalid_argument("X and Y data must have same non-zero size");
    }
    plots_.emplace_back(x, y);
    plotStyles_.push_back(style);
  }

  Canvas Figure::render(const std::string &fileName) const {
    Canvas canvas(width_, height_, Pixel::White);
    canvas.drawRect(0, 0, width_ - 1, height_ - 1, Pixel::Black, 1.0f);

    if (gridProps_.visible) {
      drawGrid(canvas);
    }
    drawAxes(canvas);

    for (size_t i = 0; i < plots_.size(); ++i) {
      const auto &[x, y] = plots_[i];
      drawPlot(canvas, x, y, plotStyles_[i]);
    }

    float xScale = (width_ - padding_.left - padding_.right) / (xRange_.second - xRange_.first);
    float yScale = (height_ - padding_.top - padding_.bottom) / (yRange_.second - yRange_.first);

    for (const auto &annotation: latexAnnotations) {
      int pixelX = padding_.left + static_cast<int>((annotation.x - xRange_.first) * xScale);
      int pixelY = canvas.getHeight() - padding_.bottom - static_cast<int>((annotation.y - yRange_.first) * yScale);
      TextRenderer::renderLatex(canvas, annotation.latex, pixelX, pixelY, annotation.scale, annotation.color);
    }

    // **Determine file format and save accordingly**
    if (fileName.ends_with(".svg")) {
      Plot::SvgGenerator::saveSVG(canvas, fileName, width_, height_);
    } else if (fileName.ends_with(".png")) {
      Plot::PngGenerator::savePNG(canvas, fileName);
    } else {
      throw std::runtime_error("Unsupported file format: " + fileName);
    }

    return canvas;
  }

  void Figure::drawGrid(Canvas &canvas) const {
    // Convert grid spacing from screen pixels to data units
    const int plotWidth = getPlotWidth();
    const int plotHeight = getPlotHeight();
    const float xSpacing = gridProps_.spacing * (xRange_.second - xRange_.first) / plotWidth;
    const float ySpacing = gridProps_.spacing * (yRange_.second - yRange_.first) / plotHeight;

    // Draw vertical grid lines
    for (float x = std::ceil(xRange_.first / xSpacing) * xSpacing; x <= xRange_.second; x += xSpacing) {
      float screenX = padding_.left + (x - xRange_.first) / (xRange_.second - xRange_.first) * plotWidth;

      // Store in PNG canvas
      canvas.drawLine(screenX, padding_.top, screenX, height_ - padding_.bottom, gridProps_.color,
                      gridProps_.lineThickness);

      // Store in SVG canvas
      canvas.addSvgLine(screenX, padding_.top, screenX, height_ - padding_.bottom, gridProps_.color,
                        gridProps_.lineThickness);
    }

    // Draw horizontal grid lines
    for (float y = std::ceil(yRange_.first / ySpacing) * ySpacing; y <= yRange_.second; y += ySpacing) {
      float screenY = padding_.top + plotHeight - (y - yRange_.first) / (yRange_.second - yRange_.first) * plotHeight;

      // Store in PNG canvas
      canvas.drawLine(padding_.left, screenY, width_ - padding_.right, screenY, gridProps_.color,
                      gridProps_.lineThickness);

      // Store in SVG canvas
      canvas.addSvgLine(padding_.left, screenY, width_ - padding_.right, screenY, gridProps_.color,
                        gridProps_.lineThickness);
    }
  }

  void Figure::drawPlot(Canvas &canvas, const std::vector<float> &x, const std::vector<float> &y,
                        const PlotStyle &style) const {
    validateData(x, y);

    const float xMin = xRange_.first == 0.0f ? *std::ranges::min_element(x) : xRange_.first;
    const float xMax = xRange_.second == 0.0f ? *std::ranges::max_element(x) : xRange_.second;
    const float yMin = yRange_.first == 0.0f ? *std::ranges::min_element(y) : yRange_.first;
    const float yMax = yRange_.second == 0.0f ? *std::ranges::max_element(y) : yRange_.second;

    const int plotWidth = getPlotWidth();
    const int plotHeight = getPlotHeight();

    std::stringstream svgPathData;
    bool firstPoint = true;

    // --- Line Plots ---
    if (style.plotType == PlotStyle::Type::Line || style.plotType == PlotStyle::Type::Both) {
      for (size_t i = 1; i < x.size(); ++i) {
        if (isValidNumber(x[i - 1]) && isValidNumber(x[i]) && isValidNumber(y[i - 1]) && isValidNumber(y[i])) {
          float x1 = padding_.left + (x[i - 1] - xMin) / (xMax - xMin) * plotWidth;
          float y1 = padding_.top + plotHeight - (y[i - 1] - yMin) / (yMax - yMin) * plotHeight;
          float x2 = padding_.left + (x[i] - xMin) / (xMax - xMin) * plotWidth;
          float y2 = padding_.top + plotHeight - (y[i] - yMin) / (yMax - yMin) * plotHeight;

          // Store in PNG canvas
          canvas.drawLineClipped(x1, y1, x2, y2, style.color, style.lineWidth);

          // Store in SVG path format
          if (firstPoint) {
            svgPathData << "M " << x1 << " " << y1;
            firstPoint = false;
          }
          svgPathData << " L " << x2 << " " << y2;
        }
      }

      // If path is non-empty, add it to SVG
      if (!firstPoint) {
        std::stringstream svgPath;
        svgPath << "<path d=\"" << svgPathData.str() << "\" stroke=\"rgb(" << (int) style.color.r << ","
                << (int) style.color.g << "," << (int) style.color.b << ")\" stroke-width=\"" << style.lineWidth
                << "\" stroke-linecap=\"round\" stroke-linejoin=\"round\" fill=\"none\" />";
        canvas.svgElements.push_back({"path", svgPath.str()});
      }
    }

    // --- Scatter Plot ---
    if (style.plotType == PlotStyle::Type::Scatter || style.plotType == PlotStyle::Type::Both) {
      for (size_t i = 0; i < x.size(); ++i) {
        float screenX = padding_.left + (x[i] - xMin) / (xMax - xMin) * plotWidth;
        float screenY = padding_.top + plotHeight - (y[i] - yMin) / (yMax - yMin) * plotHeight;

        // Store in PNG canvas
        drawPoint(canvas, screenX, screenY, style.color, style.pointSize);

        // Store in SVG as circle
        std::stringstream svgCircle;
        svgCircle << "<circle cx=\"" << screenX << "\" cy=\"" << screenY << "\" r=\"" << style.pointSize / 2
                  << "\" fill=\"rgb(" << (int) style.color.r << "," << (int) style.color.g << "," << (int) style.color.b
                  << ")\" stroke=\"black\" stroke-width=\"1\" />";
        canvas.svgElements.push_back({"circle", svgCircle.str()});
      }
    }

    // --- Markers ---
    if (style.marker != PlotStyle::Marker::None) {
      for (size_t i = 0; i < x.size(); ++i) {
        float screenX = padding_.left + (x[i] - xMin) / (xMax - xMin) * getPlotWidth();
        float screenY = padding_.top + getPlotHeight() - (y[i] - yMin) / (yMax - yMin) * getPlotHeight();

        // Store in PNG canvas
        drawMarker(canvas, screenX, screenY, style.color, style.marker, style.markerSize);

        // Store markers as SVG (example: small squares as an alternative marker)
        std::stringstream svgMarker;
        svgMarker << "<rect x=\"" << (screenX - style.markerSize / 2) << "\" y=\"" << (screenY - style.markerSize / 2)
                  << "\" width=\"" << style.markerSize << "\" height=\"" << style.markerSize << "\" fill=\"rgb("
                  << (int) style.color.r << "," << (int) style.color.g << "," << (int) style.color.b
                  << ")\" stroke=\"black\" stroke-width=\"1\" />";
        canvas.svgElements.push_back({"rect", svgMarker.str()});
      }
    }
  }


  void Figure::drawPoint(Canvas &canvas, float x, float y, const Pixel &color, float size) const {
    const float radius = size / 2.0f;
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int r = static_cast<int>(radius);

    for (int dy = -r; dy <= r; ++dy) {
      for (int dx = -r; dx <= r; ++dx) {
        if (dx * dx + dy * dy <= r * r) {
          canvas.setPixel(x0 + dx, y0 + dy, color);
        }
      }
    }
  }

  namespace {
    void pngErrorHandler(png_structp png, const png_const_charp error_msg) {
      throw std::runtime_error(std::string("PNG error: ") + error_msg);
    }

    void pngWarningHandler(png_structp /*png*/, png_const_charp warning_msg) {}
  } // namespace

  void PngGenerator::savePNG(const Canvas &canvas, const std::string &filename) {
    const int width = canvas.getWidth();
    const int height = canvas.getHeight();
    const std::vector<Pixel> &pixels = canvas.getPixels();

    const struct FileGuard {
      FILE *fp;
      explicit FileGuard(const char *fname) : fp(fopen(fname, "wb")) {}
      ~FileGuard() {
        if (fp)
          fclose(fp);
      }
    } file(filename.c_str());

    if (!file.fp) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    const png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, pngErrorHandler, pngWarningHandler);
    if (!png) {
      throw std::runtime_error("Failed to create PNG write structure");
    }

    const struct PngGuard {
      png_structp png;
      png_infop info;
      PngGuard(png_structp p, png_infop i) : png(p), info(i) {}
      ~PngGuard() { png_destroy_write_struct(&png, &info); }
    } png_guard(png, png_create_info_struct(png));

    if (!png_guard.info) {
      throw std::runtime_error("Failed to create PNG info structure");
    }

    png_init_io(png, file.fp);
    png_set_IHDR(png, png_guard.info, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<unsigned char> buffer(width * height * 4);
    for (int y = 0; y < height; ++y) {
      memcpy(&buffer[y * width * 4], &pixels[y * width], width * sizeof(Pixel));
    }

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; ++y) {
      row_pointers[y] = &buffer[y * width * 4];
    }

    png_write_info(png, png_guard.info);
    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);
  }

  FT_Library TextRenderer::library_ = nullptr;
  FT_Face TextRenderer::face_ = nullptr;
  std::unordered_map<char32_t, TextRenderer::GlyphInfo> TextRenderer::glyphCache_;

  void TextRenderer::initialize(const std::string &fontPath) {
    if (FT_Init_FreeType(&library_)) {
      throw std::runtime_error("Failed to initialize FreeType");
    }

    if (FT_New_Face(library_, fontPath.c_str(), 0, &face_)) {
      FT_Done_FreeType(library_);
      throw std::runtime_error("Failed to load font: " + fontPath);
    }

    if (FT_Set_Pixel_Sizes(face_, 0, 24)) {
      FT_Done_Face(face_);
      FT_Done_FreeType(library_);
      throw std::runtime_error("Failed to set font size");
    }
  }

  void TextRenderer::shutdown() {
    if (face_) {
      FT_Done_Face(face_);
    }
    if (library_) {
      FT_Done_FreeType(library_);
    }
    glyphCache_.clear();
  }

  TextRenderer::GlyphInfo &TextRenderer::getCachedGlyph(char32_t c) {
    auto it = glyphCache_.find(c);
    if (it != glyphCache_.end()) {
      return it->second;
    }

    if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
      throw std::runtime_error("Failed to load glyph");
    }

    GlyphInfo glyph;
    FT_GlyphSlot slot = face_->glyph;
    glyph.width = slot->bitmap.width;
    glyph.height = slot->bitmap.rows;
    glyph.bearingX = slot->bitmap_left;
    glyph.bearingY = slot->bitmap_top;
    glyph.advance = slot->advance.x >> 6;

    glyph.bitmap.resize(glyph.width * glyph.height);
    memcpy(glyph.bitmap.data(), slot->bitmap.buffer, glyph.bitmap.size());

    return glyphCache_.emplace(c, std::move(glyph)).first->second;
  }

  void TextRenderer::renderText(Canvas &canvas, const std::u32string &text, int x, int y, const Pixel &color,
                                float scale, float angleDegrees) {
    if (!face_ || !library_) {
      throw std::runtime_error("FreeType not initialized");
    }

    float angleRadians = angleDegrees * M_PI / 180.0f;
    float cosAngle = std::cos(angleRadians);
    float sinAngle = std::sin(angleRadians);


    int cursorX = x;
    const int width = canvas.getWidth();
    const int height = canvas.getHeight();


    for (char32_t c: text) {
      GlyphInfo &glyph = getCachedGlyph(c);

      for (int row = 0; row < glyph.height; ++row) {
        for (int col = 0; col < glyph.width; ++col) {
          if (glyph.bitmap[row * glyph.width + col] > 0) {


            int glyphX = cursorX + static_cast<int>(glyph.bearingX * scale);
            int glyphY = y - static_cast<int>(glyph.bearingY * scale);

            float rotatedX = glyphX + col * scale;
            float rotatedY = glyphY + row * scale;


            int px = static_cast<int>(x + (rotatedX - x) * cosAngle - (rotatedY - y) * sinAngle);
            int py = static_cast<int>(y + (rotatedX - x) * sinAngle + (rotatedY - y) * cosAngle);


            if (px >= 0 && px < width && py >= 0 && py < height) {


              float alpha = glyph.bitmap[row * glyph.width + col] / 255.0f;

              Pixel &dst = canvas.getPixelRef(px, py);

              blendPixel(dst, color, alpha);
            }
          }
        }
      }
      cursorX += static_cast<int>(glyph.advance * scale);
    }
  }

  int Figure::worldToPixelX(float x) const {
    return static_cast<int>(getPlotX() + (x - xRange_.first) / (xRange_.second - xRange_.first) * getPlotWidth());
  }

  void Figure::drawMarker(Canvas &canvas, float x, float y, const Pixel &color, PlotStyle::Marker marker,
                          float size) const {
    int xCenter = static_cast<int>(x);
    int yCenter = static_cast<int>(y);
    int radius = static_cast<int>(size / 2.0f);


    switch (marker) {
      case PlotStyle::Marker::Circle:
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy <= radius * radius) {
              canvas.setPixel(xCenter + dx, yCenter + dy, color);
            }
          }
        }
        break;

      case PlotStyle::Marker::Square:
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dx = -radius; dx <= radius; ++dx) {
            canvas.setPixel(xCenter + dx, yCenter + dy, color);
          }
        }
        break;


      case PlotStyle::Marker::Triangle: {
        for (int dy = -radius; dy <= radius; ++dy) {
          int widthAtY = static_cast<int>(2 * radius * (1.0f - static_cast<float>(std::abs(dy)) / radius));
          int startX = xCenter - widthAtY / 2;
          for (int dx = 0; dx < widthAtY; ++dx) {
            canvas.setPixel(startX + dx, yCenter + dy, color);
          }
        }


      } break;

      case PlotStyle::Marker::None:
      default:
        break;
    }
  }

  int Figure::worldToPixelY(float y) const {
    return static_cast<int>(getPlotY() + getPlotHeight() -
                            (y - yRange_.first) / (yRange_.second - yRange_.first) * getPlotHeight());
  }

  std::pair<int, int> TextRenderer::getTextSize(const std::u32string &text, float scale) {
    int width = 0;
    int height = 0;

    for (char c: text) {
      const GlyphInfo &glyph = getCachedGlyph(c);
      width += static_cast<int>(glyph.advance * scale);
      height = std::max(height, static_cast<int>(glyph.height * scale));
    }

    return {width, height};
  }

  void Figure::validateData(const std::vector<float> &x, const std::vector<float> &y) const {
    if (x.empty() || y.empty()) {
      throw std::invalid_argument("Empty data vectors");
    }
    if (x.size() != y.size()) {
      throw std::invalid_argument("X and Y vectors must have same size");
    }

    // Check for invalid numbers
    for (size_t i = 0; i < x.size(); ++i) {
      if (!isValidNumber(x[i]) || !isValidNumber(y[i])) {
        throw std::invalid_argument("Invalid number detected in data");
      }
    }
  }

  bool Figure::isValidNumber(float n) const { return !std::isnan(n) && !std::isinf(n); }

  void Canvas::drawLineClipped(float x1, float y1, float x2, float y2, const Pixel &color, float thickness) {
    if (clipLine(x1, y1, x2, y2)) {
      if (thickness <= 1.0f) {
        drawLineAA(x1, y1, x2, y2, color, thickness);
      } else {
        drawLine(x1, y1, x2, y2, color, thickness);
      }
    }
  }

  void Canvas::drawLineAA(float x1, float y1, float x2, float y2, const Pixel &color, float thickness) {
    // Xiaolin Wu's line algorithm with alpha blending
    auto plot = [this, &color](int x, int y, float brightness) {
      if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        Pixel &dst = pixels_[y * width_ + x];
        blendPixel(dst, color, brightness);
      }
    };

    // Implementation of Xiaolin Wu's line algorithm
    bool steep = std::abs(y2 - y1) > std::abs(x2 - x1);
    if (steep) {
      std::swap(x1, y1);
      std::swap(x2, y2);
    }
    if (x1 > x2) {
      std::swap(x1, x2);
      std::swap(y1, y2);
    }

    float dx = x2 - x1;
    float dy = y2 - y1;
    float gradient = (dx == 0) ? 1 : dy / dx;

    // Handle first endpoint
    float xend = std::round(x1);
    float yend = y1 + gradient * (xend - x1);
    float xgap = 1 - std::fmod(x1 + 0.5f, 1.0f);
    int xpxl1 = static_cast<int>(xend);
    int ypxl1 = static_cast<int>(yend);

    if (steep) {
      plot(ypxl1, xpxl1, (1 - std::fmod(yend, 1.0f)) * xgap);
      plot(ypxl1 + 1, xpxl1, std::fmod(yend, 1.0f) * xgap);
    } else {
      plot(xpxl1, ypxl1, (1 - std::fmod(yend, 1.0f)) * xgap);
      plot(xpxl1, ypxl1 + 1, std::fmod(yend, 1.0f) * xgap);
    }

    float intery = yend + gradient;

    // Handle second endpoint
    xend = std::round(x2);
    yend = y2 + gradient * (xend - x2);
    xgap = std::fmod(x2 + 0.5f, 1.0f);
    int xpxl2 = static_cast<int>(xend);
    int ypxl2 = static_cast<int>(yend);

    // Main loop
    if (steep) {
      for (int x = xpxl1 + 1; x < xpxl2; x++) {
        plot(static_cast<int>(intery), x, 1 - std::fmod(intery, 1.0f));
        plot(static_cast<int>(intery) + 1, x, std::fmod(intery, 1.0f));
        intery += gradient;
      }
    } else {
      for (int x = xpxl1 + 1; x < xpxl2; x++) {
        plot(x, static_cast<int>(intery), 1 - std::fmod(intery, 1.0f));
        plot(x, static_cast<int>(intery) + 1, std::fmod(intery, 1.0f));
        intery += gradient;
      }
    }
  }

  static ThreadPool latexThreadPool(std::thread::hardware_concurrency());

  void TextRenderer::renderLatex(Canvas &canvas, const std::string &latexCode, int x, int y, float scale,
                                 const Pixel &textColor) {
    auto &cache = LaTeXCache::getInstance();
    auto cachedPath = cache.getCachedFilePath(latexCode, scale, textColor);

    if (cachedPath) {
      PNGLoader pngLoader(*cachedPath);
      auto pixels = pngLoader.readImageData(textColor);
      canvas.blendImage(pixels, pngLoader.getWidth(), pngLoader.getHeight(), x, y, 1.0f);

      // Ensure the file path exists before adding to SVG
      if (!cachedPath->empty()) {
        std::stringstream svgImage;
        svgImage << "<image x=\"" << x << "\" y=\"" << y << "\" width=\"" << pngLoader.getWidth() << "\" height=\""
                 << pngLoader.getHeight() << "\" xlink:href=\"" << *cachedPath << "\" />";
        canvas.svgElements.push_back({"image", svgImage.str()});
      }

      return;
    }

    // Use thread pool to render LaTeX asynchronously
    latexThreadPool.enqueue([&]() {
      std::string generatedImagePath = generateLatexImage(latexCode, scale);
      cache.store(latexCode, scale, textColor, generatedImagePath);

      PNGLoader pngLoader(generatedImagePath);
      auto pixels = pngLoader.readImageData(textColor);

      modifyCanvas(canvas, pixels, pngLoader.getWidth(), pngLoader.getHeight(), x, y);

      // Ensure the generated file exists before adding it to SVG
      if (!generatedImagePath.empty()) {
        std::stringstream svgImage;
        svgImage << "<image x=\"" << x << "\" y=\"" << y << "\" width=\"" << pngLoader.getWidth() << "\" height=\""
                 << pngLoader.getHeight() << "\" xlink:href=\"" << generatedImagePath << "\" />";
        canvas.svgElements.push_back({"image", svgImage.str()});
      }
    });

    // Alternative: Embed LaTeX as raw SVG text (if text rendering is supported)
    std::stringstream svgText;
    svgText << "<text x=\"" << x << "\" y=\"" << y << "\" font-size=\"" << scale * 20 << "\" fill=\"rgb("
            << (int) textColor.r << "," << (int) textColor.g << "," << (int) textColor.b << ")\">" << latexCode
            << "</text>";

    canvas.svgElements.push_back({"text", svgText.str()});
  }

  void Canvas::addSvgLine(float x1, float y1, float x2, float y2, const Pixel &color, float width,
                          const std::string &cssClass) {
    std::ostringstream ss;
    ss << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" stroke=\"rgb("
       << (int) color.r << "," << (int) color.g << "," << (int) color.b << ")\" stroke-width=\"" << width << "\"";
    if (!cssClass.empty()) {
      ss << " class=\"" << cssClass << "\"";
    }
    ss << " />";
    svgElements.push_back({"line", ss.str()});
  }


  void TextRenderer::modifyCanvas(Canvas &canvas, const std::vector<Pixel> &pixels, int width, int height, int x,
                                  int y) {
    canvas.blendImage(pixels, width, height, x, y, 1.0f);
  }


  void LaTeXCache::cleanupCache(size_t maxCacheSize) {
    std::vector<std::pair<std::string, std::time_t>> files;

    for (const auto &entry: fs::directory_iterator(cacheDir)) {
      files.emplace_back(entry.path().string(), fs::last_write_time(entry.path()).time_since_epoch().count());
    }

    if (files.size() > maxCacheSize) {
      std::sort(files.begin(), files.end(), [](auto &a, auto &b) { return a.second < b.second; });

      for (size_t i = 0; i < files.size() - maxCacheSize; ++i) {
        fs::remove(files[i].first);
      }
    }
  }

  std::string generateLatexImage(const std::string &latexCode, float scale) {
    namespace fs = std::filesystem;

    // Create a unique temporary directory
    static std::atomic<uint64_t> uniqueCounter{0};
    uint64_t currentId = uniqueCounter.fetch_add(1, std::memory_order_relaxed);
    fs::path tempDir = fs::temp_directory_path() / ("LaTeX_" + std::to_string(currentId));
    fs::create_directories(tempDir);

    std::string texFilePath = tempDir.string() + "/equation.tex";
    std::string pdfPath = tempDir.string() + "/equation.pdf";
    std::string pngPath = tempDir.string() + "/equation.png";

    // LaTeX document template
    std::ofstream texFile(texFilePath);
    texFile << "\\documentclass[preview]{standalone}\n"
               "\\usepackage{amsmath,amssymb}\n"
               "\\usepackage[utf8]{inputenc}\n"
               "\\pagestyle{empty}\n"
               "\\begin{document}\n"
            << "$" << latexCode << "$\n"
            << "\\end{document}\n";
    texFile.close();

    // Compile LaTeX to PDF
    std::string pdflatexCmd =
            "pdflatex -interaction=batchmode -halt-on-error -output-directory=" + tempDir.string() + " " + texFilePath;
    CommandExecutor::execute(pdflatexCmd);

    if (!fs::exists(pdfPath)) {
      throw std::runtime_error("LaTeX failed: PDF not generated at " + pdfPath);
    }

    // Convert PDF to PNG
    std::string convertCmd =
            "gs -q -dQUIET -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r300 -o " + pngPath + " " + pdfPath;
    CommandExecutor::execute(convertCmd);

    return pngPath; // Return the path of the generated PNG
  }


  void Canvas::blendImage(const std::vector<Pixel> &image, int imgWidth, int imgHeight, int x, int y, float alpha) {
    // Early return if completely transparent or outside canvas
    if (alpha <= 0.0f || x >= width_ || y >= height_ || x + imgWidth <= 0 || y + imgHeight <= 0) {
      return;
    }

    // Pre-compute constants
    const float invAlpha = 1.0f / 255.0f;

    // Calculate clipping boundaries to avoid bounds checking in inner loop
    const int startY = std::max(0, y);
    const int endY = std::min(height_, y + imgHeight);
    const int startX = std::max(0, x);
    const int endX = std::min(width_, x + imgWidth);

    // Pre-compute image offset based on clipping
    const int imgOffsetX = startX - x;
    const int imgOffsetY = startY - y;

    // Use multi-threading for large images
    const bool useParallel = (endY - startY) * (endX - startX) > 10000;

// SIMD-friendly optimized blending
#pragma omp parallel for if (useParallel)
    for (int canvasY = startY; canvasY < endY; ++canvasY) {
      const int imgY = canvasY - y;
      const int imgRowOffset = imgY * imgWidth;
      const int canvasRowOffset = canvasY * width_;

      // Process 4 pixels at a time where possible for SIMD optimization
      int canvasX = startX;

      // Main SIMD-friendly loop (no bounds checking needed due to pre-calculation)
      for (; canvasX < endX; ++canvasX) {
        const int imgX = canvasX - x;
        const int imgIndex = imgRowOffset + imgX;
        const int canvasIndex = canvasRowOffset + canvasX;

        const Pixel &srcPixel = image[imgIndex];

        // Skip fully transparent pixels
        if (srcPixel.a == 0)
          continue;

        // Fast path for fully opaque pixels
        if (srcPixel.a == 255 && alpha >= 0.99f) {
          pixels_[canvasIndex] = srcPixel;
          continue;
        }

        // Calculate effective alpha once
        const float effectiveAlpha = (srcPixel.a * invAlpha) * alpha;
        const float inverseAlpha = 1.0f - effectiveAlpha;

        // Direct calculation instead of function call for better inlining
        Pixel &destPixel = pixels_[canvasIndex];
        destPixel.r = static_cast<uint8_t>(srcPixel.r * effectiveAlpha + destPixel.r * inverseAlpha);
        destPixel.g = static_cast<uint8_t>(srcPixel.g * effectiveAlpha + destPixel.g * inverseAlpha);
        destPixel.b = static_cast<uint8_t>(srcPixel.b * effectiveAlpha + destPixel.b * inverseAlpha);
        destPixel.a = static_cast<uint8_t>(
                std::min(255.0f, srcPixel.a * alpha + destPixel.a * (1.0f - (srcPixel.a * invAlpha * alpha))));
      }
    }
  }

  bool Canvas::clipLine(float &x1, float &y1, float &x2, float &y2) const {
    // Cohen-Sutherland line clipping algorithm
    const int INSIDE = 0;
    const int LEFT = 1;
    const int RIGHT = 2;
    const int BOTTOM = 4;
    const int TOP = 8;

    auto computeCode = [this](float x, float y) {
      int code = INSIDE;
      if (x < 0)
        code |= LEFT;
      else if (x >= width_)
        code |= RIGHT;
      if (y < 0)
        code |= BOTTOM;
      else if (y >= height_)
        code |= TOP;
      return code;
    };

    int code1 = computeCode(x1, y1);
    int code2 = computeCode(x2, y2);

    while (true) {
      if (!(code1 | code2))
        return true;
      if (code1 & code2)
        return false;

      float x, y;
      int codeOut = code1 ? code1 : code2;

      if (codeOut & TOP) {
        x = x1 + (x2 - x1) * (height_ - y1) / (y2 - y1);
        y = height_ - 1;
      } else if (codeOut & BOTTOM) {
        x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1);
        y = 0;
      } else if (codeOut & RIGHT) {
        y = y1 + (y2 - y1) * (width_ - x1) / (x2 - x1);
        x = width_ - 1;
      } else {
        y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1);
        x = 0;
      }

      if (codeOut == code1) {
        x1 = x;
        y1 = y;
        code1 = computeCode(x1, y1);
      } else {
        x2 = x;
        y2 = y;
        code2 = computeCode(x2, y2);
      }
    }
  }
} // namespace Plot
