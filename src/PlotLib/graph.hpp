#ifndef GENERATE_IMAGE_GRAPH_HPP
#define GENERATE_IMAGE_GRAPH_HPP

#include <freetype2/ft2build.h>
#include <unordered_map>
#include <vector>
#include FT_FREETYPE_H

namespace Plot {
  std::string generateLatexImage(const std::string &latexCode, float scale);
  struct Pixel {
    uint8_t r, g, b, a;
    Pixel() : r(0), g(0), b(0), a(255) {}
    Pixel(const uint8_t r, const uint8_t g, const uint8_t b, const uint8_t a = 255) : r(r), g(g), b(b), a(a) {}
    static const Pixel Black;
    static const Pixel White;
    static const Pixel Red;
    static const Pixel Green;
    static const Pixel Blue;
    static const Pixel Gray;
    static const Pixel DarkGray;
    static const Pixel LightGray;
    static const Pixel Cyan;
    static const Pixel Magenta;
    static const Pixel Yellow;
    static const Pixel Orange;
    static const Pixel Purple;
    static const Pixel Pink;
    static const Pixel Brown;
    static const Pixel Lime;
    static const Pixel Olive;
    static const Pixel Teal;
    static const Pixel Navy;
    static const Pixel Maroon;
    static const Pixel Gold;
    static const Pixel Beige;
    static const Pixel Salmon;
    static const Pixel Turquoise;
    static const Pixel Lavender;
    static const Pixel Indigo;
    static const Pixel Violet;
    static const Pixel SkyBlue;
    static const Pixel Coral;
    static const Pixel Mint;
    static const Pixel Transparent;
  };
  struct AxisProperties {
    bool visible = true;
    bool showTicks = true;
    bool showLabels = true;
    std::u32string label;
    Pixel color = Pixel::Black;
    float thickness = 1.0f;
    float tickLength = 5.0f;
    float tickSpacing = 1.0f;
    std::u32string labelFormat = U"%.1f";
  };

  struct Viewport {
    float xMin, xMax, yMin, yMax;
    [[nodiscard]] bool contains(const float x, const float y) const {
      return x >= xMin && x <= xMax && y >= yMin && y <= yMax;
    }
  };

  class Canvas {
public:
    Canvas(int width, int height, const Pixel &background);
    void drawLine(float x1, float y1, float x2, float y2, const Pixel &color, float thickness = 1.0f);
    void setPixel(int x, int y, const Pixel &color);
    void clear(const Pixel &color);
    void blendImage(const std::vector<Pixel> &image, int imgWidth, int imgHeight, int x, int y, float alpha = 1.0f);
    void drawRect(int x, int y, int width, int height, const Pixel &color, float thickness = 1.0f);
    Pixel &getPixelRef(int x, int y) { return pixels_[y * width_ + x]; } // Add this method
    void drawLineClipped(float x1, float y1, float x2, float y2, const Pixel &color, float thickness = 1.0f);
    bool clipLine(float &x1, float &y1, float &x2, float &y2) const;

    [[nodiscard]] int getWidth() const { return width_; }
    [[nodiscard]] int getHeight() const { return height_; }
    [[nodiscard]] const std::vector<Pixel> &getPixels() const { return pixels_; }

private:
    int width_;
    int height_;
    std::vector<Pixel> pixels_;
    void drawLineAA(float x1, float y1, float x2, float y2, const Pixel &color, float thickness);
  };

  class TextRenderer {
public:
    static void initialize(const std::string &fontPath);
    static void shutdown();
    static void renderText(Canvas &canvas, const std::u32string &text, int x, int y, const Pixel &color,
                           float scale = 1.0f, float angleDegrees = 0.0f);
    static std::pair<int, int> getTextSize(const std::u32string &text, float scale = 1.0f);
    static void renderLatex(Canvas &canvas, const std::string &latexCode, int x, int y, float scale = 1.0f,
                            const Pixel &textColor = Pixel::Black);
    static std::vector<Pixel> loadImage(const std::string &filename);
    static void modifyCanvas(Canvas &canvas, const std::vector<Pixel> &pixels, int width, int height, int x, int y);

private:
    struct GlyphInfo {
      std::vector<unsigned char> bitmap;
      int width, height;
      int bearingX, bearingY;
      int advance;
    };

    static FT_Library library_;
    static FT_Face face_;
    static std::unordered_map<char32_t, GlyphInfo> glyphCache_;
    static GlyphInfo &getCachedGlyph(char32_t c);
  };

  struct PlotStyle {
    Pixel color = Pixel::Blue;
    float lineWidth = 1.0f;
    enum class Type { Line, Scatter, Both } plotType = Type::Line;
    float pointSize = 5.0f;
    enum class Marker { None, Circle, Square, Triangle };
    Marker marker = Marker::None;
    float markerSize = 5.0f;
  };

  struct GridProperties {
    bool visible = true;
    Pixel color = Pixel{200, 200, 200, 255};
    int spacing = 50;
    float lineThickness = 0.5f;
  };

  class Figure {
public:
    Figure(int width, int height);
    void setXRange(float min, float max);
    void setYRange(float min, float max);
    void setGridProperties(const GridProperties &props);
    [[nodiscard]] int worldToPixelX(float x) const;
    [[nodiscard]] int worldToPixelY(float y) const;
    void addLatexAnnotation(const std::string &latex, float x, float y, float scale = 1.0f,
                            const Pixel &color = Pixel(0, 0, 0, 255)) {
      LatexAnnotation annotation;
      annotation.latex = latex;
      annotation.x = x;
      annotation.y = y;
      annotation.scale = scale;
      annotation.color = color;
      latexAnnotations.push_back(annotation);
    }
    static std::u32string formatNumber(float value, const std::u32string &format = U"%.1f");
    static void drawPoint(Canvas &canvas, float x, float y, const Pixel &color, float size);
    void addPlot(const std::vector<float> &x, const std::vector<float> &y, const PlotStyle &style = PlotStyle());

    [[nodiscard]] Canvas render() const;
    void setAxisProperties(const AxisProperties &xProps, const AxisProperties &yProps);
    void drawAxes(Canvas &canvas) const;

    struct Padding {
      int left = 60; // Space for y-axis labels
      int right = 20; // Space on right edge
      int top = 20; // Space above plot
      int bottom = 40; // Space for x-axis labels
    };
    void setPadding(const Padding &padding) { padding_ = padding; }

private:
    void drawGrid(Canvas &canvas) const;
    void drawPlot(Canvas &canvas, const std::vector<float> &x, const std::vector<float> &y,
                  const PlotStyle &style) const;
    void validateData(const std::vector<float> &x, const std::vector<float> &y) const;
    static bool isValidNumber(float n);
    static void drawMarker(Canvas &canvas, float x, float y, const Pixel &color, PlotStyle::Marker marker, float size);
    struct LatexAnnotation {
      std::string latex;
      float x{};
      float y{};
      float scale{};
      Pixel color;
    };

    std::vector<LatexAnnotation> latexAnnotations;
    int width_;
    int height_;
    std::pair<float, float> xRange_ = {0.0f, 0.0f};
    std::pair<float, float> yRange_ = {0.0f, 0.0f};
    GridProperties gridProps_;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> plots_;
    std::vector<PlotStyle> plotStyles_;
    AxisProperties xAxisProps_;
    AxisProperties yAxisProps_;
    Viewport viewport_{};
    Padding padding_;
    [[nodiscard]] int getPlotWidth() const { return width_ - padding_.left - padding_.right; }
    [[nodiscard]] int getPlotHeight() const { return height_ - padding_.top - padding_.bottom; }
    [[nodiscard]] int getPlotX() const { return padding_.left; }
    [[nodiscard]] int getPlotY() const { return padding_.top; }
  };

  class PngGenerator {
public:
    static void savePNG(const Canvas &canvas, const std::string &filename);
  };
} // namespace Plot

#endif
