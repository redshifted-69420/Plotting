#include "mathlib.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace Math {
  void printVec3(const std::vector<std::vector<std::vector<int>>> &Vec3) {
    std::cout << "[\n";
    for (size_t i = 0; i < Vec3.size(); ++i) {
      std::cout << "  [\n";
      for (const auto &j: Vec3[i]) {
        std::cout << "    [";
        for (size_t k = 0; k < j.size(); ++k) {
          std::cout << j[k] << (k < j.size() - 1 ? ", " : "");
        }
        std::cout << "]\n";
      }
      std::cout << "  ]" << (i < Vec3.size() - 1 ? "," : "") << "\n";
    }
    std::cout << "]\n";
  }

  void printVec2(const std::vector<std::vector<int>> &Vec2) {
    std::cout << "[\n";
    for (const auto &row: Vec2) {
      std::cout << "  [";
      for (size_t j = 0; j < row.size(); ++j) {
        std::cout << row[j] << (j < row.size() - 1 ? ", " : "");
      }
      std::cout << "]\n";
    }
    std::cout << "]\n";
  }

  void printVec(const std::vector<float> &vec) {
    std::cout << "[";
    bool first = true;
    for (const float val: vec) {
      if (!first) {
        std::cout << ", ";
      }
      std::cout << val;
      first = false;
    }
    std::cout << "]\n";
  }

  std::vector<float> linspace(const float start, const float end, const int step) {
    if (step < 2) {
      throw std::invalid_argument("Number of points must be at least 2");
    }

    std::vector<float> result;
    result.reserve(step);

    const float h = (end - start) / static_cast<float>(step - 1);
    for (int i = 0; i < step; ++i) {
      result.push_back(start + static_cast<float>(i) * h);
    }

    return result;
  }

  std::vector<float> Sin(const std::vector<float> &x) {
    std::vector<float> result;
    result.reserve(x.size());

    for (const float val: x) {
      result.push_back(std::sin(val));
    }

    return result;
  }

  std::vector<float> Cos(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::cos(val));
    }
    return result;
  }
  std::vector<float> Tan(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::tan(val));
    }
    return result;
  }
  std::vector<float> Log(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::log(val));
    }
    return result;
  }
  std::vector<float> Log10(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::log10(val));
    }
    return result;
  }
  std::vector<float> Log2(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::log2(val));
    }
    return result;
  }
  std::vector<float> Log(const std::vector<float> &vec, const float base) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::log(val) / std::log(base));
    }
    return result;
  }
  std::vector<float> ArcTan(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::atan(val));
    }
    return result;
  }
  std::vector<float> ArcSin(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::asin(val));
    }
    return result;
  }

  std::vector<float> ArcCos(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    for (const float val: vec) {
      result.push_back(std::acos(val));
    }
    return result;
  }

  std::vector<float> Sinh(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::sinh(x); });
    return result;
  }

  std::vector<float> Cosh(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::cosh(x); });
    return result;
  }

  std::vector<float> Tanh(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::tanh(x); });
    return result;
  }

  std::vector<float> Exp(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::exp(x); });
    return result;
  }

  std::vector<float> Pow(const std::vector<float> &vec, float exponent) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [&](float x) { return std::pow(x, exponent); });
    return result;
  }

  std::vector<float> Abs(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::abs(x); });
    return result;
  }

  std::vector<float> Sqrt(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) {
      if (x < 0) {
        throw std::domain_error("Cannot take square root of negative number.");
      }
      return std::sqrt(x);
    });
    return result;
  }

  std::vector<float> Round(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::round(x); });
    return result;
  }

  std::vector<float> Floor(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::floor(x); });
    return result;
  }

  std::vector<float> Ceil(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result), [](float x) { return std::ceil(x); });
    return result;
  }

  std::vector<float> Fmod(const std::vector<float> &vec1, const std::vector<float> &vec2) {
    if (vec1.size() != vec2.size()) {
      throw std::invalid_argument("Vector sizes must match for Fmod.");
    }
    std::vector<float> result;
    result.reserve(vec1.size());

    std::transform(vec1.begin(), vec1.end(), vec2.begin(), std::back_inserter(result),
                   [](const float a, const float b) { return std::fmod(a, b); });

    return result;
  }

  std::vector<float> RandomUniform(const int size, const float min, const float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    std::vector<float> result(size);
    std::ranges::generate(result, [&]() { return dis(gen); });
    return result;
  }

  std::vector<float> RandomNormal(const int size, const float mean, const float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(mean, stddev);

    std::vector<float> result(size);
    std::ranges::generate(result, [&]() { return dis(gen); });
    return result;
  }

  float Sum(const std::vector<float> &vec) { return std::accumulate(vec.begin(), vec.end(), 0.0f); }

  float Mean(const std::vector<float> &vec) {
    if (vec.empty()) {
      throw std::invalid_argument("Cannot calculate mean of an empty vector.");
    }
    return Sum(vec) / static_cast<float>(vec.size());
  }

  float Variance(const std::vector<float> &vec) {
    if (vec.size() < 2) {
      throw std::invalid_argument("Variance requires at least two elements.");
    }
    float mean = Mean(vec);
    const float sq_sum = std::accumulate(vec.begin(), vec.end(), 0.0f,
                                   [mean](const float acc, const float x) { return acc + std::pow(x - mean, 2); });
    return sq_sum / static_cast<float>(vec.size() - 1);
  }


  float StdDev(const std::vector<float> &vec) { return std::sqrt(Variance(vec)); }

  float Median(std::vector<float> vec) {
    std::ranges::sort(vec);
    if (const int n = static_cast<int>(vec.size()); n % 2 == 0) {
      return (vec[n / 2 - 1] + vec[n / 2]) / 2.0f;
    } else {
      return vec[n / 2];
    }
  }

  template<typename T>
  std::vector<T> Concatenate(const std::vector<T> &vec1, const std::vector<T> &vec2) {
    std::vector<T> result = vec1;
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
  }


  std::vector<float> SinDeg(const std::vector<float> &vec) {
    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result),
                   [](const float x) { return std::sin(x * M_PI / 180.0f); });
    return result;
  }
  template<typename T>
  T Min(const std::vector<T> &vec) {
    if (vec.empty()) {
      throw std::invalid_argument("Cannot find min of an empty vector.");
    }
    return *std::min_element(vec.begin(), vec.end());
  }

  template<typename T>
  T Max(const std::vector<T> &vec) {
    if (vec.empty()) {
      throw std::invalid_argument("Cannot find max of an empty vector.");
    }
    return *std::max_element(vec.begin(), vec.end());
  }

  std::vector<float> Normalize(const std::vector<float> &vec) {
    if (vec.empty()) {
      throw std::invalid_argument("Cannot normalize an empty vector.");
    }
    const float minVal = Min(vec);
    const float maxVal = Max(vec);
    if (std::abs(maxVal - minVal) < std::numeric_limits<float>::epsilon()) {
      throw std::runtime_error("Cannot normalize a vector with identical elements."); // Or return a vector of zeros
    }

    std::vector<float> result;
    result.reserve(vec.size());
    std::ranges::transform(vec, std::back_inserter(result),
                   [&](const float x) { return (x - minVal) / (maxVal - minVal); });
    return result;
  }

  std::vector<float> TrapezoidalIntegration(const std::vector<float> &y, const std::vector<float> &x) {
    if (y.size() != x.size() || y.empty()) {
      throw std::invalid_argument("Invalid input for Trapezoidal Integration");
    }

    std::vector<float> result;
    float integral = 0.0f;
    for (size_t i = 0; i < y.size() - 1; ++i) {
      integral += (y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2.0f;
      result.push_back(integral);
    }

    return result;
  }
} // namespace Math
