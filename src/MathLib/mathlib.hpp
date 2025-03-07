#ifndef MATHLIB_HPP
#define MATHLIB_HPP

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>


template<typename T>
std::vector<T> operator/(const std::vector<T> &vec, const T &scalar) {
  std::vector<T> result;
  result.reserve(vec.size());
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), [&](const T &val) {
    if (std::abs(val) < std::numeric_limits<T>::epsilon()) { // Use epsilon for floating-point comparisons
      throw std::runtime_error("Division by zero or near-zero value!");
    }
    return scalar / val;
  });
  return result;
}

template<typename T>
std::vector<T> operator/(const T &scalar, const std::vector<T> &vec) {
  std::vector<T> result;
  result.reserve(vec.size());
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), [&](const T &val) {
    if (std::abs(val) < std::numeric_limits<T>::epsilon()) { // Use epsilon for floating-point comparisons
      throw std::runtime_error("Division by zero or near-zero value!");
    }
    return scalar / val;
  });
  return result;
}

template<typename T>
std::vector<T> operator/(const std::vector<T> &vec1, const std::vector<T> &vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vector sizes must match for division.");
  }
  std::vector<T> result;
  result.reserve(vec1.size());
  std::transform(vec1.begin(), vec1.end(), vec2.begin(), std::back_inserter(result),
                 std::divides<T>()); // Use std::divides for generality
  return result;
}


template<typename T>
std::vector<T> operator*(const std::vector<T> &vec1, const std::vector<T> &vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vector sizes must match for multiplication.");
  }
  std::vector<T> result;
  result.reserve(vec1.size());
  std::transform(vec1.begin(), vec1.end(), vec2.begin(), std::back_inserter(result),
                 std::multiplies<T>()); // Use std::multiplies
  return result;
}

template<typename T>
std::vector<T> operator*(const T &scalar, const std::vector<T> &vec) {
  std::vector<T> result;
  result.reserve(vec.size());
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), [&](const T &val) { return val * scalar; });
  return result;
}

template<typename T>
std::vector<T> operator*(const std::vector<T> &vec, const T &scalar) {
  return scalar * vec; // Reuse the above operator* implementation
}


template<typename T>
std::vector<T> operator+(const std::vector<T> &vec, const T &scalar) {
  std::vector<T> result;
  result.reserve(vec.size());
  std::transform(vec.begin(), vec.end(), std::back_inserter(result), [&](const T &val) { return val + scalar; });
  return result;
}

template<typename T>
std::vector<T> operator+(const T &scalar, const std::vector<T> &vec) {
  return vec + scalar; // Reuse the above operator+ implementation
}

namespace Math {
  void printVec3(const std::vector<std::vector<std::vector<int>>> &Vec3);
  void printVec2(const std::vector<std::vector<int>> &Vec2);
  void printVec(const std::vector<float> &vec);
  std::vector<float> linspace(float start, float end, int step);
  std::vector<float> Sin(const std::vector<float> &vec);
  std::vector<float> Cos(const std::vector<float> &vec);
  std::vector<float> Tan(const std::vector<float> &vec);
  std::vector<float> Log(const std::vector<float> &vec);
  std::vector<float> Log10(const std::vector<float> &vec);
  std::vector<float> Log2(const std::vector<float> &vec);
  std::vector<float> Log(const std::vector<float> &vec, float base);
  std::vector<float> ArcSin(const std::vector<float> &vec);
  std::vector<float> ArcCos(const std::vector<float> &vec);
  std::vector<float> ArcTan(const std::vector<float> &vec);
  std::vector<float> Sinh(const std::vector<float> &vec);
  std::vector<float> Cosh(const std::vector<float> &vec);
  std::vector<float> Tanh(const std::vector<float> &vec);
  std::vector<float> Exp(const std::vector<float> &vec);
  std::vector<float> Pow(const std::vector<float> &vec, float exponent);
  std::vector<float> Abs(const std::vector<float> &vec);
  std::vector<float> Sqrt(const std::vector<float> &vec);
  std::vector<float> Round(const std::vector<float> &vec);
  std::vector<float> Floor(const std::vector<float> &vec);
  std::vector<float> Ceil(const std::vector<float> &vec);
  std::vector<float> Fmod(const std::vector<float> &vec1, const std::vector<float> &vec2);
  std::vector<float> RandomUniform(int size, float min = 0.0f, float max = 1.0f);
  std::vector<float> RandomNormal(int size, float mean = 0.0f, float stddev = 1.0f);
  float Sum(const std::vector<float> &vec);
  float Mean(const std::vector<float> &vec);
  float Variance(const std::vector<float> &vec);
  float StdDev(const std::vector<float> &vec);
  float Median(std::vector<float> vec);
  template<typename T>
  std::vector<T> Concatenate(const std::vector<T> &vec1, const std::vector<T> &vec2);
  std::vector<float> SinDeg(const std::vector<float> &vec);
  template<typename T>
  T Min(const std::vector<T> &vec);
  template<typename T>
  T Max(const std::vector<T> &vec);
  std::vector<float> Normalize(const std::vector<float> &vec);
  std::vector<float> TrapezoidalIntegration(const std::vector<float> &y, const std::vector<float> &x);
} // namespace Math

#endif
