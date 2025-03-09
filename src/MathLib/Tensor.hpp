#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

class Tensor {
  public:
  // Constructors
  Tensor(); // Default constructor
  Tensor(const std::vector<size_t> &shape);
  Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);
  Tensor(const Tensor &other); // Copy constructor
  Tensor(Tensor &&other) noexcept; // Move constructor
  ~Tensor(); // Destructor

  // Assignment operators
  Tensor &operator=(const Tensor &other);
  Tensor &operator=(Tensor &&other) noexcept;

  // Tensor operations
  Tensor multiply(const Tensor &other) const; // Element-wise multiplication
  Tensor dot(const Tensor &other) const; // Dot product along specified axes

  // Basic operations
  [[nodiscard]] Tensor operator*(const Tensor &other) const { return multiply(other); };
  Tensor add(const Tensor &other) const; // Element-wise addition
  Tensor subtract(const Tensor &other) const; // Element-wise subtraction
  Tensor scale(float scalar) const; // Scale tensor by a scalar
  Tensor transpose(const std::vector<size_t> &order) const; // Transpose dimensions

  // Accessors
  float &at(const std::vector<size_t> &indices);
  const float &at(const std::vector<size_t> &indices) const;
  float *data();
  const float *data() const;
  std::vector<size_t> shape() const;
  size_t size() const;

  // Comparison
  bool operator==(const Tensor &other) const;
  bool operator!=(const Tensor &other) const;
  float maxDifference(const Tensor &other) const;
  float meanDifference(const Tensor &other) const;

  // Output
  std::string toString() const;
  void print() const;

  private:
  std::vector<size_t> m_shape;
  std::vector<float> m_data;
  std::vector<size_t> m_strides;

  size_t computeIndex(const std::vector<size_t> &indices) const;
  void validateShape(const std::vector<size_t> &shape) const;
  void computeStrides();
};

#endif // TENSOR_H
