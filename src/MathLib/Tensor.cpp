#include "Tensor.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// Constructor with shape
Tensor::Tensor(const std::vector<size_t> &shape) : m_shape(shape) {
  validateShape(shape);
  m_data.resize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), 0.0f);
}

void Tensor::validateShape(const std::vector<size_t> &shape) const {
  for (size_t dim: shape) {
    if (dim == 0) {
      throw std::invalid_argument("Tensor shape cannot have zero dimensions");
    }
  }
}

size_t Tensor::size() const { return m_data.size(); }

// Constructor with shape and data
Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data) : m_shape(shape) {
  validateShape(shape);
  if (data.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) {
    throw std::invalid_argument("Data size doesn't match tensor dimensions");
  }
  m_data = data;
}

// Destructor
Tensor::~Tensor() {}

// Element-wise multiplication
Tensor Tensor::multiply(const Tensor &other) const {
  if (m_shape != other.m_shape) {
    throw std::invalid_argument("Tensor shapes mismatch for element-wise multiplication");
  }

  Tensor result(m_shape);
  for (size_t i = 0; i < m_data.size(); ++i) {
    result.m_data[i] = m_data[i] * other.m_data[i];
  }
  return result;
}

// Dot product along specified axes
Tensor Tensor::dot(const Tensor &other) const {
  // Check if the last dimension of this tensor matches the first dimension of
  // the other tensor
  if (m_shape.back() != other.m_shape.front()) {
    throw std::invalid_argument("Tensor shapes mismatch for dot product");
  }

  // Compute the resulting shape
  std::vector<size_t> result_shape;
  result_shape.insert(result_shape.end(), m_shape.begin(),
                      m_shape.end() - 1); // All but the last dimension of this tensor
  result_shape.insert(result_shape.end(), other.m_shape.begin() + 1,
                      other.m_shape.end()); // All but the first dimension of the other tensor

  // Create the result tensor
  Tensor result(result_shape);

  // Compute the dot product
  size_t inner_dim = m_shape.back(); // Common dimension to sum over
  size_t outer_dim1 = std::accumulate(m_shape.begin(), m_shape.end() - 1, 1,
                                      std::multiplies<size_t>()); // Outer dimensions of this tensor
  size_t outer_dim2 = std::accumulate(other.m_shape.begin() + 1, other.m_shape.end(), 1,
                                      std::multiplies<size_t>()); // Outer dimensions of the other tensor

  for (size_t i = 0; i < outer_dim1; ++i) {
    for (size_t j = 0; j < outer_dim2; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < inner_dim; ++k) {
        sum += m_data[i * inner_dim + k] * other.m_data[k * outer_dim2 + j];
      }
      result.m_data[i * outer_dim2 + j] = sum;
    }
  }

  return result;
}

// Transpose dimensions
Tensor Tensor::transpose(const std::vector<size_t> &order) const {
  // Validate the order
  if (order.size() != m_shape.size()) {
    throw std::invalid_argument("Order size must match tensor rank");
  }

  // Check if the order is a valid permutation of dimensions
  std::vector<size_t> sorted_order = order;
  std::sort(sorted_order.begin(), sorted_order.end());
  for (size_t i = 0; i < sorted_order.size(); ++i) {
    if (sorted_order[i] != i) {
      throw std::invalid_argument("Order must be a valid permutation of dimensions");
    }
  }

  // Compute the new shape and strides
  std::vector<size_t> new_shape(order.size());
  std::vector<size_t> new_strides(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    new_shape[i] = m_shape[order[i]];
    new_strides[i] = std::accumulate(m_shape.begin() + order[i] + 1, m_shape.end(), 1, std::multiplies<size_t>());
  }

  // Create the result tensor
  Tensor result(new_shape);

  // Compute the transposed data
  std::vector<size_t> indices(new_shape.size(), 0);
  for (size_t i = 0; i < result.size(); ++i) {
    // Compute the original index
    size_t original_index = 0;
    for (size_t j = 0; j < order.size(); ++j) {
      original_index += indices[j] * new_strides[j];
    }

    // Copy the value
    result.m_data[i] = m_data[original_index];

    // Update indices
    for (int j = indices.size() - 1; j >= 0; --j) {
      if (++indices[j] < new_shape[j]) {
        break;
      }
      indices[j] = 0;
    }
  }

  return result;
}

// String representation
std::string Tensor::toString() const {
  std::stringstream ss;
  ss << "Tensor(";
  for (size_t i = 0; i < m_shape.size(); ++i) {
    ss << m_shape[i];
    if (i < m_shape.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")" << std::endl;

  // For large tensors, only show a preview
  const size_t maxPreviewSize = 10;
  const bool showPreview = m_data.size() > maxPreviewSize;

  if (m_data.empty()) {
    ss << "[Empty]";
    return ss.str();
  }

  size_t elementsToShow = showPreview ? maxPreviewSize : m_data.size();
  for (size_t i = 0; i < elementsToShow; ++i) {
    ss << std::fixed << std::setprecision(4) << m_data[i];
    if (i < elementsToShow - 1) {
      ss << ", ";
    }
  }

  if (showPreview) {
    ss << ", ...";
  }

  return ss.str();
}
