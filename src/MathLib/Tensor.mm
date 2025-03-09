#include "Tensor.hpp"
#include "MetalUtils.h"
#include <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

// Reusable Metal device and command queue to avoid recreation overhead
static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> metalCommandQueue = nil;

// Constructor with shape
Tensor::Tensor(const std::vector<size_t> &shape) : m_shape(shape) {
  computeStrides();
  m_data.resize(std::accumulate(m_shape.begin(), m_shape.end(), 1,
                                std::multiplies<size_t>()),
                0.0f);
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data)
    : m_shape(shape) {
  computeStrides();
  if (data.size() != std::accumulate(m_shape.begin(), m_shape.end(), 1,
                                     std::multiplies<size_t>())) {
    throw std::invalid_argument("Data size doesn't match tensor dimensions");
  }
  m_data = data;
}

// Compute strides for indexing
void Tensor::computeStrides() {
  m_strides.resize(m_shape.size());
  m_strides.back() = 1;
  for (int i = m_shape.size() - 2; i >= 0; --i) {
    m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
  }
}

// Compute linear index from multi-dimensional indices
size_t Tensor::computeIndex(const std::vector<size_t> &indices) const {
  size_t index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= m_shape[i]) {
      throw std::out_of_range("Tensor indices out of range");
    }
    index += indices[i] * m_strides[i];
  }
  return index;
}

void Tensor::validateShape(const std::vector<size_t> &shape) const {
  for (size_t dim : shape) {
    if (dim == 0) {
      throw std::invalid_argument("Tensor shape cannot have zero dimensions");
    }
  }
}

size_t Tensor::size() const { return m_data.size(); }

// Destructor
Tensor::~Tensor() {}

// Access element using multi-dimensional indices
float &Tensor::at(const std::vector<size_t> &indices) {
  return m_data[computeIndex(indices)];
}

const float &Tensor::at(const std::vector<size_t> &indices) const {
  return m_data[computeIndex(indices)];
}

// Accelerate BLAS tensor addition
void tensorAddBLAS(const float *A, const float *B, float *C, size_t size) {
  vDSP_vadd(A, 1, B, 1, C, 1, size);
}

// Metal GPU tensor addition
void metalTensorAdd(const float *A, const float *B, float *C, size_t size) {
  // Initialize Metal once if needed
  if (metalDevice == nil) {
    initializeMetal();
    if (metalDevice == nil)
      return; // Exit if Metal initialization failed
  }

  @autoreleasepool {
    // Get the Metal function
    id<MTLFunction> addFunction =
        [metalLibrary newFunctionWithName:@"tensor_add"];
    if (!addFunction) {
      std::cerr << "Error: Failed to load Metal function 'tensor_add'"
                << std::endl;
      return;
    }

    // Create a compute pipeline state
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState =
        [metalDevice newComputePipelineStateWithFunction:addFunction
                                                   error:&error];
    if (!pipelineState) {
      std::cerr << "Error: Failed to create compute pipeline state: "
                << error.localizedDescription.UTF8String << std::endl;
      return;
    }

    // Create buffers for the input and output data
    MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    id<MTLBuffer> bufferA = [metalDevice newBufferWithBytes:A
                                                     length:size * sizeof(float)
                                                    options:options];
    id<MTLBuffer> bufferB = [metalDevice newBufferWithBytes:B
                                                     length:size * sizeof(float)
                                                    options:options];
    id<MTLBuffer> bufferC =
        [metalDevice newBufferWithLength:size * sizeof(float) options:options];

    // Create a command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    // Set the pipeline state and buffers
    [computeEncoder setComputePipelineState:pipelineState];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];

    // Dispatch the compute kernel
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > size) {
      threadGroupSize = size;
    }
    MTLSize threadgroups =
        MTLSizeMake((size + threadGroupSize - 1) / threadGroupSize, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreadgroups:threadgroups
                   threadsPerThreadgroup:threadsPerThreadgroup];

    // End encoding and commit the command buffer
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy result back to host memory
    memcpy(C, bufferC.contents, size * sizeof(float));
  }
}

// Adaptive tensor addition with dynamic selection
void adaptiveTensorAdd(const float *A, const float *B, float *C, size_t size,
                       bool &usedMetal) {
  // Determine the addition approach based on tensor size and past performance
  const size_t METAL_THRESHOLD = 1000000; // Threshold for using Metal

  // If tensor is very large, use Metal
  if (size >= METAL_THRESHOLD) {
    metalTensorAdd(A, B, C, size);
    usedMetal = true;
  } else {
    tensorAddBLAS(A, B, C, size);
    usedMetal = false;
  }
}

// Element-wise addition
Tensor Tensor::add(const Tensor &other) const {
  if (m_shape != other.m_shape) {
    throw std::invalid_argument(
        "Tensor shapes mismatch for element-wise addition");
  }

  Tensor result(m_shape);
  for (size_t i = 0; i < m_data.size(); ++i) {
    result.m_data[i] = m_data[i] + other.m_data[i];
  }
  return result;
}

// Element-wise multiplication
Tensor Tensor::multiply(const Tensor &other) const {
  if (m_shape != other.m_shape) {
    throw std::invalid_argument(
        "Tensor shapes mismatch for element-wise multiplication");
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
                      m_shape.end() -
                          1); // All but the last dimension of this tensor
  result_shape.insert(
      result_shape.end(), other.m_shape.begin() + 1,
      other.m_shape.end()); // All but the first dimension of the other tensor

  // Create the result tensor
  Tensor result(result_shape);

  // Compute the dot product
  size_t inner_dim = m_shape.back(); // Common dimension to sum over
  size_t outer_dim1 = std::accumulate(
      m_shape.begin(), m_shape.end() - 1, 1,
      std::multiplies<size_t>()); // Outer dimensions of this tensor
  size_t outer_dim2 = std::accumulate(
      other.m_shape.begin() + 1, other.m_shape.end(), 1,
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
      throw std::invalid_argument(
          "Order must be a valid permutation of dimensions");
    }
  }

  // Compute the new shape and strides
  std::vector<size_t> new_shape(order.size());
  std::vector<size_t> new_strides(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    new_shape[i] = m_shape[order[i]];
    new_strides[i] =
        std::accumulate(m_shape.begin() + order[i] + 1, m_shape.end(), 1,
                        std::multiplies<size_t>());
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

// Scale tensor by a scalar
Tensor Tensor::scale(float scalar) const {
  Tensor result(m_shape);
  vDSP_vsmul(m_data.data(), 1, &scalar, result.m_data.data(), 1, m_data.size());
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

void Tensor::print() const {
  std::stringstream ss;
  const std::string RESET = "\033[0m";
  const std::string SHAPE_COLOR = "\033[1;38;5;75m";
  const std::string COMMA_COLOR = "\033[1;38;5;250m";
  const std::string ELLIPSIS_COLOR = "\033[38;5;240m";
  const std::string ELEMENT_INFO = "\033[38;5;140m";
  const std::string ZERO_COLOR = "\033[38;5;245m";
  const std::string POSITIVE_COLOR = "\033[32m";
  const std::string NEGATIVE_COLOR = "\033[38;5;174m";

  const std::vector<std::string> DIM_COLORS = {
      "\033[38;5;233m", // Very dark grey
      "\033[38;5;235m", // Dark grey
      "\033[38;5;237m", // Medium-dark grey
      "\033[38;5;240m", // Medium grey
      "\033[38;5;243m", // Light-medium grey
      "\033[38;5;246m", // Light grey
      "\033[38;5;250m"  // Very light grey
  };

  const size_t MAX_ELEMENTS_PER_DIM = 8;
  const int PRECISION = 3;
  const float ZERO_THRESHOLD = 1e-10;

  size_t total_elements = 1;
  for (size_t s : m_shape) {
    total_elements *= s;
  }

  ss << SHAPE_COLOR << "Tensor(";
  for (size_t i = 0; i < m_shape.size(); ++i) {
    ss << m_shape[i];
    if (i < m_shape.size() - 1) {
      ss << COMMA_COLOR << " × " << SHAPE_COLOR;
    }
  }
  ss << ")" << RESET << "  " << ELEMENT_INFO << "containing " << total_elements
     << (total_elements == 1 ? " element" : " elements") << RESET << std::endl;

  std::function<void(size_t, size_t, std::vector<size_t> &, bool)> printTensor;
  printTensor = [&](size_t dim, size_t index, std::vector<size_t> &indices,
                    bool isCompact) {
    if (dim == m_shape.size()) {
      float val = m_data[index];
      if (std::abs(val) < ZERO_THRESHOLD) {
        ss << " " << ZERO_COLOR << "0.0" << std::string(PRECISION - 2, '0')
           << RESET;
      } else if (val > 0) {
        std::stringstream numStr;
        numStr << std::fixed << std::setprecision(PRECISION) << val;
        ss << " " << POSITIVE_COLOR << numStr.str() << RESET;
      } else {
        std::stringstream numStr;
        numStr << std::fixed << std::setprecision(PRECISION) << val;
        ss << NEGATIVE_COLOR << numStr.str() << RESET;
      }
      return;
    }

    std::string bracketColor = DIM_COLORS[dim % DIM_COLORS.size()];

    ss << bracketColor << "【" << RESET;
    size_t elemsToPrint = isCompact && m_shape[dim] > MAX_ELEMENTS_PER_DIM
                              ? MAX_ELEMENTS_PER_DIM / 2
                              : m_shape[dim];

    for (size_t i = 0; i < elemsToPrint; ++i) {
      indices[dim] = i;
      size_t nextIndex = index + i * m_strides[dim];

      printTensor(dim + 1, nextIndex, indices, isCompact);

      if (i < m_shape[dim] - 1) {
        ss << COMMA_COLOR << "," << RESET;

        if (dim == m_shape.size() - 1) {
          ss << " ";
        } else {
          ss << std::endl;
          for (size_t j = 0; j <= dim; ++j) {
            ss << "  ";
          }
        }
      }
    }

    if (isCompact && m_shape[dim] > MAX_ELEMENTS_PER_DIM) {
      ss << COMMA_COLOR << " " << ELLIPSIS_COLOR << "⋯" << RESET;
      size_t startIdx = m_shape[dim] - MAX_ELEMENTS_PER_DIM / 2;
      for (size_t i = startIdx; i < m_shape[dim]; ++i) {
        indices[dim] = i;
        size_t nextIndex = index + i * m_strides[dim];
        ss << COMMA_COLOR << "," << RESET;
        if (dim == m_shape.size() - 1) {
          ss << " ";
        } else {
          ss << std::endl;
          for (size_t j = 0; j <= dim; ++j) {
            ss << "  ";
          }
        }
        printTensor(dim + 1, nextIndex, indices, isCompact);
      }
    }
    ss << bracketColor << "】" << RESET;
  };

  std::vector<size_t> indices(m_shape.size(), 0);

  bool useCompactRepresentation = total_elements > 120;
  printTensor(0, 0, indices, useCompactRepresentation);

  std::cout << ss.str() << '\n';
}