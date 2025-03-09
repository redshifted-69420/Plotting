#include "Matrix.hpp"
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
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

// Reusable Metal device and command queue to avoid recreation overhead
static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> metalCommandQueue = nil;
static id<MTLLibrary> metalLibrary = nil;

// Initialize Metal resources once
void initializeMetal() {
  if (metalDevice == nil) {
    metalDevice = MTLCreateSystemDefaultDevice();
    if (!metalDevice) {
      std::cerr << "Error: Metal is not supported on this device!" << std::endl;
      return;
    }
    metalCommandQueue = [metalDevice newCommandQueue];

    // Load the Metal library
    NSError *error = nil;
    metalLibrary = [metalDevice newDefaultLibrary];
    if (!metalLibrary) {
      std::cerr << "Error: Failed to load Metal library!" << std::endl;
      return;
    }
  }
}

void matrixMultiplyBLAS(const float *A, const float *B, float *C, int M, int N,
                        int K) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B,
              N, 0.0f, C, N);
}

// Improved Metal GPU matrix multiplication with reused resources
void metalMatrixMultiply(const float *A, const float *B, float *C, int M, int N,
                         int K) {
  // Initialize Metal once if needed
  if (metalDevice == nil) {
    initializeMetal();
    if (metalDevice == nil)
      return; // Exit if Metal initialization failed
  }

  @autoreleasepool {
    // Create matrix multiplication kernel
    MPSMatrixMultiplication *matMul =
        [[MPSMatrixMultiplication alloc] initWithDevice:metalDevice
                                          transposeLeft:false
                                         transposeRight:false
                                             resultRows:M
                                          resultColumns:N
                                        interiorColumns:K
                                                  alpha:1.0
                                                   beta:0.0];

    // Create buffers with optimal alignment and usage
    MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;

    id<MTLBuffer> bufferA =
        [metalDevice newBufferWithBytes:A
                                 length:M * K * sizeof(float)
                                options:options];
    id<MTLBuffer> bufferB =
        [metalDevice newBufferWithBytes:B
                                 length:K * N * sizeof(float)
                                options:options];
    id<MTLBuffer> bufferC =
        [metalDevice newBufferWithLength:M * N * sizeof(float) options:options];

    // Create matrix descriptors
    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                              columns:K
                                             rowBytes:K * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                              columns:N
                                             rowBytes:N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    // Create MPS matrices
    MPSMatrix *metalA =
        [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix *metalB =
        [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix *metalC =
        [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

    // Execute computation
    id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
    [matMul encodeToCommandBuffer:commandBuffer
                       leftMatrix:metalA
                      rightMatrix:metalB
                     resultMatrix:metalC];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy result back to host memory
    memcpy(C, bufferC.contents, M * N * sizeof(float));
  }
}

void adaptiveMatrixMultiply(const float *A, const float *B, float *C, int M,
                            int N, int K, bool &usedMetal) {
  const int METAL_THRESHOLD = 10000;

  // If matrix is very large, use Metal
  if (std::max({M, N, K}) >= METAL_THRESHOLD) {
    metalMatrixMultiply(A, B, C, M, N, K);
    usedMetal = true;
  } else {
    matrixMultiplyBLAS(A, B, C, M, N, K);
    usedMetal = false;
  }
}

Matrix::Matrix() : m_rows(0), m_cols(0), m_data() {}

Matrix::Matrix(size_t rows, size_t cols)
    : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0f) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float> &data)
    : m_rows(rows), m_cols(cols) {
  if (data.size() != rows * cols) {
    throw std::invalid_argument("Data size doesn't match matrix dimensions");
  }
  m_data = data;
}

Matrix::Matrix(size_t rows, size_t cols, float *data)
    : m_rows(rows), m_cols(cols), m_data(data, data + rows * cols) {}

Matrix::Matrix(const Matrix &other)
    : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {}

Matrix::Matrix(Matrix &&other) noexcept
    : m_rows(other.m_rows), m_cols(other.m_cols),
      m_data(std::move(other.m_data)) {
  other.m_rows = 0;
  other.m_cols = 0;
}

Matrix::~Matrix() {}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_data = other.m_data;
  }
  return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
  if (this != &other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_data = std::move(other.m_data);
    other.m_rows = 0;
    other.m_cols = 0;
  }
  return *this;
}

Matrix Matrix::multiply(const Matrix &other) const {
  return multiplyAdaptive(other);
}

Matrix Matrix::multiplyBLAS(const Matrix &other) const {
  if (m_cols != other.m_rows) {
    throw std::invalid_argument(
        "Matrix dimensions mismatch for multiplication");
  }

  Matrix result(m_rows, other.m_cols);
  matrixMultiplyBLAS(m_data.data(), other.m_data.data(), result.m_data.data(),
                     m_rows, other.m_cols, m_cols);
  return result;
}

Matrix Matrix::multiplyMetal(const Matrix &A, const Matrix &B) const {
  if (A.cols() != B.rows()) {
    throw std::invalid_argument(
        "Matrix dimensions mismatch for multiplication");
  }

  // Initialize Metal once if needed
  if (metalDevice == nil) {
    initializeMetal();
    if (metalDevice == nil) {
      throw std::runtime_error("Metal is not supported on this device!");
    }
  }

  Matrix result(A.rows(), B.cols());

  @autoreleasepool {
    // Create buffers for the input and output matrices
    MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;

    id<MTLBuffer> bufferA =
        [metalDevice newBufferWithBytes:A.data()
                                 length:A.size() * sizeof(float)
                                options:options];
    id<MTLBuffer> bufferB =
        [metalDevice newBufferWithBytes:B.data()
                                 length:B.size() * sizeof(float)
                                options:options];
    id<MTLBuffer> bufferC =
        [metalDevice newBufferWithLength:result.size() * sizeof(float)
                                 options:options];

    // Create buffers for the matrix dimensions
    uint M = static_cast<uint>(A.rows());
    uint N = static_cast<uint>(B.cols());
    uint K = static_cast<uint>(A.cols());

    id<MTLBuffer> bufferM =
        [metalDevice newBufferWithBytes:&M length:sizeof(uint) options:options];
    id<MTLBuffer> bufferN =
        [metalDevice newBufferWithBytes:&N length:sizeof(uint) options:options];
    id<MTLBuffer> bufferK =
        [metalDevice newBufferWithBytes:&K length:sizeof(uint) options:options];

    // Load the Metal kernel function
    id<MTLFunction> kernelFunction =
        [metalLibrary newFunctionWithName:@"matrix_mul"];
    if (!kernelFunction) {
      throw std::runtime_error("Failed to load Metal function 'matrix_mul'");
    }

    // Create a compute pipeline state
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState =
        [metalDevice newComputePipelineStateWithFunction:kernelFunction
                                                   error:&error];
    if (!pipelineState) {
      throw std::runtime_error(
          "Failed to create Metal pipeline state: " +
          std::string([[error localizedDescription] UTF8String]));
    }

    // Create a command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    // Set the pipeline state and buffers
    [computeEncoder setComputePipelineState:pipelineState];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferM offset:0 atIndex:3];
    [computeEncoder setBuffer:bufferN offset:0 atIndex:4];
    [computeEncoder setBuffer:bufferK offset:0 atIndex:5];

    // Dispatch the kernel
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroups = MTLSizeMake((M + 15) / 16, (N + 15) / 16, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);

    [computeEncoder dispatchThreadgroups:threadgroups
                   threadsPerThreadgroup:threadsPerThreadgroup];

    // End encoding and commit the command buffer
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy the result back to host memory
    memcpy(result.data(), [bufferC contents], result.size() * sizeof(float));
  }

  return result;
}

// Adaptive variant
Matrix Matrix::multiplyAdaptive(const Matrix &other) const {
  if (m_cols != other.m_rows) {
    throw std::invalid_argument(
        "Matrix dimensions mismatch for multiplication");
  }

  Matrix result(m_rows, other.m_cols);
  bool usedMetal = false;
  adaptiveMatrixMultiply(m_data.data(), other.m_data.data(),
                         result.m_data.data(), m_rows, other.m_cols, m_cols,
                         usedMetal);
  return result;
}

// Transpose
// Matrix Matrix::transpose() const {
//   Matrix result(m_cols, m_rows);
//   for (size_t i = 0; i < m_rows; ++i) {
//     for (size_t j = 0; j < m_cols; ++j) {
//       result.at(j, i) = at(i, j);
//     }
//   }
//   return result;
// }

Matrix Matrix::transposeMetal() const {
  // Initialize Metal once if needed
  if (metalDevice == nil) {
    initializeMetal();
    if (metalDevice == nil) {
      throw std::runtime_error("Metal is not supported on this device!");
    }
  }

  // Create the result matrix (transposed dimensions)
  Matrix result(m_cols, m_rows);

  @autoreleasepool {
    // Create buffers for the input and output matrices
    MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;

    id<MTLBuffer> bufferInput =
        [metalDevice newBufferWithBytes:m_data.data()
                                 length:m_data.size() * sizeof(float)
                                options:options];
    id<MTLBuffer> bufferOutput =
        [metalDevice newBufferWithLength:result.size() * sizeof(float)
                                 options:options];

    // Create buffers for the matrix dimensions
    uint rows = static_cast<uint>(m_rows);
    uint cols = static_cast<uint>(m_cols);

    id<MTLBuffer> bufferRows = [metalDevice newBufferWithBytes:&rows
                                                        length:sizeof(uint)
                                                       options:options];
    id<MTLBuffer> bufferCols = [metalDevice newBufferWithBytes:&cols
                                                        length:sizeof(uint)
                                                       options:options];

    // Load the Metal kernel function
    id<MTLFunction> kernelFunction =
        [metalLibrary newFunctionWithName:@"matrix_transpose"];
    if (!kernelFunction) {
      throw std::runtime_error(
          "Failed to load Metal function 'matrix_transpose'");
    }

    // Create a compute pipeline state
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState =
        [metalDevice newComputePipelineStateWithFunction:kernelFunction
                                                   error:&error];
    if (!pipelineState) {
      throw std::runtime_error(
          "Failed to create Metal pipeline state: " +
          std::string([[error localizedDescription] UTF8String]));
    }

    // Create a command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    // Set the pipeline state and buffers
    [computeEncoder setComputePipelineState:pipelineState];
    [computeEncoder setBuffer:bufferInput offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferOutput offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferRows offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferCols offset:0 atIndex:3];

    // Dispatch the kernel
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroups = MTLSizeMake((cols + 15) / 16, (rows + 15) / 16, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);

    [computeEncoder dispatchThreadgroups:threadgroups
                   threadsPerThreadgroup:threadsPerThreadgroup];

    // End encoding and commit the command buffer
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy the result back to host memory
    memcpy(result.data(), [bufferOutput contents],
           result.size() * sizeof(float));
  }

  return result;
}

// Create random matrix
Matrix Matrix::random(size_t rows, size_t cols, float min, float max) {
  Matrix result(rows, cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);

  for (auto &val : result.m_data) {
    val = dist(gen);
  }

  return result;
}

// Element access
float &Matrix::at(size_t row, size_t col) {
  if (row >= m_rows || col >= m_cols) {
    throw std::out_of_range("Matrix indices out of range");
  }
  return m_data[row * m_cols + col];
}

const float &Matrix::at(size_t row, size_t col) const {
  if (row >= m_rows || col >= m_cols) {
    throw std::out_of_range("Matrix indices out of range");
  }
  return m_data[row * m_cols + col];
}

// Get raw data pointer
float *Matrix::data() { return m_data.data(); }

const float *Matrix::data() const { return m_data.data(); }

// Dimensions
size_t Matrix::rows() const { return m_rows; }

size_t Matrix::cols() const { return m_cols; }

size_t Matrix::size() const { return m_data.size(); }

// Equality comparison
bool Matrix::operator==(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    return false;
  }

  // Allow small floating point differences
  const float epsilon = 1e-5f;
  for (size_t i = 0; i < m_data.size(); ++i) {
    if (std::abs(m_data[i] - other.m_data[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

bool Matrix::operator!=(const Matrix &other) const { return !(*this == other); }

// Maximum absolute difference
float Matrix::maxDifference(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::invalid_argument("Matrix dimensions mismatch for comparison");
  }

  float maxDiff = 0.0f;
  for (size_t i = 0; i < m_data.size(); ++i) {
    float diff = std::abs(m_data[i] - other.m_data[i]);
    maxDiff = std::max(maxDiff, diff);
  }
  return maxDiff;
}

// Mean absolute difference
float Matrix::meanDifference(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::invalid_argument("Matrix dimensions mismatch for comparison");
  }

  float sumDiff = 0.0f;
  for (size_t i = 0; i < m_data.size(); ++i) {
    sumDiff += std::abs(m_data[i] - other.m_data[i]);
  }
  return sumDiff / m_data.size();
}

// Relative error norm
float Matrix::relativeErrorNorm(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::invalid_argument("Matrix dimensions mismatch for comparison");
  }

  float normDiff = 0.0f;
  float normThis = 0.0f;

  for (size_t i = 0; i < m_data.size(); ++i) {
    float diff = m_data[i] - other.m_data[i];
    normDiff += diff * diff;
    normThis += m_data[i] * m_data[i];
  }

  return std::sqrt(normDiff) / std::sqrt(normThis);
}

// String representation
std::string Matrix::toString() const {
  std::stringstream ss;

  // For large matrices, only show a preview
  const size_t maxPreviewSize = 10;
  const bool showPreview = m_rows > maxPreviewSize || m_cols > maxPreviewSize;

  ss << m_rows << "x" << m_cols << " Matrix" << std::endl;

  if (m_rows == 0 || m_cols == 0) {
    ss << "[Empty]";
    return ss.str();
  }

  size_t rowsToShow = showPreview ? std::min(maxPreviewSize, m_rows) : m_rows;
  size_t colsToShow = showPreview ? std::min(maxPreviewSize, m_cols) : m_cols;

  for (size_t i = 0; i < rowsToShow; ++i) {
    ss << "[";
    for (size_t j = 0; j < colsToShow; ++j) {
      ss << std::fixed << std::setprecision(4) << at(i, j);
      if (j < colsToShow - 1) {
        ss << ", ";
      }
    }

    if (colsToShow < m_cols) {
      ss << ", ...";
    }

    ss << "]" << std::endl;
  }

  if (rowsToShow < m_rows) {
    ss << "..." << std::endl;
  }

  return ss.str();
}

Matrix Matrix::matrixAddAccelerate(const Matrix &A, const Matrix &B) const {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument("Matrix dimensions mismatch for addition");
  }

  Matrix result(A.rows(), A.cols());

  vDSP_vadd(A.data(), 1, B.data(), 1, result.data(), 1,
            static_cast<vDSP_Length>(A.size()));

  return result;
}

Matrix Matrix::matrixAddMetal(const Matrix &A, const Matrix &B) const {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument("Matrix dimensions mismatch for addition");
  }

  if (metalDevice == nil) {
    initializeMetal();
    if (metalDevice == nil) {
      throw std::runtime_error("Metal is not supported on this device!");
    }
  }

  Matrix result(A.rows(), A.cols());
  uint size = static_cast<uint>(A.size());

  @autoreleasepool {
    MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;

    id<MTLBuffer> bufferA = [metalDevice newBufferWithBytes:A.data()
                                                     length:size * sizeof(float)
                                                    options:options];
    id<MTLBuffer> bufferB = [metalDevice newBufferWithBytes:B.data()
                                                     length:size * sizeof(float)
                                                    options:options];
    id<MTLBuffer> bufferC =
        [metalDevice newBufferWithLength:size * sizeof(float) options:options];

    id<MTLBuffer> sizeBuffer = [metalDevice newBufferWithBytes:&size
                                                        length:sizeof(uint)
                                                       options:options];

    id<MTLFunction> addFunction =
        [metalLibrary newFunctionWithName:@"add_arrays"];
    if (!addFunction) {
      throw std::runtime_error("Failed to load Metal function 'add_arrays'");
    }

    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState =
        [metalDevice newComputePipelineStateWithFunction:addFunction
                                                   error:&error];
    if (!pipelineState) {
      throw std::runtime_error(
          "Failed to create Metal pipeline state: " +
          std::string([[error localizedDescription] UTF8String]));
    }

    id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pipelineState];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:2];
    [computeEncoder setBuffer:sizeBuffer offset:0 atIndex:3];

    NSUInteger maxThreadsPerGroup = pipelineState.maxTotalThreadsPerThreadgroup;
    NSUInteger threadGroupSize = MIN(256, maxThreadsPerGroup);

    NSUInteger totalThreads = (size + 3) / 4;
    NSUInteger numThreadGroups =
        (totalThreads + threadGroupSize - 1) / threadGroupSize;

    MTLSize threadgroups = MTLSizeMake(numThreadGroups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreadgroups:threadgroups
                   threadsPerThreadgroup:threadsPerThreadgroup];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    memcpy(result.data(), [bufferC contents], size * sizeof(float));
  }

  return result;
}

Matrix Matrix::adaptiveMatrixAdd(const Matrix &A, const Matrix &B,
                                 bool &usedMetal) const {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument("Matrix dimensions mismatch for addition");
  }

  static size_t METAL_THRESHOLD = 10000;
  static bool metalInitialized = false;
  static bool metalSupported = true;

  if (!metalInitialized) {
    if (metalDevice == nil) {
      try {
        initializeMetal();
        metalSupported = (metalDevice != nil);
      } catch (...) {
        metalSupported = false;
      }
    }
    metalInitialized = true;
  }

  if (metalSupported && A.size() >= METAL_THRESHOLD) {
    try {
      usedMetal = true;
      return matrixAddMetal(A, B);
    } catch (const std::exception &e) {
      METAL_THRESHOLD *= 2;
      usedMetal = false;
      return matrixAddAccelerate(A, B);
    }
  } else {
    usedMetal = false;
    return matrixAddAccelerate(A, B);
  }
}