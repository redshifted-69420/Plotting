#include "Matrix.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <algorithm> // for std::min
#include <iostream>
#include <omp.h>
#include <random>
#include <simde/x86/avx.h>
#include <simde/x86/fma.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h> // AVX2
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <vecLib/vDSP.h>
#endif

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

float *Matrix::data() { return m_data.data(); }

const float *Matrix::data() const { return m_data.data(); }

size_t Matrix::rows() const { return m_rows; }

size_t Matrix::cols() const { return m_cols; }

size_t Matrix::size() const { return m_data.size(); }

Matrix::Matrix() : m_rows(0), m_cols(0), m_data() {}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> init) {
  m_rows = init.size();
  m_cols = (m_rows > 0) ? init.begin()->size() : 0;

  m_data.reserve(m_rows * m_cols);
  for (const auto &row: init) {
    if (row.size() != m_cols) {
      throw std::invalid_argument("All rows must have the same number of columns");
    }
    m_data.insert(m_data.end(), row.begin(), row.end());
  }
}

Matrix::Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0f) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float> &data) : m_rows(rows), m_cols(cols) {
  if (data.size() != rows * cols) {
    throw std::invalid_argument("Data size doesn't match matrix dimensions");
  }
  m_data = data;
}

Matrix::Matrix(size_t rows, size_t cols, float *data) : m_rows(rows), m_cols(cols), m_data(data, data + rows * cols) {}

Matrix::Matrix(const Matrix &other) : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {}

Matrix::Matrix(Matrix &&other) noexcept : m_rows(other.m_rows), m_cols(other.m_cols), m_data(std::move(other.m_data)) {
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

// Matrix Matrix::multiply(const Matrix &other) const {
//   return multiplyAdaptive(other);
// }

Matrix Matrix::transpose() const {
  Matrix result(m_cols, m_rows);
  constexpr size_t BLOCK_SIZE = 32; // Tune for cache efficiency
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m_rows; i += BLOCK_SIZE) {
    for (size_t j = 0; j < m_cols; j += BLOCK_SIZE) {
      size_t i_end = std::min(i + BLOCK_SIZE, m_rows);
      size_t j_end = std::min(j + BLOCK_SIZE, m_cols);
      for (size_t ii = i; ii < i_end; ++ii) {
        for (size_t jj = j; jj < j_end; ++jj) {
          result.at(jj, ii) = at(ii, jj);
        }
      }
    }
  }
  return result;
}

// Create random matrix
Matrix Matrix::random(size_t rows, size_t cols, float min, float max) {
  Matrix result(rows, cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);

  for (auto &val: result.m_data) {
    val = dist(gen);
  }

  return result;
}

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

float Matrix::det() const {
  if (m_rows != m_cols) {
    throw std::invalid_argument("Determinant calculation is only implemented for square matrices");
  }
  Matrix U(*this);
  float det = 1.0f;
  for (size_t i = 0; i < m_rows; ++i) {
    if (U.at(i, i) == 0.0f)
      return 0.0f;
    for (size_t j = i + 1; j < m_rows; ++j) {
      float factor = U.at(j, i) / U.at(i, i);
      for (size_t k = i; k < m_cols; ++k) {
        U.at(j, k) -= factor * U.at(i, k);
      }
    }
    det *= U.at(i, i);
  }
  return det;
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

Matrix Matrix::ParallelMatrixSub(const Matrix &A, const Matrix &B) const { return Matrix(); }

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

void Matrix::print() const {
  std::stringstream ss;

  // For large matrices, only show a preview
  const size_t maxPreviewSize = 10;
  const bool showPreview = m_rows > maxPreviewSize || m_cols > maxPreviewSize;

  ss << m_rows << "x" << m_cols << " Matrix" << std::endl;

  if (m_rows == 0 || m_cols == 0) {
    ss << "[Empty]";
    std::cout << ss.str();
    return;
  }

  size_t rowsToShow = showPreview ? std::min(maxPreviewSize, m_rows) : m_rows;
  size_t colsToShow = showPreview ? std::min(maxPreviewSize, m_cols) : m_cols;

  // Compute maximum width for alignment
  size_t maxWidth = 0;
  for (size_t i = 0; i < rowsToShow; ++i) {
    for (size_t j = 0; j < colsToShow; ++j) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(4) << at(i, j);
      maxWidth = std::max(maxWidth, oss.str().length());
    }
  }

  for (size_t i = 0; i < rowsToShow; ++i) {
    ss << "[";
    for (size_t j = 0; j < colsToShow; ++j) {
      ss << "\033[32m" << std::fixed << std::setprecision(4) << std::setw(maxWidth) << at(i, j) << "\033[0m";
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
    ss << "...";
  }

  std::cout << ss.str();
}


Matrix Matrix::multiplyStandard(const Matrix &other) const {
  const size_t M = m_rows;
  const size_t N = other.cols();
  const size_t K = m_cols;

  Matrix result(M, N);

#ifdef __APPLE__
  // Only use Accelerate when matrices are large enough to overcome the overhead
  if (M > 4 && N > 4 && K > 4) {
    // Direct use of vDSP for small matrices
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        // Use vDSP_dotpr for each dot product (row of A with column of B)
        float dotProduct = 0.0f;
        vDSP_dotpr(&at(i, 0), 1, &other.at(0, j), N, &dotProduct, K);
        result.at(i, j) = dotProduct;
      }
    }
  } else {
#endif
    // Standard implementation for small matrices or non-Apple devices
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          sum += at(i, k) * other.at(k, j);
        }
        result.at(i, j) = sum;
      }
    }
#ifdef __APPLE__
  }
#endif

  return result;
}

Matrix Matrix::multiply(const Matrix &other) const {
  if (m_cols != other.rows()) {
    throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
  }

  const size_t M = m_rows;
  const size_t N = other.cols();
  const size_t K = m_cols;

  // Use naive multiplication for small matrices (≤ 16x16)
  if (M <= 16 && N <= 16 && K <= 16) {
    return multiplyStandard(other);
  }

#ifdef __APPLE__
  // Use Accelerate for large matrices (≥ 64x64)
  if (M >= 64 && N >= 64 && K >= 64) {
    Matrix result(M, N);

    std::vector<float> A_flat(M * K);
    std::vector<float> B_flat(K * N);
    std::vector<float> C_flat(M * N, 0.0f);

    // Flatten matrix data
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < K; ++j)
        A_flat[i * K + j] = at(i, j);

    for (size_t i = 0; i < K; ++i)
      for (size_t j = 0; j < N; ++j)
        B_flat[i * N + j] = other.at(i, j);

    // Call Accelerate BLAS function
    const float alpha = 1.0f, beta = 0.0f;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A_flat.data(), K, B_flat.data(), N, beta,
                C_flat.data(), N);
#pragma clang diagnostic pop
    // Copy result back to matrix format
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        result.at(i, j) = C_flat[i * N + j];

    return result;
  }
#endif

  // Fallback for medium matrices (16x16 < size < 64x64)
  return multiplyStandard(other);
}


Matrix Matrix::padMatrix(size_t newSize) const {
  Matrix padded(newSize, newSize);

  for (size_t i = 0; i < m_rows; ++i) {
    for (size_t j = 0; j < m_cols; ++j) {
      padded.at(i, j) = at(i, j);
    }
  }

  return padded;
}

Matrix Matrix::subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const {
  Matrix sub(numRows, numCols);

  for (size_t i = 0; i < numRows && i + startRow < m_rows; ++i) {
    for (size_t j = 0; j < numCols && j + startCol < m_cols; ++j) {
      sub.at(i, j) = at(i + startRow, j + startCol);
    }
  }

  return sub;
}

Matrix Matrix::coppersmithWinograd(const Matrix &A, const Matrix &B) {
  size_t n = A.m_rows;

  if (n <= 128) {
    return A.multiplyStandard(B);
  }

  size_t m = n / 2;

  Matrix A11 = A.subMatrix(0, 0, m, m);
  Matrix A12 = A.subMatrix(0, m, m, m);
  Matrix A21 = A.subMatrix(m, 0, m, m);
  Matrix A22 = A.subMatrix(m, m, m, m);

  Matrix B11 = B.subMatrix(0, 0, m, m);
  Matrix B12 = B.subMatrix(0, m, m, m);
  Matrix B21 = B.subMatrix(m, 0, m, m);
  Matrix B22 = B.subMatrix(m, m, m, m);

  std::vector<Matrix> M(15);

#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      M[0] = coppersmithWinograd(A11 + A22, B11 + B22);

#pragma omp section
      M[1] = coppersmithWinograd(A21 + A22, B11);

#pragma omp section
      M[2] = coppersmithWinograd(A11, B12 - B22);

#pragma omp section
      M[3] = coppersmithWinograd(A22, B21 - B11);

#pragma omp section
      M[4] = coppersmithWinograd(A11 + A12, B22);

#pragma omp section
      M[5] = coppersmithWinograd(A21 - A11, B11 + B12);

#pragma omp section
      M[6] = coppersmithWinograd(A12 - A22, B21 + B22);

#pragma omp section
      M[7] = coppersmithWinograd(A11 + A22, B11 - B22);

#pragma omp section
      M[8] = coppersmithWinograd(A11 - A21, B11 + B21);

#pragma omp section
      M[9] = coppersmithWinograd(A11 + A12, B11 - B21);

#pragma omp section
      M[10] = coppersmithWinograd(A22, B12 - B11);

#pragma omp section
      M[11] = coppersmithWinograd(A21 + A22, B22);

#pragma omp section
      M[12] = coppersmithWinograd(A12, B21);

#pragma omp section
      M[13] = coppersmithWinograd(A21, B12);

#pragma omp section
      M[14] = coppersmithWinograd(A12 - A11, B21 - B22);
    }
  }

  Matrix C11 = M[0] + M[3] - M[4] + M[6] + M[7];
  Matrix C12 = M[2] + M[4] + M[9] + M[10];
  Matrix C21 = M[1] + M[3] + M[8] + M[11];
  Matrix C22 = M[0] + M[2] - M[1] + M[5] + M[14];

  Matrix result(n, n);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      result.at(i, j) = C11.at(i, j);
      result.at(i, j + m) = C12.at(i, j);
      result.at(i + m, j) = C21.at(i, j);
      result.at(i + m, j + m) = C22.at(i, j);
    }
  }

  return result;
}

Matrix Matrix::operator+(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
  }

  Matrix result(m_rows, m_cols);

  for (size_t i = 0; i < m_rows; ++i) {
    for (size_t j = 0; j < m_cols; ++j) {
      result.at(i, j) = at(i, j) + other.at(i, j);
    }
  }

  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::invalid_argument("Matrix dimensions must match for subtraction");
  }

  Matrix result(m_rows, m_cols);

  for (size_t i = 0; i < m_rows; ++i) {
    for (size_t j = 0; j < m_cols; ++j) {
      result.at(i, j) = at(i, j) - other.at(i, j);
    }
  }

  return result;
}
