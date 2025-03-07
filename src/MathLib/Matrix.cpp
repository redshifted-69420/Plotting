#include "Matrix.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Math {
  Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init) {
    rows_ = init.size();
    if (rows_ > 0) {
      cols_ = init.begin()->size();
    }

    data_.reserve(rows_);
    for (const auto &row: init) {
      data_.emplace_back(cols_);
      auto &vec = data_.back();
      std::copy(row.begin(), row.end(), vec.begin());
    }
  }

  Matrix::Matrix(const size_t rows, const size_t cols, const MatrixType type) : rows_(rows), cols_(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (type == Identity && rows != cols) {
      throw std::invalid_argument("Identity matrix must be square");
    }

    // Allocate data
    data_ = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));

    if (type == Identity) {
      for (size_t i = 0; i < rows; ++i) {
        data_[i][i] = 1.0f;
      }
    }
  }

  void Matrix::print() const {
    // Find the maximum width needed for any element
    size_t maxWidth = 0;
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << data_[i][j];
        maxWidth = std::max(maxWidth, ss.str().length());
      }
    }

    std::cout << "\n";
    for (size_t i = 0; i < rows_; ++i) {
      std::cout << " [";
      for (size_t j = 0; j < cols_; ++j) {
        std::cout << "\033[32m" << std::fixed << std::setprecision(2) << std::setw(maxWidth) << data_[i][j]
                  << "\033[0m";

        if (j < cols_ - 1) {
          std::cout << " ";
        }
      }
      std::cout << "]";
      std::cout << '\n';
    }
    std::cout << '\n';
  }

  float &Matrix::at(const size_t i, const size_t j) {
    if (i >= rows_ || j >= cols_) {
      throw std::out_of_range("Matrix indices out of range");
    }
    return data_[i][j];
  }

  const float &Matrix::at(const size_t i, const size_t j) const {
    if (i >= rows_ || j >= cols_) {
      throw std::out_of_range("Matrix indices out of range");
    }
    return data_[i][j];
  }

  Matrix Matrix::multiply(const Matrix &other) const {
    if (cols_ != other.getRows()) {
      throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    if (rows_ < 64 || cols_ < 64 || other.getCols() < 64) {
      return multiplyStandard(other);
    }
    size_t maxDim = std::max(rows_, std::max(cols_, other.getCols()));
    size_t nextPowerOf2 = 1;
    while (nextPowerOf2 < maxDim) {
      nextPowerOf2 *= 2;
    }
    Matrix A = padMatrix(nextPowerOf2);
    Matrix B = other.padMatrix(nextPowerOf2);
    Matrix result = coppersmithWinograd(A, B);
    return result.subMatrix(0, 0, rows_, other.getCols());
  }

  Matrix Matrix::multiplyStandard(const Matrix &other) const {
    const size_t M = rows_;
    const size_t N = other.getCols();
    const size_t K = cols_;

    Matrix result(M, N, Zeros);

    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          sum += data_[i][k] * other.data_[k][j];
        }
        result.data_[i][j] = sum;
      }
    }

    return result;
  }

  Matrix Matrix::padMatrix(size_t newSize) const {
    Matrix padded(newSize, newSize, Zeros);

    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        padded.data_[i][j] = data_[i][j];
      }
    }

    return padded;
  }

  Matrix Matrix::subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const {
    Matrix sub(numRows, numCols, Zeros);

    for (size_t i = 0; i < numRows && i + startRow < rows_; ++i) {
      for (size_t j = 0; j < numCols && j + startCol < cols_; ++j) {
        sub.data_[i][j] = data_[i + startRow][j + startCol];
      }
    }

    return sub;
  }

  Matrix Matrix::coppersmithWinograd(const Matrix &A, const Matrix &B) {
    size_t n = A.rows_;

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

    Matrix result(n, n, Zeros);

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        result.data_[i][j] = C11.data_[i][j];
        result.data_[i][j + m] = C12.data_[i][j];
        result.data_[i + m][j] = C21.data_[i][j];
        result.data_[i + m][j + m] = C22.data_[i][j];
      }
    }

    return result;
  }

  Matrix Matrix::operator+(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows_, cols_, Zeros);

    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result.data_[i][j] = data_[i][j] + other.data_[i][j];
      }
    }

    return result;
  }

  Matrix Matrix::operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows_, cols_, Zeros);

    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result.data_[i][j] = data_[i][j] - other.data_[i][j];
      }
    }

    return result;
  }

  float Matrix::determinant() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("Determinant calculation is only implemented for square matrices");
    }

    if (rows_ == 2) {
      return (data_[0][0] * data_[1][1]) - (data_[0][1] * data_[1][0]);
    }

    float det = 0.0L;

    for (size_t col = 0; col < cols_; col++) {
      std::vector<std::vector<float>> minor;
      for (size_t i = 1; i < rows_; ++i) {
        std::vector<float> row;
        for (size_t j = 0; j < cols_; ++j) {
          if (j != col) {
            row.push_back(data_[i][j]);
          }
        }
        minor.push_back(row);
      }
      Matrix minorMatrix(minor);
      const float sign = (col % 2 == 0) ? 1.0 : -1.0;
      det += sign * data_[0][col] * minorMatrix.determinant();
    }
    return det;
  }

  Matrix Matrix::getMinor(const size_t row, const size_t col) const {
    std::vector<std::vector<float>> minorData;

    for (size_t i = 0; i < rows_; ++i) {
      if (i != row) {
        std::vector<float> minorRow;
        for (size_t j = 0; j < cols_; ++j) {
          if (j != col) {
            minorRow.push_back(data_[i][j]);
          }
        }
        minorData.push_back(minorRow);
      }
    }
    return Matrix(minorData);
  }

  float Matrix::cofactor(const size_t row, const size_t col) const {
    const Matrix minor = getMinor(row, col);
    const float sign = ((row + col) % 2 == 0) ? 1.0L : -1.0L;
    return sign * minor.determinant();
  }

  Matrix Matrix::getCofactorMatrix() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("Matrix must be square to compute cofactor matrix");
    }

    std::vector<std::vector<float>> cofactorData(rows_, std::vector<float>(cols_));

    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        cofactorData[i][j] = cofactor(i, j);
      }
    }
    return Matrix(cofactorData);
  }

  Matrix Matrix::getAdjugateMatrix() const {
    const Matrix cofactorMatrix = getCofactorMatrix();
    std::vector<std::vector<float>> adjugateData(cols_, std::vector<float>(rows_));

    // Transpose of cofactor matrix
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        adjugateData[j][i] = cofactorMatrix.data_[i][j];
      }
    }
    return Matrix(adjugateData);
  }

  Matrix Matrix::inverse() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("Only square matrices can be inverted");
    }

    const float det = determinant();
    if (std::abs(det) < TOLERANCE) {
      throw std::runtime_error("Matrix is singular (non-invertible)");
    }

    const Matrix adjugate = getAdjugateMatrix();
    std::vector<std::vector<float>> inverseData(rows_, std::vector<float>(cols_));

    // Multiply adjugate by 1/determinant
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        inverseData[i][j] = adjugate.data_[i][j] / det;
      }
    }
    return Matrix(inverseData);
  }

  float Matrix::trace() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("Trace is only defined for square matrices");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < rows_; ++i) {
      sum += data_[i][i];
    }
    return sum;
  }

  Matrix Matrix::elementWiseMultiply(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrices must have same dimensions for Hadamard product");
    }

    std::vector result(rows_, std::vector<float>(cols_));
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result[i][j] = data_[i][j] * other.data_[i][j];
      }
    }
    return Matrix(result);
  }

  Matrix Matrix::transpose() const {
    std::vector transposed(cols_, std::vector<float>(rows_));
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        transposed[j][i] = data_[i][j];
      }
    }
    return Matrix(transposed);
  }

  float Matrix::vectorNorm(const std::vector<float> &vec) {
    float sum = 0.0f;
    for (const float val: vec) {
      sum += val * val;
    }
    return std::sqrt(sum);
  }

  bool Matrix::isUpperTriangular(const float tolerance) const {
    if (rows_ != cols_)
      return false;

    for (size_t i = 1; i < rows_; ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (std::abs(data_[i][j]) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

  std::vector<float> Matrix::getDiagonal() const {
    size_t minDim = std::min(rows_, cols_);
    std::vector<float> diagonal(minDim);
    for (size_t i = 0; i < minDim; ++i) {
      diagonal[i] = data_[i][i];
    }
    return diagonal;
  }

  std::pair<Matrix, Matrix> Matrix::qrDecomposition() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("QR decomposition requires a square matrix");
    }

    size_t n = rows_;
    Matrix Q = *this;
    Matrix R(n, n, Zeros);

    for (size_t j = 0; j < n; ++j) {
      std::vector<float> v(n);
      for (size_t i = 0; i < n; ++i) {
        v[i] = Q.data_[i][j];
      }

      R.data_[j][j] = vectorNorm(v);

      if (R.data_[j][j] > TOLERANCE) {
        for (size_t i = 0; i < n; ++i) {
          Q.data_[i][j] /= R.data_[j][j];
        }
      }

      for (size_t k = j + 1; k < n; ++k) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          sum += Q.data_[i][j] * Q.data_[i][k];
        }
        R.data_[j][k] = sum;

        for (size_t i = 0; i < n; ++i) {
          Q.data_[i][k] -= R.data_[j][k] * Q.data_[i][j];
        }
      }
    }

    return std::make_pair(Q, R);
  }

  std::vector<float> Matrix::calculateEigenvalues(int maxIterations) const {
    if (rows_ != cols_) {
      throw std::invalid_argument("Eigenvalue calculation requires a square matrix");
    }

    Matrix A = *this;
    int iterations = 0;

    while (!A.isUpperTriangular(TOLERANCE) && iterations < maxIterations) {
      auto [Q, R] = A.qrDecomposition();
      A = R * Q;
      iterations++;
    }

    return A.getDiagonal();
  }

  std::pair<Matrix, Matrix> Matrix::luDecomposition() const {
    if (rows_ != cols_) {
      throw std::invalid_argument("LU decomposition requires square matrix");
    }

    const size_t n = rows_;
    Matrix L(n, n, Zeros);
    Matrix U = *this;

    for (size_t i = 0; i < n; ++i) {
      L.data_[i][i] = 1.0f; // Diagonal elements of L are 1

      for (size_t j = i + 1; j < n; ++j) {
        if (std::abs(U.data_[i][i]) < TOLERANCE) {
          throw std::runtime_error("Zero pivot encountered in LU decomposition");
        }

        const float factor = U.data_[j][i] / U.data_[i][i];
        L.data_[j][i] = factor;

        for (size_t k = i; k < n; ++k) {
          U.data_[j][k] -= factor * U.data_[i][k];
        }
      }
    }

    return std::make_pair(L, U);
  }

  bool Matrix::isOrthogonal(const float tolerance) const {
    if (rows_ != cols_)
      return false;

    // Compute A * A^T
    const Matrix product = multiply(transpose());

    // Check if it's close to identity matrix
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (const float expected = (i == j) ? 1.0f : 0.0f; std::abs(product.data_[i][j] - expected) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

  std::pair<size_t, size_t> Matrix::rank() const {
    // Simple rank calculation using Gaussian elimination
    Matrix reduced = *this;
    size_t r = 0;

    for (size_t col = 0; col < cols_; ++col) {
      // Find pivot
      bool pivotFound = false;
      for (size_t row = r; row < rows_; ++row) {
        if (std::abs(reduced.data_[row][col]) > 1e-10) {
          // Swap rows
          if (row != r) {
            std::swap(reduced.data_[row], reduced.data_[r]);
          }
          pivotFound = true;
          break;
        }
      }

      if (pivotFound) {
        // Eliminate below
        for (size_t row = r + 1; row < rows_; ++row) {
          const float factor = reduced.data_[row][col] / reduced.data_[r][col];
          for (size_t j = col; j < cols_; ++j) {
            reduced.data_[row][j] -= factor * reduced.data_[r][j];
          }
        }
        r++;
      }
    }

    return {r, cols_ - r};
  }

  std::vector<std::vector<float>> Matrix::findNullVectors() const {
    size_t rows = getRows();
    size_t cols = getCols();
    if (rows == 0 || cols == 0) {
      return {};
    }
    std::vector<std::vector<float>> M(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        M[i][j] = at(i, j);
      }
    }
    std::vector<size_t> pivots(cols);
    for (size_t j = 0; j < cols; ++j) {
      pivots[j] = j;
    }
    std::vector<float> colNorm(cols, 0.0f);
    for (size_t j = 0; j < cols; ++j) {
      double sumSq = 0.0;
      for (size_t i = 0; i < rows; ++i) {
        sumSq += double(M[i][j]) * double(M[i][j]);
      }
      colNorm[j] = static_cast<float>(std::sqrt(sumSq));
    }
    auto columnSegmentNorm = [&](size_t k, size_t col) {
      double s = 0.0;
      for (size_t i = k; i < rows; ++i) {
        s += double(M[i][col]) * double(M[i][col]);
      }
      return static_cast<float>(std::sqrt(s));
    };
    size_t minRC = std::min(rows, cols);
    for (size_t k = 0; k < minRC; ++k) {
      size_t pivotCol = k;
      float maxNorm = colNorm[k];
      for (size_t j = k + 1; j < cols; ++j) {
        if (colNorm[j] > maxNorm) {
          maxNorm = colNorm[j];
          pivotCol = j;
        }
      }
      if (pivotCol != k) {
        std::swap(pivots[pivotCol], pivots[k]);
        std::swap(colNorm[pivotCol], colNorm[k]);
        for (size_t i = 0; i < rows; ++i) {
          std::swap(M[i][pivotCol], M[i][k]);
        }
      }
      float normX = columnSegmentNorm(k, k);
      if (normX < TOLERANCE) {
        continue;
      }
      float alpha = (M[k][k] > 0.f) ? -normX : normX;
      float v0 = M[k][k] - alpha;
      M[k][k] = alpha;
      for (size_t j = k + 1; j < cols; ++j) {
        double dot = double(v0) * double(M[k][j]);
        for (size_t i = k + 1; i < rows; ++i) {
          dot += double(M[i][k]) * double(M[i][j]);
        }
        double denom = double(v0) * double(v0);
        for (size_t i = k + 1; i < rows; ++i) {
          denom += double(M[i][k]) * double(M[i][k]);
        }
        if (std::fabs(denom) < 1e-32) {
          continue; // avoid dividing by zero
        }
        double scale = dot / denom;
        M[k][j] -= static_cast<float>(scale * double(v0));
        for (size_t i = k + 1; i < rows; ++i) {
          M[i][j] -= static_cast<float>(scale * double(M[i][k]));
        }
      }
      for (size_t j = k + 1; j < cols; ++j) {
        double s = 0.0;
        for (size_t i = k; i < rows; ++i) {
          s += double(M[i][j]) * double(M[i][j]);
        }
        colNorm[j] = static_cast<float>(std::sqrt(s));
      }
    }
    float maxDiag = 0.0f;
    for (size_t i = 0; i < minRC; ++i) {
      float val = std::fabs(M[i][i]);
      if (val > maxDiag) {
        maxDiag = val;
      }
    }
    float rankThreshold = maxDiag * float(std::max(rows, cols)) * TOLERANCE;
    size_t rank = 0;
    for (size_t i = 0; i < minRC; ++i) {
      if (std::fabs(M[i][i]) > rankThreshold) {
        rank++;
      } else {
        break;
      }
    }
    if (rank >= cols) {
      return {};
    }
    std::vector<std::vector<float>> nullSpace;
    nullSpace.reserve(cols - rank);

    for (size_t freeIdx = rank; freeIdx < cols; ++freeIdx) {
      std::vector<float> xPrime(cols, 0.0f);
      xPrime[freeIdx] = 1.0f;
      for (int i = int(rank) - 1; i >= 0; --i) {
        float rhs = -M[size_t(i)][freeIdx];
        for (size_t c = size_t(i) + 1; c < rank; ++c) {
          rhs -= M[size_t(i)][c] * xPrime[c];
        }
        float diagVal = M[size_t(i)][size_t(i)];
        if (std::fabs(diagVal) > TOLERANCE) {
          xPrime[size_t(i)] = rhs / diagVal;
        } else {
          xPrime[size_t(i)] = 0.0f;
        }
      }
      std::vector<float> basis(cols, 0.0f);
      for (size_t c = 0; c < cols; ++c) {
        basis[pivots[c]] = xPrime[c];
      }

      nullSpace.push_back(std::move(basis));
    }
    for (auto &v: nullSpace) {
      double sumSq = 0.0;
      for (float val: v) {
        sumSq += double(val) * double(val);
      }
      float nrm = static_cast<float>(std::sqrt(sumSq));
      if (nrm > TOLERANCE) {
        for (float &val: v) {
          val /= nrm;
        }
      }
    }
    return nullSpace;
  }
}; // namespace Math
