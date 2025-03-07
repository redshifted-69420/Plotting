#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

namespace Math {
  enum MatrixType { Identity, Zeros };
  class Matrix {
    std::vector<std::vector<float>> data_;
    size_t rows_{0}, cols_{0};
    static const float TOLERANCE;

public:
    Matrix() = default;

    [[nodiscard]] Matrix(std::initializer_list<std::initializer_list<double>> init);
    [[nodiscard]] explicit Matrix(const std::vector<std::vector<float>> &mat_vals) :
        data_(mat_vals), rows_(mat_vals.size()), cols_(mat_vals.empty() ? 0 : mat_vals[0].size()) {}
    [[nodiscard]] Matrix(size_t rows, size_t cols, MatrixType type = Zeros);
    void print() const;
    [[nodiscard]] size_t getRows() const { return rows_; }
    [[nodiscard]] size_t getCols() const { return cols_; }
    [[nodiscard]] float &at(size_t i, size_t j);
    [[nodiscard]] const float &at(size_t i, size_t j) const;
    [[nodiscard]] float determinant() const;
    [[nodiscard]] Matrix multiply(const Matrix &other) const;
    [[nodiscard]] Matrix multiplyStandard(const Matrix &other) const;
    [[nodiscard]] Matrix operator*(const Matrix &other) const { return multiply(other); }
    [[nodiscard]] Matrix operator+(const Matrix &other) const;
    [[nodiscard]] Matrix operator-(const Matrix &other) const;
    [[nodiscard]] Matrix getCofactorMatrix() const;
    [[nodiscard]] Matrix getAdjugateMatrix() const;
    [[nodiscard]] Matrix inverse() const;
    [[nodiscard]] bool isUpperTriangular(float tolerance) const;
    [[nodiscard]] Matrix getMinor(size_t row, size_t col) const;
    [[nodiscard]] float cofactor(size_t row, size_t col) const;
    [[nodiscard]] float trace() const;
    [[nodiscard]] bool isOrthogonal(float tolerance) const;
    [[nodiscard]] std::vector<float> calculateEigenvalues(int maxIterations = 100) const;
    [[nodiscard]] std::pair<Matrix, Matrix> qrDecomposition() const;
    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] static float vectorNorm(const std::vector<float> &vec);
    [[nodiscard]] std::vector<float> getDiagonal() const;
    [[nodiscard]] std::pair<size_t, size_t> rank() const;
    [[nodiscard]] Matrix elementWiseMultiply(const Matrix &other) const;
    [[nodiscard]] std::pair<Matrix, Matrix> luDecomposition() const;
    [[nodiscard]] std::vector<std::vector<float>> findNullVectors() const;
    [[nodiscard]] Matrix padMatrix(size_t newSize) const;
    [[nodiscard]] Matrix subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const;
    [[nodiscard]] static Matrix coppersmithWinograd(const Matrix &A, const Matrix &B);
    [[nodiscard]] Matrix multiplyBlocked(const Matrix &other) const;
  };

  constexpr float Matrix::TOLERANCE = 1e-10;
}; // namespace Math
#endif // MATRIX_H
