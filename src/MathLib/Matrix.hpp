#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>

class Matrix {
  public:
  // Constructors
  Matrix(); // Default constructor
  Matrix(std::initializer_list<std::initializer_list<float>> init);
  Matrix(size_t rows, size_t cols); // Create a zero-initialized matrix
  Matrix(size_t rows, size_t cols,
         const std::vector<float> &data); // Create from vector
  Matrix(size_t rows, size_t cols, float *data); // Create from raw array
  Matrix(const Matrix &other); // Copy constructor
  Matrix(Matrix &&other) noexcept; // Move constructor
  ~Matrix(); // Destructor

  // Accessors
  float &at(size_t row, size_t col);
  const float &at(size_t row, size_t col) const;
  float *data();
  const float *data() const;
  size_t rows() const;
  size_t cols() const;
  size_t size() const;

  // Assignment operators
  Matrix &operator=(const Matrix &other);
  Matrix &operator=(Matrix &&other) noexcept;

  // Matrix operations
  [[nodiscard]] Matrix operator*(const Matrix &other) const { return multiply(other); };
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix T() const { return transpose(); }
  [[nodiscard]] float det() const;

  // Basic operations
  static Matrix random(size_t rows, size_t cols, float min = -10.0f, float max = 10.0f);

  // Comparison
  bool operator==(const Matrix &other) const;
  bool operator!=(const Matrix &other) const;
  float maxDifference(const Matrix &other) const;
  float meanDifference(const Matrix &other) const;
  float relativeErrorNorm(const Matrix &other) const;

  // Output
  std::string toString() const;
  void print() const;

  private:
  size_t m_rows;
  size_t m_cols;
  std::vector<float> m_data;

  private:
  Matrix multiply(const Matrix &other) const;
  Matrix multiplyStandard(const Matrix &other) const;
  Matrix ParallelMatrixSub(const Matrix &A, const Matrix &B) const;
  Matrix transpose() const;
  Matrix classicalMultiply(const Matrix &B) const;
  Matrix strassenMultiply(const Matrix &B) const;
  Matrix padMatrix(size_t newSize) const;
  Matrix coppersmithWinograd(const Matrix &A, const Matrix &B);
  Matrix subMatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const;
};

#endif // MATRIX_H
