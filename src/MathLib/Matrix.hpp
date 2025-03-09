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

  // Assignment operators
  Matrix &operator=(const Matrix &other);
  Matrix &operator=(Matrix &&other) noexcept;

  // Matrix operations
  [[nodiscard]] Matrix operator*(const Matrix &other) const { return multiply(other); };
  Matrix operator+(const Matrix &other) const {
    bool usedMetal = false;
    return adaptiveMatrixAdd(*this, other, usedMetal);
  }
  Matrix operator-(const Matrix &other) const {
    bool usedMetal = false;
    return adaptiveMatrixSub(*this, other, usedMetal);
  }

  // [{Matrix Methods}]
  [[nodiscard]] float det() const;

  // Basic operations
  Matrix transpose() const { return optimisedTrasnpose(); };
  static Matrix random(size_t rows, size_t cols, float min = -10.0f, float max = 10.0f);

  // Accessors
  float &at(size_t row, size_t col);
  const float &at(size_t row, size_t col) const;
  float *data();
  const float *data() const;
  size_t rows() const;
  size_t cols() const;
  size_t size() const;

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
  Matrix multiplyAdaptive(const Matrix &other) const; // Adaptive variant
  Matrix multiplyBLAS(const Matrix &other) const; // BLAS variant
  Matrix multiplyMetal(const Matrix &A, const Matrix &B) const; // Metal variant
  Matrix multiply(const Matrix &other) const; // General multiplication
  //// add method
  Matrix matrixAddMetal(const Matrix &A, const Matrix &B) const; ////
  Matrix matrixAddAccelerate(const Matrix &A, const Matrix &B) const; ////
  Matrix adaptiveMatrixAdd(const Matrix &A, const Matrix &B, bool &usedMetal) const; ////

  /// sun method
  Matrix matrixSubMetal(const Matrix &A, const Matrix &B) const;
  Matrix matrixSubAccelerate(const Matrix &A, const Matrix &B) const; ////
  Matrix adaptiveMatrixSub(const Matrix &A, const Matrix &B, bool &usedMetal) const; ////

  /// [other method]
  Matrix transposeMetal() const;
  Matrix optimisedTrasnpose() const;
  Matrix transposeNAIVE() const;
};

#endif // MATRIX_H
