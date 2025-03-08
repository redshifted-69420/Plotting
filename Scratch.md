```c++
  // Measure time for BLAS multiplication
  start = std::chrono::high_resolution_clock::now();
  Matrix C_BLAS = A.multiplyBLAS(B);
  end = std::chrono::high_resolution_clock::now();
  auto duration_BLAS = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "BLAS Execution Time: " << duration_BLAS.count() << " ms" << std::endl;

  // Measure time for Metal multiplication
  start = std::chrono::high_resolution_clock::now();
  Matrix C_Metal = A.multiplyMetal(B);
  end = std::chrono::high_resolution_clock::now();
  auto duration_Metal = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Metal Execution Time: " << duration_Metal.count() << " ms" << std::endl;
```