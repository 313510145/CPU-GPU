#include "CPU_manager.cuh"

#include <random>
#include <stdexcept>
#include <omp.h>

unsigned long long CPU_manager::get_max_threads() {
    return omp_get_max_threads();
}

void CPU_manager::set_time_stamp(const std::string& key) {
    time_stamp[key] = omp_get_wtime();
}

double CPU_manager::get_time_duration(const std::string& key_start, const std::string& key_end) {
    if (time_stamp.find(key_start) == time_stamp.end() || time_stamp.find(key_end) == time_stamp.end()) {
        throw std::runtime_error("Time stamp not found for keys: " + key_start + " or " + key_end);
    }
    return time_stamp[key_end] - time_stamp[key_start];
}

void CPU_manager::test_CPU_matrix_add() {
    std::vector<double> value_a;
    std::vector<double> value_b;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
        value_b.push_back(static_cast<double>(10 - i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(value_b, 3, 3);
    CPU_matrix<double> matrix_c_(matrix_a + 1.0);
    CPU_matrix<double> matrix_d_ = openmp_add(matrix_a, 1.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("Matrix addition with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("Matrix addition with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("Matrix addition with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("Matrix addition with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix<double> matrix_c = matrix_a + matrix_b;
    CPU_matrix<double> matrix_d(openmp_add(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("Matrix addition results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("Matrix addition results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("Matrix addition results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("Matrix addition results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_subtract() {
    std::vector<double> value_a;
    std::vector<double> value_b;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
        value_b.push_back(static_cast<double>(10 - i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(value_b, 3, 3);
    CPU_matrix<double> matrix_c_(matrix_a - 1.0);
    CPU_matrix<double> matrix_d_ = openmp_subtract(matrix_a, 1.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("Matrix subtraction with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("Matrix subtraction with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("Matrix subtraction with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("Matrix subtraction with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix<double> matrix_c = matrix_a - matrix_b;
    CPU_matrix<double> matrix_d(openmp_subtract(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("Matrix subtraction results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("Matrix subtraction results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("Matrix subtraction results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("Matrix subtraction results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_multiply() {
    std::vector<double> value_a;
    std::vector<double> value_b;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
        value_b.push_back(static_cast<double>(10 - i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(value_b, 3, 3);
    CPU_matrix<double> matrix_c_(matrix_a * 2.0);
    CPU_matrix<double> matrix_d_ = openmp_multiply(matrix_a, 2.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("Matrix multiplication with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("Matrix multiplication with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("Matrix multiplication with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("Matrix multiplication with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix<double> matrix_c = matrix_a * matrix_b;
    CPU_matrix<double> matrix_d(openmp_multiply(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("Matrix multiplication results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("Matrix multiplication results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("Matrix multiplication results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("Matrix multiplication results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_transpose() {
    std::vector<double> value_a;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(transpose(matrix_a));
    CPU_matrix<double> matrix_c(openmp_transpose(matrix_a));
    if (!(matrix_b == matrix_c)) {
        throw std::runtime_error("Matrix transpose results do not match.");
    }
    if (!openmp_equal(matrix_b, matrix_c)) {
        throw std::runtime_error("Matrix transpose results do not match (openmp_equal check).");
    }
    if (matrix_b != matrix_c) {
        throw std::runtime_error("Matrix transpose results do not match.");
    }
    if (openmp_not_equal(matrix_b, matrix_c)) {
        throw std::runtime_error("Matrix transpose results do not match (openmp_not_equal check).");
    }
    matrix_b = openmp_transpose(matrix_b);
    matrix_c = transpose(matrix_c);
    if (!(matrix_a == matrix_b)) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix.");
    }
    if (!openmp_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix (openmp_equal check).");
    }
    if (matrix_a != matrix_b) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix.");
    }
    if (openmp_not_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    }
    if (!(matrix_a == matrix_c)) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix (openmp).");
    }
    if (!openmp_equal(matrix_a, matrix_c)) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix (openmp_equal check).");
    }
    if (matrix_a != matrix_c) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix (openmp).");
    }
    if (openmp_not_equal(matrix_a, matrix_c)) {
        throw std::runtime_error("Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_CSR_add() {
    std::vector<unsigned long long> row_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned long long> row_ptr_b = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_b = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_b = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    CPU_matrix_CSR<double> matrix_a(row_ptr_a, col_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_b(row_ptr_b, col_idx_b, value_b, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_c_(matrix_a + 1.0);
    CPU_matrix_CSR<double> matrix_d_ = openmp_add(matrix_a, 1.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("CSR Matrix addition with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSR Matrix addition with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("CSR Matrix addition with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSR Matrix addition with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix_CSR<double> matrix_c = matrix_a + matrix_b;
    CPU_matrix_CSR<double> matrix_d(openmp_add(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("CSR Matrix addition results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSR Matrix addition results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("CSR Matrix addition results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSR Matrix addition results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_CSR_subtract() {
    std::vector<unsigned long long> row_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned long long> row_ptr_b = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_b = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_b = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    CPU_matrix_CSR<double> matrix_a(row_ptr_a, col_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_b(row_ptr_b, col_idx_b, value_b, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_c_(matrix_a - 1.0);
    CPU_matrix_CSR<double> matrix_d_ = openmp_subtract(matrix_a, 1.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("CSR Matrix subtraction with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSR Matrix subtraction with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("CSR Matrix subtraction with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSR Matrix subtraction with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix_CSR<double> matrix_c = matrix_a - matrix_b;
    CPU_matrix_CSR<double> matrix_d(openmp_subtract(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("CSR Matrix subtraction results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSR Matrix subtraction results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("CSR Matrix subtraction results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSR Matrix subtraction results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_CSR_multiply() {
    std::vector<unsigned long long> row_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned long long> row_ptr_b = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_b = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_b = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    CPU_matrix_CSR<double> matrix_a(row_ptr_a, col_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_b(row_ptr_b, col_idx_b, value_b, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_c_(matrix_a * 2.0);
    CPU_matrix_CSR<double> matrix_d_ = openmp_multiply(matrix_a, 2.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("CSR Matrix multiplication with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSR Matrix multiplication with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("CSR Matrix multiplication with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSR Matrix multiplication with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix_CSR<double> matrix_c = matrix_a * matrix_b;
    CPU_matrix_CSR<double> matrix_d(openmp_multiply(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("CSR Matrix multiplication results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSR Matrix multiplication results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("CSR Matrix multiplication results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSR Matrix multiplication results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_CSR_transpose() {
    std::vector<unsigned long long> row_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    CPU_matrix_CSR<double> matrix_a(row_ptr_a, col_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSR<double> matrix_b(transpose(matrix_a));
    // CPU_matrix_CSR<double> matrix_c = openmp_transpose(matrix_a);
    // if (!(matrix_b == matrix_c)) {
    //     throw std::runtime_error("CSR Matrix transpose results do not match.");
    // }
    // if (!openmp_equal(matrix_b, matrix_c)) {
    //     throw std::runtime_error("CSR Matrix transpose results do not match (openmp_equal check).");
    // }
    // if (matrix_b != matrix_c) {
    //     throw std::runtime_error("CSR Matrix transpose results do not match.");
    // }
    // if (openmp_not_equal(matrix_b, matrix_c)) {
    //     throw std::runtime_error("CSR Matrix transpose results do not match (openmp_not_equal check).");
    // }
    // matrix_b = openmp_transpose(matrix_b);
    matrix_b = transpose(matrix_b);
    // matrix_c = transpose(matrix_c);
    if (!(matrix_a == matrix_b)) {
        throw std::runtime_error("CSR Matrix transpose back results do not match original matrix.");
    }
    if (!openmp_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("CSR Matrix transpose back results do not match original matrix (openmp_equal check).");
    }
    if (matrix_a != matrix_b) {
        throw std::runtime_error("CSR Matrix transpose back results do not match original matrix.");
    }
    if (openmp_not_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("CSR Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    }
    // if (!(matrix_a == matrix_c)) {
    //     throw std::runtime_error("CSR Matrix transpose back results do not match original matrix (openmp).");
    // }
    // if (!openmp_equal(matrix_a, matrix_c)) {
    //     throw std::runtime_error("CSR Matrix transpose back results do not match original matrix (openmp_equal check).");
    // }
    // if (matrix_a != matrix_c) {
    //     throw std::runtime_error("CSR Matrix transpose back results do not match original matrix (openmp).");
    // }
    // if (openmp_not_equal(matrix_a, matrix_c)) {
    //     throw std::runtime_error("CSR Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    // }
}

void CPU_manager::test_CPU_matrix_CSC_add() {
    std::vector<unsigned long long> col_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> row_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    std::vector<unsigned long long> col_ptr_b = {0, 3, 6, 9};
    std::vector<unsigned long long> row_idx_b = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_b = {9.0, 6.0, 3.0, 8.0, 5.0, 2.0, 7.0, 4.0, 1.0};
    CPU_matrix_CSC<double> matrix_a(col_ptr_a, row_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSC<double> matrix_b(col_ptr_b, row_idx_b, value_b, 9, 3, 3);
    CPU_matrix_CSC<double> matrix_c_(matrix_a + 1.0);
    CPU_matrix_CSC<double> matrix_d_ = openmp_add(matrix_a, 1.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("CSC Matrix addition with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSC Matrix addition with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("CSC Matrix addition with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSC Matrix addition with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix_CSC<double> matrix_c = matrix_a + matrix_b;
    CPU_matrix_CSC<double> matrix_d(openmp_add(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("CSC Matrix addition results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSC Matrix addition results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("CSC Matrix addition results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSC Matrix addition results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_CSC_subtract() {
    std::vector<unsigned long long> col_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> row_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    std::vector<unsigned long long> col_ptr_b = {0, 3, 6, 9};
    std::vector<unsigned long long> row_idx_b = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_b = {9.0, 6.0, 3.0, 8.0, 5.0, 2.0, 7.0, 4.0, 1.0};
    CPU_matrix_CSC<double> matrix_a(col_ptr_a, row_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSC<double> matrix_b(col_ptr_b, row_idx_b, value_b, 9, 3, 3);
    CPU_matrix_CSC<double> matrix_c_(matrix_a - 1.0);
    CPU_matrix_CSC<double> matrix_d_ = openmp_subtract(matrix_a, 1.0);
    if (!(matrix_c_ == matrix_d_)) {
        throw std::runtime_error("CSC Matrix subtraction with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSC Matrix subtraction with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != matrix_d_) {
        throw std::runtime_error("CSC Matrix subtraction with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, matrix_d_)) {
        throw std::runtime_error("CSC Matrix subtraction with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix_CSC<double> matrix_c = matrix_a - matrix_b;
    CPU_matrix_CSC<double> matrix_d(openmp_subtract(matrix_a, matrix_b));
    if (!(matrix_c == matrix_d)) {
        throw std::runtime_error("CSC Matrix subtraction results do not match.");
    }
    if (!openmp_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSC Matrix subtraction results do not match (openmp_equal check).");
    }
    if (matrix_c != matrix_d) {
        throw std::runtime_error("CSC Matrix subtraction results do not match.");
    }
    if (openmp_not_equal(matrix_c, matrix_d)) {
        throw std::runtime_error("CSC Matrix subtraction results do not match (openmp_not_equal check).");
    }
}

void CPU_manager::test_CPU_matrix_CSC_transpose() {
    std::vector<unsigned long long> col_ptr_a = {0, 3, 6, 9};
    std::vector<unsigned long long> row_idx_a = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> value_a = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    CPU_matrix_CSC<double> matrix_a(col_ptr_a, row_idx_a, value_a, 9, 3, 3);
    CPU_matrix_CSC<double> matrix_b(transpose(matrix_a));
    // CPU_matrix_CSC<double> matrix_c = openmp_transpose(matrix_a);
    // if (!(matrix_b == matrix_c)) {
    //     throw std::runtime_error("CSC Matrix transpose results do not match.");
    // }
    // if (!openmp_equal(matrix_b, matrix_c)) {
    //     throw std::runtime_error("CSC Matrix transpose results do not match (openmp_equal check).");
    // }
    // if (matrix_b != matrix_c) {
    //     throw std::runtime_error("CSC Matrix transpose results do not match.");
    // }
    // if (openmp_not_equal(matrix_b, matrix_c)) {
    //     throw std::runtime_error("CSC Matrix transpose results do not match (openmp_not_equal check).");
    // }
    // matrix_b = openmp_transpose(matrix_b);
    matrix_b = transpose(matrix_b);
    // matrix_c = transpose(matrix_c);
    if (!(matrix_a == matrix_b)) {
        throw std::runtime_error("CSC Matrix transpose back results do not match original matrix.");
    }
    if (!openmp_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("CSC Matrix transpose back results do not match original matrix (openmp_equal check).");
    }
    if (matrix_a != matrix_b) {
        throw std::runtime_error("CSC Matrix transpose back results do not match original matrix.");
    }
    if (openmp_not_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("CSC Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    }
    // if (!(matrix_a == matrix_c)) {
    //     throw std::runtime_error("CSC Matrix transpose back results do not match original matrix (openmp).");
    // }
    // if (!openmp_equal(matrix_a, matrix_c)) {
    //     throw std::runtime_error("CSC Matrix transpose back results do not match original matrix (openmp_equal check).");
    // }
    // if (matrix_a != matrix_c) {
    //     throw std::runtime_error("CSC Matrix transpose back results do not match original matrix (openmp).");
    // }
    // if (openmp_not_equal(matrix_a, matrix_c)) {
    //     throw std::runtime_error("CSC Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    // }
}

void CPU_manager::test_GPU_matrix_add() {
    std::vector<double> value_a;
    std::vector<double> value_b;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
        value_b.push_back(static_cast<double>(10 - i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(value_b, 3, 3);
    GPU_matrix<double> gpu_matrix_a;
    gpu_matrix_a.construct_from(value_a, 3, 3);
    GPU_matrix<double> gpu_matrix_b;
    gpu_matrix_b.construct_from(value_b, 3, 3);
    CPU_matrix<double> matrix_c_(matrix_a + 1.0);
    GPU_matrix<double> matrix_d_;
    matrix_d_ = add_2d(gpu_matrix_a, 1.0);
    CPU_matrix<double> cpu_matrix_d_;
    cpu_matrix_d_.allocate(3, 3);
    matrix_d_.copy_to(cpu_matrix_d_);
    if (!(matrix_c_ == cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix addition with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix addition with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != cpu_matrix_d_) {
        throw std::runtime_error("GPU Matrix addition with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix addition with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix<double> matrix_c = matrix_a + matrix_b;
    GPU_matrix<double> matrix_d;
    matrix_d = add_2d(gpu_matrix_a, gpu_matrix_b);
    CPU_matrix<double> cpu_matrix_d;
    cpu_matrix_d.allocate(3, 3);
    matrix_d.copy_to(cpu_matrix_d);
    if (!(matrix_c == cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix addition results do not match.");
    }
    if (!openmp_equal(matrix_c, cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix addition results do not match (openmp_equal check).");
    }
    if (matrix_c != cpu_matrix_d) {
        throw std::runtime_error("GPU Matrix addition results do not match.");
    }
    if (openmp_not_equal(matrix_c, cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix addition results do not match (openmp_not_equal check).");
    }
    gpu_matrix_a.free();
    gpu_matrix_b.free();
    matrix_d_.free();
    matrix_d.free();
}

void CPU_manager::test_GPU_matrix_subtract() {
    std::vector<double> value_a;
    std::vector<double> value_b;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
        value_b.push_back(static_cast<double>(10 - i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(value_b, 3, 3);
    GPU_matrix<double> gpu_matrix_a;
    gpu_matrix_a.construct_from(value_a, 3, 3);
    GPU_matrix<double> gpu_matrix_b;
    gpu_matrix_b.construct_from(value_b, 3, 3);
    CPU_matrix<double> matrix_c_(matrix_a - 1.0);
    GPU_matrix<double> matrix_d_;
    matrix_d_ = subtract_2d(gpu_matrix_a, 1.0);
    CPU_matrix<double> cpu_matrix_d_;
    cpu_matrix_d_.allocate(3, 3);
    matrix_d_.copy_to(cpu_matrix_d_);
    if (!(matrix_c_ == cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix subtraction with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix subtraction with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != cpu_matrix_d_) {
        throw std::runtime_error("GPU Matrix subtraction with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix subtraction with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix<double> matrix_c = matrix_a - matrix_b;
    GPU_matrix<double> matrix_d;
    matrix_d = subtract_2d(gpu_matrix_a, gpu_matrix_b);
    CPU_matrix<double> cpu_matrix_d;
    cpu_matrix_d.allocate(3, 3);
    matrix_d.copy_to(cpu_matrix_d);
    if (!(matrix_c == cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix subtraction results do not match.");
    }
    if (!openmp_equal(matrix_c, cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix subtraction results do not match (openmp_equal check).");
    }
    if (matrix_c != cpu_matrix_d) {
        throw std::runtime_error("GPU Matrix subtraction results do not match.");
    }
    if (openmp_not_equal(matrix_c, cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix subtraction results do not match (openmp_not_equal check).");
    }
    gpu_matrix_a.free();
    gpu_matrix_b.free();
    matrix_d_.free();
    matrix_d.free();
}

void CPU_manager::test_GPU_matrix_multiply() {
    std::vector<double> value_a;
    std::vector<double> value_b;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
        value_b.push_back(static_cast<double>(10 - i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    CPU_matrix<double> matrix_b(value_b, 3, 3);
    GPU_matrix<double> gpu_matrix_a;
    gpu_matrix_a.construct_from(value_a, 3, 3);
    GPU_matrix<double> gpu_matrix_b;
    gpu_matrix_b.construct_from(value_b, 3, 3);
    CPU_matrix<double> matrix_c_(matrix_a * 2.0);
    GPU_matrix<double> matrix_d_;
    matrix_d_ = multiply_2d(gpu_matrix_a, 2.0);
    CPU_matrix<double> cpu_matrix_d_;
    cpu_matrix_d_.allocate(3, 3);
    matrix_d_.copy_to(cpu_matrix_d_);
    if (!(matrix_c_ == cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix multiplication with scalar results do not match.");
    }
    if (!openmp_equal(matrix_c_, cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix multiplication with scalar results do not match (openmp_equal check).");
    }
    if (matrix_c_ != cpu_matrix_d_) {
        throw std::runtime_error("GPU Matrix multiplication with scalar results do not match.");
    }
    if (openmp_not_equal(matrix_c_, cpu_matrix_d_)) {
        throw std::runtime_error("GPU Matrix multiplication with scalar results do not match (openmp_not_equal check).");
    }
    CPU_matrix<double> matrix_c = matrix_a * matrix_b;
    GPU_matrix<double> matrix_d;
    matrix_d = multiply_2d(gpu_matrix_a, gpu_matrix_b);
    CPU_matrix<double> cpu_matrix_d;
    cpu_matrix_d.allocate(3, 3);
    matrix_d.copy_to(cpu_matrix_d);
    if (!(matrix_c == cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix multiplication results do not match.");
    }
    if (!openmp_equal(matrix_c, cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix multiplication results do not match (openmp_equal check).");
    }
    if (matrix_c != cpu_matrix_d) {
        throw std::runtime_error("GPU Matrix multiplication results do not match.");
    }
    if (openmp_not_equal(matrix_c, cpu_matrix_d)) {
        throw std::runtime_error("GPU Matrix multiplication results do not match (openmp_not_equal check).");
    }
    gpu_matrix_a.free();
    gpu_matrix_b.free();
    matrix_d_.free();
    matrix_d.free();
}

void CPU_manager::test_GPU_matrix_transpose() {
    std::vector<double> value_a;
    for (unsigned long long i = 1; i < 10; ++i) {
        value_a.push_back(static_cast<double>(i));
    }
    CPU_matrix<double> matrix_a(value_a, 3, 3);
    GPU_matrix<double> gpu_matrix_a;
    gpu_matrix_a.construct_from(value_a, 3, 3);
    CPU_matrix<double> matrix_b(transpose(matrix_a));
    GPU_matrix<double> matrix_c = transpose_2d(gpu_matrix_a);
    CPU_matrix<double> cpu_matrix_c;
    cpu_matrix_c.allocate(3, 3);
    matrix_c.copy_to(cpu_matrix_c);
    if (!(matrix_b == cpu_matrix_c)) {
        throw std::runtime_error("GPU Matrix transpose results do not match.");
    }
    if (!openmp_equal(matrix_b, cpu_matrix_c)) {
        throw std::runtime_error("GPU Matrix transpose results do not match (openmp_equal check).");
    }
    if (matrix_b != cpu_matrix_c) {
        throw std::runtime_error("GPU Matrix transpose results do not match.");
    }
    if (openmp_not_equal(matrix_b, cpu_matrix_c)) {
        throw std::runtime_error("GPU Matrix transpose results do not match (openmp_not_equal check).");
    }
    matrix_b = transpose(matrix_b);
    matrix_c = transpose_2d(matrix_c);
    matrix_c.copy_to(cpu_matrix_c);
    if (!(matrix_a == matrix_b)) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix.");
    }
    if (!openmp_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix (openmp_equal check).");
    }
    if (matrix_a != matrix_b) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix.");
    }
    if (openmp_not_equal(matrix_a, matrix_b)) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    }
    if (!(matrix_a == cpu_matrix_c)) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix (openmp).");
    }
    if (!openmp_equal(matrix_a, cpu_matrix_c)) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix (openmp_equal check).");
    }
    if (matrix_a != cpu_matrix_c) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix (openmp).");
    }
    if (openmp_not_equal(matrix_a, cpu_matrix_c)) {
        throw std::runtime_error("GPU Matrix transpose back results do not match original matrix (openmp_not_equal check).");
    }
}

void CPU_manager::test_GPU_sparse_matrix_vector_multiply() {
    const unsigned long long NODE_NUM = 10000000;
    std::vector<unsigned long long> row_ptr;
    std::vector<unsigned long long> col_idx;
    std::vector<double> matrix_value;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist(1, 1000000);
    std::uniform_real_distribution<double> value_dist(1.0, 10.0);
    row_ptr.push_back(0);
    for (unsigned long long i = 1; i <= NODE_NUM; ++i) {
        row_ptr.push_back(i * 7);
        unsigned long long col_idx_accumulator = 0;
        for (unsigned long long j = 0; j < 7; ++j) {
            col_idx_accumulator += dist(gen);
            col_idx.push_back(col_idx_accumulator);
            matrix_value.push_back(value_dist(gen));
        }
    }
    CPU_matrix_CSR<double> cpu_matrix(row_ptr, col_idx, matrix_value, matrix_value.size(), NODE_NUM, NODE_NUM);
    GPU_matrix_CSR<double> gpu_matrix;
    gpu_matrix.construct_from(row_ptr, col_idx, matrix_value, matrix_value.size(), NODE_NUM, NODE_NUM);
    std::vector<double> vector_value;
    for (unsigned long long i = 0; i < NODE_NUM; ++i) {
        vector_value.push_back(value_dist(gen));
    }
    CPU_matrix<double> cpu_vector(vector_value, NODE_NUM, 1);
    GPU_matrix<double> gpu_vector;
    gpu_vector.construct_from(vector_value, NODE_NUM, 1);
    CPU_matrix<double> cpu_result = cpu_matrix * cpu_vector;
    GPU_matrix<double> gpu_result = multiply_2d(gpu_matrix, gpu_vector);
    for (unsigned long long i = 0; i < 99; ++i) {
        cpu_result = cpu_matrix * cpu_result;
        gpu_result = multiply_2d(gpu_matrix, gpu_result);
    }
    CPU_matrix<double> cpu_result_gpu;
    cpu_result_gpu.allocate(NODE_NUM, 1);
    gpu_result.copy_to(cpu_result_gpu);
    if (!(cpu_result == cpu_result_gpu)) {
        throw std::runtime_error("GPU sparse matrix-vector multiplication results do not match.");
    }
    if (!openmp_equal(cpu_result, cpu_result_gpu)) {
        throw std::runtime_error("GPU sparse matrix-vector multiplication results do not match (openmp_equal check).");
    }
    if (cpu_result != cpu_result_gpu) {
        throw std::runtime_error("GPU sparse matrix-vector multiplication results do not match.");
    }
    if (openmp_not_equal(cpu_result, cpu_result_gpu)) {
        throw std::runtime_error("GPU sparse matrix-vector multiplication results do not match (openmp_not_equal check).");
    }
    gpu_matrix.free();
    gpu_vector.free();
    gpu_result.free();
}

void CPU_manager::run_sparse_matrix_vector_multiply() {
    const unsigned long long NODE_NUM = 10000000;
    std::vector<unsigned long long> row_ptr;
    std::vector<unsigned long long> col_idx;
    std::vector<double> matrix_value;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist(1, 1000000);
    std::uniform_real_distribution<double> value_dist(1.0, 10.0);
    row_ptr.push_back(0);
    for (unsigned long long i = 1; i <= NODE_NUM; ++i) {
        row_ptr.push_back(i * 7);
        unsigned long long col_idx_accumulator = 0;
        for (unsigned long long j = 0; j < 7; ++j) {
            col_idx_accumulator += dist(gen);
            col_idx.push_back(col_idx_accumulator);
            matrix_value.push_back(value_dist(gen));
        }
    }
    CPU_matrix_CSR<double> cpu_matrix(row_ptr, col_idx, matrix_value, matrix_value.size(), NODE_NUM, NODE_NUM);
    std::vector<double> vector_value;
    for (unsigned long long i = 0; i < NODE_NUM; ++i) {
        vector_value.push_back(value_dist(gen));
    }
    CPU_matrix<double> cpu_vector(vector_value, NODE_NUM, 1);
    set_time_stamp("CPU SPMV start");
    CPU_matrix<double> cpu_result = cpu_matrix * cpu_vector;
    for (unsigned long long i = 0; i < 99; ++i) {
        cpu_result = cpu_matrix * cpu_result;
    }
    set_time_stamp("CPU SPMV end");
    std::cout << "CPU SPMV: " << get_time_duration("CPU SPMV start", "CPU SPMV end") << " seconds." << std::endl;
}

void CPU_manager::run_openmp_sparse_matrix_vector_multiply() {
    const unsigned long long NODE_NUM = 10000000;
    std::vector<unsigned long long> row_ptr;
    std::vector<unsigned long long> col_idx;
    std::vector<double> matrix_value;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist(1, 1000000);
    std::uniform_real_distribution<double> value_dist(1.0, 10.0);
    row_ptr.push_back(0);
    for (unsigned long long i = 1; i <= NODE_NUM; ++i) {
        row_ptr.push_back(i * 7);
        unsigned long long col_idx_accumulator = 0;
        for (unsigned long long j = 0; j < 7; ++j) {
            col_idx_accumulator += dist(gen);
            col_idx.push_back(col_idx_accumulator);
            matrix_value.push_back(value_dist(gen));
        }
    }
    CPU_matrix_CSR<double> cpu_matrix(row_ptr, col_idx, matrix_value, matrix_value.size(), NODE_NUM, NODE_NUM);
    std::vector<double> vector_value;
    for (unsigned long long i = 0; i < NODE_NUM; ++i) {
        vector_value.push_back(value_dist(gen));
    }
    CPU_matrix<double> cpu_vector(vector_value, NODE_NUM, 1);
    set_time_stamp("OpenMP SPMV start");
    CPU_matrix<double> cpu_result = openmp_multiply(cpu_matrix, cpu_vector);
    for (unsigned long long i = 0; i < 99; ++i) {
        cpu_result = openmp_multiply(cpu_matrix, cpu_result);
    }
    set_time_stamp("OpenMP SPMV end");
    std::cout << "OpenMP SPMV: " << get_time_duration("OpenMP SPMV start", "OpenMP SPMV end") << " seconds." << std::endl;
}

void CPU_manager::run_GPU_sparse_matrix_vector_multiply() {
    const unsigned long long NODE_NUM = 10000000;
    std::vector<unsigned long long> row_ptr;
    std::vector<unsigned long long> col_idx;
    std::vector<double> matrix_value;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist(1, 1000000);
    std::uniform_real_distribution<double> value_dist(1.0, 10.0);
    row_ptr.push_back(0);
    for (unsigned long long i = 1; i <= NODE_NUM; ++i) {
        row_ptr.push_back(i * 7);
        unsigned long long col_idx_accumulator = 0;
        for (unsigned long long j = 0; j < 7; ++j) {
            col_idx_accumulator += dist(gen);
            col_idx.push_back(col_idx_accumulator);
            matrix_value.push_back(value_dist(gen));
        }
    }
    GPU_matrix_CSR<double> gpu_matrix;
    gpu_matrix.construct_from(row_ptr, col_idx, matrix_value, matrix_value.size(), NODE_NUM, NODE_NUM);
    std::vector<double> vector_value;
    for (unsigned long long i = 0; i < NODE_NUM; ++i) {
        vector_value.push_back(value_dist(gen));
    }
    GPU_matrix<double> gpu_vector;
    gpu_vector.construct_from(vector_value, NODE_NUM, 1);
    GPU_matrix<double> gpu_result = multiply_2d(gpu_matrix, gpu_vector);
    for (unsigned long long i = 0; i < 99; ++i) {
        gpu_result = multiply_2d(gpu_matrix, gpu_result);
    }
    CPU_matrix<double> cpu_result;
    cpu_result.allocate(NODE_NUM, 1);
    gpu_result.copy_to(cpu_result);
    gpu_matrix.free();
    gpu_vector.free();
    gpu_result.free();
}

CPU_manager::CPU_manager() {}

CPU_manager::~CPU_manager() {}
