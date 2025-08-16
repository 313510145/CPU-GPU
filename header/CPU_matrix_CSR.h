#ifndef CPU_MATRIX_CSR_H
#define CPU_MATRIX_CSR_H

#include "matrix_base.h"

#include <iostream>
#include <vector>

template<typename T>
class CPU_matrix;

template<typename T>
class CPU_matrix_CSC;

template<typename T>
class GPU_matrix_CSR;

template<typename T>
class CPU_matrix_CSR: public matrix_base<T> {
    public:
        void print_all(std::ostream& os) const;
        CPU_matrix_CSR& operator=(const CPU_matrix_CSR<T>& other);
        template<typename U> friend bool operator==(const CPU_matrix_CSR<U>& lhs, const CPU_matrix_CSR<U>& rhs);
        template<typename U> friend bool operator!=(const CPU_matrix_CSR<U>& lhs, const CPU_matrix_CSR<U>& rhs);
        template<typename U> friend CPU_matrix_CSR<U> operator+(const CPU_matrix_CSR<U>& summand, const U& addend);
        template<typename U> friend CPU_matrix_CSR<U> operator+(const CPU_matrix_CSR<U>& summand, const CPU_matrix_CSR<U>& addend);
        template<typename U> friend CPU_matrix_CSR<U> operator-(const CPU_matrix_CSR<U>& minuend, const U& subtrahend);
        template<typename U> friend CPU_matrix_CSR<U> operator-(const CPU_matrix_CSR<U>& minuend, const CPU_matrix_CSR<U>& subtrahend);
        template<typename U> friend CPU_matrix_CSR<U> operator*(const CPU_matrix_CSR<U>& multiplicand, const U& multiplier);
        template<typename U> friend CPU_matrix_CSR<U> operator*(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix_CSC<U>& multiplier);
        template<typename U> friend CPU_matrix_CSR<U> operator*(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix_CSR<U>& multiplier);
        template<typename U> friend CPU_matrix_CSC<U> operator*(const CPU_matrix_CSC<U>& multiplicand, const CPU_matrix_CSC<U>& multiplier);
        template<typename U> friend CPU_matrix<U> operator*(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix<U>& multiplier);
        template<typename U> friend CPU_matrix_CSC<U> to_csc(const CPU_matrix_CSR<U>& matrix);
        template<typename U> friend CPU_matrix_CSR<U> to_csr(const CPU_matrix_CSC<U>& matrix);
        template<typename U> friend CPU_matrix_CSR<U> transpose(const CPU_matrix_CSR<U>& matrix);
        template<typename U> friend bool openmp_equal(const CPU_matrix_CSR<U>& lhs, const CPU_matrix_CSR<U>& rhs);
        template<typename U> friend bool openmp_not_equal(const CPU_matrix_CSR<U>& lhs, const CPU_matrix_CSR<U>& rhs);
        template<typename U> friend CPU_matrix_CSR<U> openmp_add(const CPU_matrix_CSR<U>& summand, const U& addend);
        template<typename U> friend CPU_matrix_CSR<U> openmp_add(const CPU_matrix_CSR<U>& summand, const CPU_matrix_CSR<U>& addend);
        template<typename U> friend CPU_matrix_CSR<U> openmp_subtract(const CPU_matrix_CSR<U>& minuend, const U& subtrahend);
        template<typename U> friend CPU_matrix_CSR<U> openmp_subtract(const CPU_matrix_CSR<U>& minuend, const CPU_matrix_CSR<U>& subtrahend);
        template<typename U> friend CPU_matrix_CSR<U> openmp_multiply(const CPU_matrix_CSR<U>& multiplicand, const U& multiplier);
        template<typename U> friend CPU_matrix_CSR<U> openmp_multiply(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix_CSC<U>& multiplier);
        template<typename U> friend CPU_matrix_CSR<U> openmp_multiply(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix_CSR<U>& multiplier);
        template<typename U> friend CPU_matrix_CSC<U> openmp_multiply(const CPU_matrix_CSC<U>& multiplicand, const CPU_matrix_CSC<U>& multiplier);
        template<typename U> friend CPU_matrix<U> openmp_multiply(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix<U>& multiplier);
        // template<typename U> friend CPU_matrix_CSC<U> openmp_to_csc(const CPU_matrix_CSR<U>& matrix);
        // template<typename U> friend CPU_matrix_CSR<U> openmp_to_csr(const CPU_matrix_CSC<U>& matrix);
        // template<typename U> friend CPU_matrix_CSR<U> openmp_transpose(const CPU_matrix_CSR<U>& matrix);
        CPU_matrix_CSR();
        CPU_matrix_CSR(unsigned long long rows, unsigned long long cols);
        CPU_matrix_CSR(const CPU_matrix_CSR<T>& other);
        CPU_matrix_CSR(CPU_matrix_CSR<T>&& other);
        CPU_matrix_CSR(const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, unsigned long long nnz, unsigned long long rows, unsigned long long cols);
        CPU_matrix_CSR(const std::vector<unsigned long long>& row_ptr, const std::vector<unsigned long long>& col_idx, const std::vector<T>& values, unsigned long long nnz, unsigned long long rows, unsigned long long cols);
        ~CPU_matrix_CSR() override;
        friend class GPU_matrix_CSR<T>;
    private:
        unsigned long long* row_ptr;
        unsigned long long* col_idx;
        T* values;
        unsigned long long nnz;
};

#include "CPU_matrix_CSR.tpp"

#endif  // CPU_MATRIX_CSR_H
