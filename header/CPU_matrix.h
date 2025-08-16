#ifndef CPU_MATRIX_H
#define CPU_MATRIX_H

#include "matrix_base.h"

#include <iostream>
#include <vector>

template<typename T>
class CPU_matrix_CSR;

template<typename T>
class GPU_matrix;

template<typename T>
class CPU_matrix: public matrix_base<T> {
    public:
        void print_all(std::ostream& os) const;
        void allocate(unsigned long long rows, unsigned long long cols);
        CPU_matrix& operator=(const CPU_matrix<T>& other);
        template<typename U> friend bool operator==(const CPU_matrix<U>& lhs, const CPU_matrix<U>& rhs);
        template<typename U> friend bool operator!=(const CPU_matrix<U>& lhs, const CPU_matrix<U>& rhs);
        template<typename U> friend CPU_matrix<U> operator+(const CPU_matrix<U>& summand, const U& addend);
        template<typename U> friend CPU_matrix<U> operator+(const CPU_matrix<U>& summand, const CPU_matrix<U>& addend);
        template<typename U> friend CPU_matrix<U> operator-(const CPU_matrix<U>& minuend, const U& subtrahend);
        template<typename U> friend CPU_matrix<U> operator-(const CPU_matrix<U>& minuend, const CPU_matrix<U>& subtrahend);
        template<typename U> friend CPU_matrix<U> operator*(const CPU_matrix<U>& multiplicand, const U& multiplier);
        template<typename U> friend CPU_matrix<U> operator*(const CPU_matrix<U>& multiplicand, const CPU_matrix<U>& multiplier);
        template<typename U> friend CPU_matrix<U> operator*(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix<U>& multiplier);
        template<typename U> friend CPU_matrix<U> transpose(const CPU_matrix<U>& matrix);
        template<typename U> friend bool openmp_equal(const CPU_matrix<U>& lhs, const CPU_matrix<U>& rhs);
        template<typename U> friend bool openmp_not_equal(const CPU_matrix<U>& lhs, const CPU_matrix<U>& rhs);
        template<typename U> friend CPU_matrix<U> openmp_add(const CPU_matrix<U>& summand, const U& addend);
        template<typename U> friend CPU_matrix<U> openmp_add(const CPU_matrix<U>& summand, const CPU_matrix<U>& addend);
        template<typename U> friend CPU_matrix<U> openmp_subtract(const CPU_matrix<U>& minuend, const U& subtrahend);
        template<typename U> friend CPU_matrix<U> openmp_subtract(const CPU_matrix<U>& minuend, const CPU_matrix<U>& subtrahend);
        template<typename U> friend CPU_matrix<U> openmp_multiply(const CPU_matrix<U>& multiplicand, const U& multiplier);
        template<typename U> friend CPU_matrix<U> openmp_multiply(const CPU_matrix<U>& multiplicand, const CPU_matrix<U>& multiplier);
        template<typename U> friend CPU_matrix<U> openmp_multiply(const CPU_matrix_CSR<U>& multiplicand, const CPU_matrix<U>& multiplier);
        template<typename U> friend CPU_matrix<U> openmp_transpose(const CPU_matrix<U>& matrix);
        CPU_matrix();
        CPU_matrix(const CPU_matrix<T>& other);
        CPU_matrix(CPU_matrix<T>&& other);
        CPU_matrix(const T* data, unsigned long long rows, unsigned long long cols);
        CPU_matrix(const T** data, unsigned long long rows, unsigned long long cols);
        CPU_matrix(const std::vector<T>& vec, unsigned long long rows, unsigned long long cols);
        CPU_matrix(const std::vector<std::vector<T>>& vec, unsigned long long rows, unsigned long long cols);
        ~CPU_matrix() override;
        friend class CPU_matrix_CSR<T>;
        friend class GPU_matrix<T>;
    private:
        T* data;
};

#include "CPU_matrix.tpp"

#endif  // CPU_MATRIX_H
