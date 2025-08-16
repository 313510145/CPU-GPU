#ifndef GPU_MATRIX_CSR_CUH
#define GPU_MATRIX_CSR_CUH

#include "matrix_base.h"

#include <cuda_runtime.h>
#include <vector>

template<typename T>
class GPU_matrix;

template<typename T>
class CPU_matrix_CSR;

template<typename T>
class GPU_matrix_CSR: public matrix_base<T> {
    public:
        void allocate();
        void allocate_async(cudaStream_t& stream);
        void free();
        void free_async(cudaStream_t& stream);
        void copy_to(GPU_matrix_CSR<T>& other) const;
        void copy_to(CPU_matrix_CSR<T>& other) const;
        void copy_to_async(cudaStream_t& stream, GPU_matrix_CSR<T>& other) const;
        void copy_to_async(cudaStream_t& stream, CPU_matrix_CSR<T>& other) const;
        void copy_from(const GPU_matrix_CSR<T>& other);
        void copy_from(const CPU_matrix_CSR<T>& other);
        void copy_from_async(cudaStream_t& stream, const GPU_matrix_CSR<T>& other);
        void copy_from_async(cudaStream_t& stream, const CPU_matrix_CSR<T>& other);
        void construct_from(const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, unsigned long long nnz, unsigned long long rows, unsigned long long cols);
        void construct_from(const std::vector<unsigned long long>& row_ptr, const std::vector<unsigned long long>& col_idx, const std::vector<T>& values, unsigned long long nnz, unsigned long long rows, unsigned long long cols);
        void construct_from_async(cudaStream_t& stream, const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, unsigned long long nnz, unsigned long long rows, unsigned long long cols);
        void construct_from_async(cudaStream_t& stream, const std::vector<unsigned long long>& row_ptr, const std::vector<unsigned long long>& col_idx, const std::vector<T>& values, unsigned long long nnz, unsigned long long rows, unsigned long long cols);
        template<typename U> friend GPU_matrix_CSR<U> add(const GPU_matrix_CSR<U>& summand, const T& addend);
        template<typename U> friend GPU_matrix_CSR<U> add(const GPU_matrix_CSR<U>& summand, const GPU_matrix_CSR<U>& addend);
        template<typename U> friend GPU_matrix_CSR<U> add_2d(const GPU_matrix_CSR<U>& summand, const T& addend);
        template<typename U> friend GPU_matrix_CSR<U> add_2d(const GPU_matrix_CSR<U>& summand, const GPU_matrix_CSR<U>& addend);
        template<typename U> friend GPU_matrix_CSR<U> add_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& summand, const T& addend);
        template<typename U> friend GPU_matrix_CSR<U> add_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& summand, const GPU_matrix_CSR<U>& addend);
        template<typename U> friend GPU_matrix_CSR<U> add_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& summand, const T& addend);
        template<typename U> friend GPU_matrix_CSR<U> add_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& summand, const GPU_matrix_CSR<U>& addend);
        template<typename U> friend GPU_matrix_CSR<U> subtract(const GPU_matrix_CSR<U>& minuend, const T& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract(const GPU_matrix_CSR<U>& minuend, const GPU_matrix_CSR<U>& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract_2d(const GPU_matrix_CSR<U>& minuend, const T& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract_2d(const GPU_matrix_CSR<U>& minuend, const GPU_matrix_CSR<U>& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& minuend, const T& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& minuend, const GPU_matrix_CSR<U>& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& minuend, const T& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> subtract_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& minuend, const GPU_matrix_CSR<U>& subtrahend);
        template<typename U> friend GPU_matrix_CSR<U> multiply(const GPU_matrix_CSR<U>& multiplicand, const T& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply(const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix_CSR<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply(const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply_2d(const GPU_matrix_CSR<U>& multiplicand, const T& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply_2d(const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix_CSR<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d(const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& multiplicand, const T& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix_CSR<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& multiplicand, const T& multiplier);
        template<typename U> friend GPU_matrix_CSR<U> multiply_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix_CSR<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d_async(cudaStream_t& stream, const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix<U>& multiplier);
        GPU_matrix_CSR();
        ~GPU_matrix_CSR() override;
    private:
        unsigned long long* row_ptr;
        unsigned long long* col_idx;
        T* values;
        unsigned long long nnz;
};

#include "GPU_matrix_CSR.inl"

#endif  // GPU_MATRIX_CSR_CUH
