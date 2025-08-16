#ifndef GPU_MATRIX_CUH
#define GPU_MATRIX_CUH

#include "matrix_base.h"

#include <cuda_runtime.h>
#include <vector>

template<typename T>
class CPU_matrix;

template<typename T>
class GPU_matrix_CSR;

template<typename T>
class GPU_matrix: public matrix_base<T> {
    public:
        void allocate();
        void allocate_async(cudaStream_t& stream);
        void free();
        void free_async(cudaStream_t& stream);
        void copy_to(GPU_matrix<T>& other) const;
        void copy_to(CPU_matrix<T>& other) const;
        void copy_to_async(cudaStream_t& stream, GPU_matrix<T>& other) const;
        void copy_to_async(cudaStream_t& stream, CPU_matrix<T>& other) const;
        void copy_from(const GPU_matrix<T>& other);
        void copy_from(const CPU_matrix<T>& other);
        void copy_from_async(cudaStream_t& stream, const GPU_matrix<T>& other);
        void copy_from_async(cudaStream_t& stream, const CPU_matrix<T>& other);
        void construct_from(const T* data, unsigned long long rows, unsigned long long cols);
        void construct_from(const T** data, unsigned long long rows, unsigned long long cols);
        void construct_from(const std::vector<T>& vec, unsigned long long rows, unsigned long long cols);
        void construct_from(const std::vector<std::vector<T>>& vec, unsigned long long rows, unsigned long long cols);
        void construct_from_async(cudaStream_t& stream, const T* data, unsigned long long rows, unsigned long long cols);
        void construct_from_async(cudaStream_t& stream, const T** data, unsigned long long rows, unsigned long long cols);
        void construct_from_async(cudaStream_t& stream, const std::vector<T>& vec, unsigned long long rows, unsigned long long cols);
        void construct_from_async(cudaStream_t& stream, const std::vector<std::vector<T>>& vec, unsigned long long rows, unsigned long long cols);
        template<typename U> friend GPU_matrix<U> add(const GPU_matrix<U>& summand, const U& addend);
        template<typename U> friend GPU_matrix<U> add(const GPU_matrix<U>& summand, const GPU_matrix<U>& addend);
        template<typename U> friend GPU_matrix<U> add_2d(const GPU_matrix<U>& summand, const U& addend);
        template<typename U> friend GPU_matrix<U> add_2d(const GPU_matrix<U>& summand, const GPU_matrix<U>& addend);
        template<typename U> friend GPU_matrix<U> add_async(cudaStream_t& stream, const GPU_matrix<U>& summand, const U& addend);
        template<typename U> friend GPU_matrix<U> add_async(cudaStream_t& stream, const GPU_matrix<U>& summand, const GPU_matrix<U>& addend);
        template<typename U> friend GPU_matrix<U> add_2d_async(cudaStream_t& stream, const GPU_matrix<U>& summand, const U& addend);
        template<typename U> friend GPU_matrix<U> add_2d_async(cudaStream_t& stream, const GPU_matrix<U>& summand, const GPU_matrix<U>& addend);
        template<typename U> friend GPU_matrix<U> subtract(const GPU_matrix<U>& minuend, const U& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract(const GPU_matrix<U>& minuend, const GPU_matrix<U>& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract_2d(const GPU_matrix<U>& minuend, const U& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract_2d(const GPU_matrix<U>& minuend, const GPU_matrix<U>& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract_async(cudaStream_t& stream, const GPU_matrix<U>& minuend, const U& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract_async(cudaStream_t& stream, const GPU_matrix<U>& minuend, const GPU_matrix<U>& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract_2d_async(cudaStream_t& stream, const GPU_matrix<U>& minuend, const U& subtrahend);
        template<typename U> friend GPU_matrix<U> subtract_2d_async(cudaStream_t& stream, const GPU_matrix<U>& minuend, const GPU_matrix<U>& subtrahend);
        template<typename U> friend GPU_matrix<U> multiply(const GPU_matrix<U>& multiplicand, const U& multiplier);
        template<typename U> friend GPU_matrix<U> multiply(const GPU_matrix<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply(const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d(const GPU_matrix<U>& multiplicand, const U& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d(const GPU_matrix<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d(const GPU_matrix_CSR<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_async(cudaStream_t& stream, const GPU_matrix<U>& multiplicand, const U& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_async(cudaStream_t& stream, const GPU_matrix<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d_async(cudaStream_t& stream, const GPU_matrix<U>& multiplicand, const U& multiplier);
        template<typename U> friend GPU_matrix<U> multiply_2d_async(cudaStream_t& stream, const GPU_matrix<U>& multiplicand, const GPU_matrix<U>& multiplier);
        template<typename U> friend GPU_matrix<U> transpose(const GPU_matrix<U>& matrix);
        template<typename U> friend GPU_matrix<U> transpose_2d(const GPU_matrix<U>& matrix);
        template<typename U> friend GPU_matrix<U> transpose_async(cudaStream_t& stream, const GPU_matrix<U>& matrix);
        template<typename U> friend GPU_matrix<U> transpose_2d_async(cudaStream_t& stream, const GPU_matrix<U>& matrix);
        GPU_matrix();
        ~GPU_matrix() override;
        friend class GPU_matrix_CSR<T>;
    private:
        T* device_data;
};

#include "GPU_matrix.inl"

#endif  // GPU_MATRIX_CUH
