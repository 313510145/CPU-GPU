#include "CPU_matrix_CSR.h"

#include <stdexcept>

template<typename T>
__global__ void multiply_kernel(const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, const T* multiplier, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        unsigned long long row = idx / cols;
        unsigned long long col = idx % cols;
        T sum = T();
        unsigned long long end = row_ptr[row + 1];
        for (unsigned long long j = row_ptr[row]; j < end; ++j) {
            unsigned long long local_col_idx = col_idx[j];
            sum += values[j] * multiplier[local_col_idx * cols + col];
        }
        result[idx] = sum;
    }
}

template<typename T>
__global__ void multiply_kernel_2d(const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, const T* multiplier, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        T sum = T();
        unsigned long long end = row_ptr[row + 1];
        for (unsigned long long j = row_ptr[row]; j < end; ++j) {
            unsigned long long local_col_idx = col_idx[j];
            sum += values[j] * multiplier[local_col_idx * cols + col];
        }
        result[row * cols + col] = sum;
    }
}

template<typename T>
void GPU_matrix_CSR<T>::allocate() {
    cudaMalloc(&this->row_ptr, (this->rows + 1) * sizeof(unsigned long long));
    cudaMalloc(&this->col_idx, this->nnz * sizeof(unsigned long long));
    cudaMalloc(&this->values, this->nnz * sizeof(T));
}

template<typename T>
void GPU_matrix_CSR<T>::copy_to(CPU_matrix_CSR<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for copy.");
    }
    cudaMemcpy(other.row_ptr, this->row_ptr, (this->rows + 1) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(other.col_idx, this->col_idx, this->nnz * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(other.values, this->values, this->nnz * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void GPU_matrix_CSR<T>::construct_from(const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, unsigned long long nnz, unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    allocate();
    cudaMemcpy(this->row_ptr, row_ptr, (this->rows + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(this->col_idx, col_idx, this->nnz * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(this->values, values, this->nnz * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPU_matrix_CSR<T>::construct_from(const std::vector<unsigned long long>& row_ptr, const std::vector<unsigned long long>& col_idx, const std::vector<T>& values, unsigned long long nnz, unsigned long long rows, unsigned long long cols) {
    if (row_ptr.size() != rows + 1 || col_idx.size() != nnz || values.size() != nnz) {
        throw std::invalid_argument("Input vectors do not match the expected sizes for CSR format.");
    }
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    allocate();
    cudaMemcpy(this->row_ptr, row_ptr.data(), (this->rows + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(this->col_idx, col_idx.data(), this->nnz * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(this->values, values.data(), this->nnz * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
GPU_matrix<T> multiply(const GPU_matrix_CSR<T>& multiplicand, const GPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((multiplicand.rows * multiplier.cols + thread_per_block.x - 1) / thread_per_block.x);
    multiply_kernel<<<block_num, thread_per_block>>>(multiplicand.row_ptr, multiplicand.col_idx, multiplicand.values, multiplier.device_data, result.device_data, multiplicand.rows, multiplier.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_2d(const GPU_matrix_CSR<T>& multiplicand, const GPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((multiplicand.rows + thread_per_block.x - 1) / thread_per_block.x, (multiplier.cols + thread_per_block.y - 1) / thread_per_block.y);
    multiply_kernel_2d<<<block_num, thread_per_block>>>(multiplicand.row_ptr, multiplicand.col_idx, multiplicand.values, multiplier.device_data, result.device_data, multiplicand.rows, multiplier.cols);
    return result;
}
