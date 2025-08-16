#include <stdexcept>

template<typename T>
__global__ void add_kernel(const T* summand, const T addend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        result[idx] = summand[idx] + addend;
    }
}

template<typename T>
__global__ void add_kernel(const T* summand, const T* addend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        result[idx] = summand[idx] + addend[idx];
    }
}

template<typename T>
__global__ void add_kernel_2d(const T* summand, const T addend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = summand[row * cols + col] + addend;
    }
}

template<typename T>
__global__ void add_kernel_2d(const T* summand, const T* addend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = summand[row * cols + col] + addend[row * cols + col];
    }
}

template<typename T>
__global__ void subtract_kernel(const T* minuend, const T subtrahend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        result[idx] = minuend[idx] - subtrahend;
    }
}

template<typename T>
__global__ void subtract_kernel(const T* minuend, const T* subtrahend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        result[idx] = minuend[idx] - subtrahend[idx];
    }
}

template<typename T>
__global__ void subtract_kernel_2d(const T* minuend, const T subtrahend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = minuend[row * cols + col] - subtrahend;
    }
}

template<typename T>
__global__ void subtract_kernel_2d(const T* minuend, const T* subtrahend, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = minuend[row * cols + col] - subtrahend[row * cols + col];
    }
}

template<typename T>
__global__ void multiply_kernel(const T* multiplicand, const T multiplier, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        result[idx] = multiplicand[idx] * multiplier;
    }
}

template<typename T>
__global__ void multiply_kernel(const T* multiplicand, const T* multiplier, T* result, unsigned long long rows, unsigned long long cols_, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row = idx / cols;
    unsigned long long col = idx % cols;
    if (row < rows && col < cols) {
        result[row * cols + col] = T();
        for (unsigned long long k = 0; k < cols_; ++k) {
            result[row * cols + col] += multiplicand[row * cols_ + k] * multiplier[k * cols + col];
        }
    }
}

template<typename T>
__global__ void multiply_kernel_2d(const T* multiplicand, const T multiplier, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = multiplicand[row * cols + col] * multiplier;
    }
}

template<typename T>
__global__ void multiply_kernel_2d(const T* multiplicand, const T* multiplier, T* result, unsigned long long rows, unsigned long long cols_, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = T();
        for (unsigned long long k = 0; k < cols_; ++k) {
            result[row * cols + col] += multiplicand[row * cols_ + k] * multiplier[k * cols + col];
        }
    }
}

template<typename T>
__global__ void transpose_kernel(const T* matrix, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row = idx / cols;
    unsigned long long col = idx % cols;
    if (row < rows && col < cols) {
        result[col * rows + row] = matrix[idx];
    }
}

template<typename T>
__global__ void transpose_kernel_2d(const T* matrix, T* result, unsigned long long rows, unsigned long long cols) {
    unsigned long long row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[col * rows + row] = matrix[row * cols + col];
    }
}

template<typename T>
void GPU_matrix<T>::allocate() {
    cudaMalloc(&this->device_data, this->rows * this->cols * sizeof(T));
}

template<typename T>
void GPU_matrix<T>::allocate_async(cudaStream_t& stream) {
    cudaMallocAsync(&this->device_data, this->rows * this->cols * sizeof(T), stream);
}

template<typename T>
void GPU_matrix<T>::copy_to(GPU_matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for copy.");
    }
    cudaMemcpy(other.device_data, this->device_data, this->rows * this->cols * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
void GPU_matrix<T>::copy_to(CPU_matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for copy.");
    }
    cudaMemcpy(other.data, this->device_data, this->rows * this->cols * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void GPU_matrix<T>::copy_to_async(cudaStream_t& stream, GPU_matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for asynchronous copy.");
    }
    cudaMemcpyAsync(other.device_data, this->device_data, this->rows * this->cols * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
void GPU_matrix<T>::copy_to_async(cudaStream_t& stream, CPU_matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for asynchronous copy.");
    }
    cudaMemcpyAsync(other.data, this->device_data, this->rows * this->cols * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void GPU_matrix<T>::copy_from(const GPU_matrix<T>& other) {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for copy.");
    }
    cudaMemcpy(this->device_data, other.device_data, this->rows * this->cols * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
void GPU_matrix<T>::copy_from(const CPU_matrix<T>& other) {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for copy.");
    }
    cudaMemcpy(this->device_data, other.data, this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPU_matrix<T>::copy_from_async(cudaStream_t& stream, const GPU_matrix<T>& other) {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for copy.");
    }
    cudaMemcpyAsync(this->device_data, other.device_data, this->rows * this->cols * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
void GPU_matrix<T>::copy_from_async(cudaStream_t& stream, const CPU_matrix<T>& other) {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for asynchronous copy.");
    }
    cudaMemcpyAsync(this->device_data, other.data, this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void GPU_matrix<T>::construct_from(const T* data, unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
    allocate();
    cudaMemcpy(this->device_data, data, this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPU_matrix<T>::construct_from(const T** data, unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
    allocate();
    std::vector<T> flat_data(this->rows * this->cols);
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            flat_data[i * this->cols + j] = data[i][j];
        }
    }
    cudaMemcpy(this->device_data, flat_data.data(), this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPU_matrix<T>::construct_from(const std::vector<T>& vec, unsigned long long rows, unsigned long long cols) {
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("Vector size does not match matrix dimensions.");
    }
    this->rows = rows;
    this->cols = cols;
    allocate();
    cudaMemcpy(this->device_data, vec.data(), this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPU_matrix<T>::construct_from(const std::vector<std::vector<T>>& vec, unsigned long long rows, unsigned long long cols) {
    if (vec.size() != rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions.");
    }
    for (const auto& row : vec) {
        if (row.size() != cols) {
            throw std::invalid_argument("Vector dimensions do not match matrix dimensions.");
        }
    }
    this->rows = rows;
    this->cols = cols;
    allocate();
    std::vector<T> flat_data(this->rows * this->cols);
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            flat_data[i * this->cols + j] = vec[i][j];
        }
    }
    cudaMemcpy(this->device_data, flat_data.data(), this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPU_matrix<T>::construct_from_async(cudaStream_t& stream, const T* data, unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
    allocate_async(stream);
    cudaMemcpyAsync(this->device_data, data, this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void GPU_matrix<T>::construct_from_async(cudaStream_t& stream, const T** data, unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
    allocate_async(stream);
    std::vector<T> flat_data(this->rows * this->cols);
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            flat_data[i * this->cols + j] = data[i][j];
        }
    }
    cudaMemcpyAsync(this->device_data, flat_data.data(), this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void GPU_matrix<T>::construct_from_async(cudaStream_t& stream, const std::vector<T>& vec, unsigned long long rows, unsigned long long cols) {
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("Vector size does not match matrix dimensions.");
    }
    this->rows = rows;
    this->cols = cols;
    allocate_async(stream);
    cudaMemcpyAsync(this->device_data, vec.data(), this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void GPU_matrix<T>::construct_from_async(cudaStream_t& stream, const std::vector<std::vector<T>>& vec, unsigned long long rows, unsigned long long cols) {
    if (vec.size() != rows) {
        throw std::invalid_argument("Vector dimensions do not match matrix dimensions.");
    }
    for (const auto& row : vec) {
        if (row.size() != cols) {
            throw std::invalid_argument("Vector dimensions do not match matrix dimensions.");
        }
    }
    this->rows = rows;
    this->cols = cols;
    allocate_async(stream);
    std::vector<T> flat_data(this->rows * this->cols);
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            flat_data[i * this->cols + j] = vec[i][j];
        }
    }
    cudaMemcpyAsync(this->device_data, flat_data.data(), this->rows * this->cols * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
GPU_matrix<T> add(const GPU_matrix<T>& summand, const T& addend) {
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((summand.rows * summand.cols + thread_per_block.x - 1) / thread_per_block.x);
    add_kernel<<<block_num, thread_per_block>>>(summand.data, addend, result.data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add(const GPU_matrix<T>& summand, const GPU_matrix<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((summand.rows * summand.cols + thread_per_block.x - 1) / thread_per_block.x);
    add_kernel<<<block_num, thread_per_block>>>(summand.data, addend.data, result.data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add_2d(const GPU_matrix<T>& summand, const T& addend) {
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((summand.rows + thread_per_block.x - 1) / thread_per_block.x, (summand.cols + thread_per_block.y - 1) / thread_per_block.y);
    add_kernel_2d<<<block_num, thread_per_block>>>(summand.device_data, addend, result.device_data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add_2d(const GPU_matrix<T>& summand, const GPU_matrix<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((summand.rows + thread_per_block.x - 1) / thread_per_block.x, (summand.cols + thread_per_block.y - 1) / thread_per_block.y);
    add_kernel_2d<<<block_num, thread_per_block>>>(summand.device_data, addend.device_data, result.device_data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add_async(cudaStream_t& stream, const GPU_matrix<T>& summand, const T& addend) {
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((summand.rows * summand.cols + thread_per_block.x - 1) / thread_per_block.x);
    add_kernel<<<block_num, thread_per_block, 0, stream>>>(summand.device_data, addend, result.device_data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add_async(cudaStream_t& stream, const GPU_matrix<T>& summand, const GPU_matrix<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((summand.rows * summand.cols + thread_per_block.x - 1) / thread_per_block.x);
    add_kernel<<<block_num, thread_per_block, 0, stream>>>(summand.device_data, addend.device_data, result.device_data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add_2d_async(cudaStream_t& stream, const GPU_matrix<T>& summand, const T& addend) {
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((summand.rows + thread_per_block.x - 1) / thread_per_block.x, (summand.cols + thread_per_block.y - 1) / thread_per_block.y);
    add_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(summand.device_data, addend, result.device_data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> add_2d_async(cudaStream_t& stream, const GPU_matrix<T>& summand, const GPU_matrix<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    GPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((summand.rows + thread_per_block.x - 1) / thread_per_block.x, (summand.cols + thread_per_block.y - 1) / thread_per_block.y);
    add_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(summand.device_data, addend.device_data, result.device_data, summand.rows, summand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract(const GPU_matrix<T>& minuend, const T& subtrahend) {
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((minuend.rows * minuend.cols + thread_per_block.x - 1) / thread_per_block.x);
    subtract_kernel<<<block_num, thread_per_block>>>(minuend.device_data, subtrahend, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract(const GPU_matrix<T>& minuend, const GPU_matrix<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((minuend.rows * minuend.cols + thread_per_block.x - 1) / thread_per_block.x);
    subtract_kernel<<<block_num, thread_per_block>>>(minuend.device_data, subtrahend.device_data, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract_2d(const GPU_matrix<T>& minuend, const T& subtrahend) {
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((minuend.rows + thread_per_block.x - 1) / thread_per_block.x, (minuend.cols + thread_per_block.y - 1) / thread_per_block.y);
    subtract_kernel_2d<<<block_num, thread_per_block>>>(minuend.device_data, subtrahend, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract_2d(const GPU_matrix<T>& minuend, const GPU_matrix<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((minuend.rows + thread_per_block.x - 1) / thread_per_block.x, (minuend.cols + thread_per_block.y - 1) / thread_per_block.y);
    subtract_kernel_2d<<<block_num, thread_per_block>>>(minuend.device_data, subtrahend.device_data, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract_async(cudaStream_t& stream, const GPU_matrix<T>& minuend, const T& subtrahend) {
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((minuend.rows * minuend.cols + thread_per_block.x - 1) / thread_per_block.x);
    subtract_kernel<<<block_num, thread_per_block, 0, stream>>>(minuend.device_data, subtrahend, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract_async(cudaStream_t& stream, const GPU_matrix<T>& minuend, const GPU_matrix<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((minuend.rows * minuend.cols + thread_per_block.x - 1) / thread_per_block.x);
    subtract_kernel<<<block_num, thread_per_block, 0, stream>>>(minuend.device_data, subtrahend.device_data, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract_2d_async(cudaStream_t& stream, const GPU_matrix<T>& minuend, const T& subtrahend) {
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((minuend.rows + thread_per_block.x - 1) / thread_per_block.x, (minuend.cols + thread_per_block.y - 1) / thread_per_block.y);
    subtract_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(minuend.device_data, subtrahend, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> subtract_2d_async(cudaStream_t& stream, const GPU_matrix<T>& minuend, const GPU_matrix<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    GPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((minuend.rows + thread_per_block.x - 1) / thread_per_block.x, (minuend.cols + thread_per_block.y - 1) / thread_per_block.y);
    subtract_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(minuend.device_data, subtrahend.device_data, result.device_data, minuend.rows, minuend.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply(const GPU_matrix<T>& multiplicand, const T& multiplier) {
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplicand.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((multiplicand.rows * multiplicand.cols + thread_per_block.x - 1) / thread_per_block.x);
    multiply_kernel<<<block_num, thread_per_block>>>(multiplicand.device_data, multiplier, result.device_data, multiplicand.rows, multiplicand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply(const GPU_matrix<T>& multiplicand, const GPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication.");
    }
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((multiplicand.rows * multiplier.cols + thread_per_block.x - 1) / thread_per_block.x);
    multiply_kernel<<<block_num, thread_per_block>>>(multiplicand.device_data, multiplier.device_data, result.device_data, multiplicand.rows, multiplicand.cols, multiplier.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_2d(const GPU_matrix<T>& multiplicand, const T& multiplier) {
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplicand.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((multiplicand.rows + thread_per_block.x - 1) / thread_per_block.x, (multiplicand.cols + thread_per_block.y - 1) / thread_per_block.y);
    multiply_kernel_2d<<<block_num, thread_per_block>>>(multiplicand.device_data, multiplier, result.device_data, multiplicand.rows, multiplicand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_2d(const GPU_matrix<T>& multiplicand, const GPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication.");
    }
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((multiplicand.rows + thread_per_block.x - 1) / thread_per_block.x, (multiplier.cols + thread_per_block.y - 1) / thread_per_block.y);
    multiply_kernel_2d<<<block_num, thread_per_block>>>(multiplicand.device_data, multiplier.device_data, result.device_data, multiplicand.rows, multiplicand.cols, multiplier.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_async(cudaStream_t& stream, const GPU_matrix<T>& multiplicand, const T& multiplier) {
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplicand.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((multiplicand.rows * multiplicand.cols + thread_per_block.x - 1) / thread_per_block.x);
    multiply_kernel<<<block_num, thread_per_block, 0, stream>>>(multiplicand.data, multiplier, result.data, multiplicand.rows, multiplicand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_async(cudaStream_t& stream, const GPU_matrix<T>& multiplicand, const GPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication.");
    }
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((multiplicand.rows * multiplier.cols + thread_per_block.x - 1) / thread_per_block.x);
    multiply_kernel<<<block_num, thread_per_block, 0, stream>>>(multiplicand.device_data, multiplier.device_data, result.device_data, multiplicand.rows, multiplicand.cols, multiplier.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_2d_async(cudaStream_t& stream, const GPU_matrix<T>& multiplicand, const T& multiplier) {
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplicand.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((multiplicand.rows + thread_per_block.x - 1) / thread_per_block.x, (multiplicand.cols + thread_per_block.y - 1) / thread_per_block.y);
    multiply_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(multiplicand.data, multiplier, result.data, multiplicand.rows, multiplicand.cols);
    return result;
}

template<typename T>
GPU_matrix<T> multiply_2d_async(cudaStream_t& stream, const GPU_matrix<T>& multiplicand, const GPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication.");
    }
    GPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((multiplicand.rows + thread_per_block.x - 1) / thread_per_block.x, (multiplier.cols + thread_per_block.y - 1) / thread_per_block.y);
    multiply_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(multiplicand.device_data, multiplier.device_data, result.device_data, multiplicand.rows, multiplicand.cols, multiplier.cols);
    return result;
}

template<typename T>
GPU_matrix<T> transpose(const GPU_matrix<T>& matrix) {
    GPU_matrix<T> result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((matrix.rows * matrix.cols + thread_per_block.x - 1) / thread_per_block.x);
    transpose_kernel<<<block_num, thread_per_block>>>(matrix.device_data, result.device_data, matrix.rows, matrix.cols);
    return result;
}

template<typename T>
GPU_matrix<T> transpose_2d(const GPU_matrix<T>& matrix) {
    GPU_matrix<T> result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.allocate();
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((matrix.rows + thread_per_block.x - 1) / thread_per_block.x, (matrix.cols + thread_per_block.y - 1) / thread_per_block.y);
    transpose_kernel_2d<<<block_num, thread_per_block>>>(matrix.device_data, result.device_data, matrix.rows, matrix.cols);
    return result;
}

template<typename T>
GPU_matrix<T> transpose_async(cudaStream_t& stream, const GPU_matrix<T>& matrix) {
    GPU_matrix<T> result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM);
    dim3 block_num((matrix.rows * matrix.cols + thread_per_block.x - 1) / thread_per_block.x);
    transpose_kernel<<<block_num, thread_per_block, 0, stream>>>(matrix.device_data, result.device_data, matrix.rows, matrix.cols);
    return result;
}

template<typename T>
GPU_matrix<T> transpose_2d_async(cudaStream_t& stream, const GPU_matrix<T>& matrix) {
    GPU_matrix<T> result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.allocate_async(stream);
    dim3 thread_per_block(THREAD_NUM_SQUARE_ROOT, THREAD_NUM_SQUARE_ROOT);
    dim3 block_num((matrix.rows + thread_per_block.x - 1) / thread_per_block.x, (matrix.cols + thread_per_block.y - 1) / thread_per_block.y);
    transpose_kernel_2d<<<block_num, thread_per_block, 0, stream>>>(matrix.device_data, result.device_data, matrix.rows, matrix.cols);
    return result;
}
