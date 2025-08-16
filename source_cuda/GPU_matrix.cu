#include "GPU_matrix.cuh"
#include "CPU_matrix.h"

template<typename T>
void GPU_matrix<T>::free() {
    if (this->device_data) {
        cudaFree(this->device_data);
        this->device_data = nullptr;
    }
}

template<typename T>
void GPU_matrix<T>::free_async(cudaStream_t& stream) {
    if (this->device_data) {
        cudaFreeAsync(this->device_data, stream);
        this->device_data = nullptr;
    }
}

template<typename T>
GPU_matrix<T>::GPU_matrix(): matrix_base<T>() {
    this->device_data = nullptr;
}

template<typename T>
GPU_matrix<T>::~GPU_matrix() {}

template class GPU_matrix<double>;
