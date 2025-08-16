#include "GPU_matrix_CSR.cuh"

template<typename T>
void GPU_matrix_CSR<T>::free() {
    if (this->row_ptr) {
        cudaFree(this->row_ptr);
        this->row_ptr = nullptr;
    }
    if (this->col_idx) {
        cudaFree(this->col_idx);
        this->col_idx = nullptr;
    }
    if (this->values) {
        cudaFree(this->values);
        this->values = nullptr;
    }
    this->nnz = 0;
}

template<typename T>
GPU_matrix_CSR<T>::GPU_matrix_CSR(): matrix_base<T>() {
    this->row_ptr = nullptr;
    this->col_idx = nullptr;
    this->values = nullptr;
    this->nnz = 0;
}

template<typename T>
GPU_matrix_CSR<T>::~GPU_matrix_CSR() {}

template class GPU_matrix_CSR<double>;
