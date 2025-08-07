#include "GPU_matrix_CSR.cuh"

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
