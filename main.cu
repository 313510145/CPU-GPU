#include "CPU_manager.cuh"

int main() {
    CPU_manager manager;
    manager.test_CPU_matrix_add();
    manager.test_CPU_matrix_subtract();
    manager.test_CPU_matrix_multiply();
    manager.test_CPU_matrix_transpose();
    manager.test_CPU_matrix_CSR_add();
    manager.test_CPU_matrix_CSR_subtract();
    manager.test_CPU_matrix_CSR_multiply();
    manager.test_CPU_matrix_CSR_transpose();
    manager.test_CPU_matrix_CSC_add();
    manager.test_CPU_matrix_CSC_subtract();
    manager.test_CPU_matrix_CSC_transpose();
    manager.test_GPU_matrix_add();
    manager.test_GPU_matrix_subtract();
    manager.test_GPU_matrix_multiply();
    manager.test_GPU_matrix_transpose();
    manager.test_GPU_sparse_matrix_vector_multiply();
    return 0;
}
