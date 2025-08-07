#include "CPU_manager.h"
#include "CPU_matrix.h"
#include "CPU_matrix_CSR.h"
#include "GPU_matrix.cuh"

#include <iostream>

int main() {
    // std::vector<unsigned long long> row_ptr = {0, 3, 6, 9};
    // std::vector<unsigned long long> col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    const unsigned long long M = 3;
    const unsigned long long N = 3;
    const unsigned long long R = 1;
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<unsigned long long> row_ptr = {0, 3, 6, 9};
    std::vector<unsigned long long> col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    // CPU_manager cpu_manager;
    CPU_matrix_CSR<double> cpu_matrix_csr(row_ptr, col_idx, values, 9, M, N);
    // CPU_matrix<double> cpu_matrix_a(values, M, N);
    CPU_matrix<double> cpu_matrix_b;
    CPU_matrix<double> cpu_matrix_c(values, M, N);
    // CPU_matrix<double> cpu_matrix_d;
    // CPU_matrix<double> cpu_matrix_e(cpu_matrix_c);
    // cpu_manager.set_time_stamp("CPU MMM start");
    for (unsigned long long i = 0; i < R; ++i) {
        cpu_matrix_b = cpu_matrix_c;
        cpu_matrix_c = openmp_multiply(cpu_matrix_csr, cpu_matrix_b);
    }
    // cpu_manager.set_time_stamp("CPU MMM end");
    // std::cout << cpu_manager.get_time_duration("CPU MMM start", "CPU MMM end") << " s" << std::endl;

    // cpu_manager.set_time_stamp("CPU OpenMP MMM start");
    // for (unsigned long long i = 0; i < R; ++i) {
    //     CPU_matrix<double> cpu_matrix_d(cpu_matrix_e);
    //     cpu_matrix_e = openmp_multiply(cpu_matrix_a, cpu_matrix_d);
    // }
    // cpu_manager.set_time_stamp("CPU OpenMP MMM end");
    // std::cout << cpu_manager.get_time_duration("CPU OpenMP MMM start", "CPU OpenMP MMM end") << " s" << std::endl;
    // if (cpu_matrix_c == cpu_matrix_e) {
    //     std::cout << "OpenMP multiplication is correct." << std::endl;
    // } else {
    //     std::cout << "OpenMP multiplication is incorrect." << std::endl;
    // }

    // GPU_matrix<double> gpu_matrix_a;
    // gpu_matrix_a.construct_from(values, M, N);
    // GPU_matrix<double> gpu_matrix_b;
    // gpu_matrix_b.set_size(N, 1);
    // gpu_matrix_b.allocate();
    // GPU_matrix<double> gpu_matrix_c;
    // gpu_matrix_c.construct_from(values2, N, 1);
    // for (unsigned long long i = 0; i < R; ++i) {
    //     gpu_matrix_c.copy_to(gpu_matrix_b);
    //     gpu_matrix_c = multiply(gpu_matrix_a, gpu_matrix_b);
    // }
    // CPU_matrix<double> cpu_matrix_gpu;
    // cpu_matrix_gpu.allocate(M, 1);
    // gpu_matrix_c.copy_to(cpu_matrix_gpu);
    // if (cpu_matrix_c == cpu_matrix_gpu) {
    //     std::cout << "GPU multiplication is correct." << std::endl;
    // } else {
    //     std::cout << "GPU multiplication is incorrect." << std::endl;
    // }

    return 0;
}
