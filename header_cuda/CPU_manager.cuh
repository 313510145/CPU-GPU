#ifndef CPU_MANAGER_CUH
#define CPU_MANAGER_CUH

#include "manager_base.h"
#include "CPU_matrix.h"
#include "CPU_matrix_CSR.h"
#include "CPU_matrix_CSC.h"
#include "GPU_matrix.cuh"
#include "GPU_matrix_CSR.cuh"

#include <unordered_map>

class CPU_manager: public manager_base {
    public:
        unsigned long long get_max_threads();
        void set_time_stamp(const std::string& key) override;
        double get_time_duration(const std::string& key_start, const std::string& key_end) override;
        void test_CPU_matrix_add();
        void test_CPU_matrix_subtract();
        void test_CPU_matrix_multiply();
        void test_CPU_matrix_transpose();
        void test_CPU_matrix_CSR_add();
        void test_CPU_matrix_CSR_subtract();
        void test_CPU_matrix_CSR_multiply();
        void test_CPU_matrix_CSR_transpose();
        void test_CPU_matrix_CSC_add();
        void test_CPU_matrix_CSC_subtract();
        void test_CPU_matrix_CSC_transpose();
        void test_GPU_matrix_add();
        void test_GPU_matrix_subtract();
        void test_GPU_matrix_multiply();
        void test_GPU_matrix_transpose();
        // void test_GPU_matrix_CSR_add();
        // void test_GPU_matrix_CSR_subtract();
        // void test_GPU_matrix_CSR_multiply();
        // void test_GPU_matrix_CSR_transpose();
        // void test_GPU_matrix_CSC_add();
        // void test_GPU_matrix_CSC_subtract();
        // void test_GPU_matrix_CSC_transpose();
        void test_GPU_sparse_matrix_vector_multiply();
        void run_sparse_matrix_vector_multiply();
        void run_openmp_sparse_matrix_vector_multiply();
        void run_GPU_sparse_matrix_vector_multiply();
        CPU_manager();
        ~CPU_manager() override;
    private:
        std::unordered_map<std::string, double> time_stamp;
};

#endif  // CPU_MANAGER_H
