#ifndef GPU_MATRIX_CSR_CUH
#define GPU_MATRIX_CSR_CUH

#include "matrix_base.h"

template<typename T>
class GPU_matrix_CSR: public matrix_base<T> {
    public:
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
