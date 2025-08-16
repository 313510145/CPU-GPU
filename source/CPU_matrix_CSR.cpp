#include "CPU_matrix_CSR.h"

template<typename T>
void CPU_matrix_CSR<T>::print_all(std::ostream& os) const {
    os << "rows: " << this->rows << ", cols: " << this->cols << ", nnz: " << this->nnz << std::endl;
    os << "row_ptr:";
    for (unsigned long long i = 0; i <= this->rows; ++i) {
        os << " " << this->row_ptr[i];
    }
    os << std::endl;
    os << "col_idx:";
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        os << " " << this->col_idx[i];
    }
    os << std::endl;
    os << "values:";
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        os << " " << this->values[i];
    }
    os << std::endl;
}

template<typename T>
bool operator==(const CPU_matrix_CSR<T>& lhs, const CPU_matrix_CSR<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return false;
    }
    for (unsigned long long i = 0; i <= lhs.rows; ++i) {
        if (lhs.row_ptr[i] != rhs.row_ptr[i]) {
            return false;
        }
    }
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.col_idx[i] != rhs.col_idx[i] || lhs.values[i] != rhs.values[i]) {
            return false;
        }
    }
    return true;
}

template bool operator==(const CPU_matrix_CSR<double>& lhs, const CPU_matrix_CSR<double>& rhs);

template<typename T>
bool operator!=(const CPU_matrix_CSR<T>& lhs, const CPU_matrix_CSR<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return true;
    }
    for (unsigned long long i = 0; i <= lhs.rows; ++i) {
        if (lhs.row_ptr[i] != rhs.row_ptr[i]) {
            return true;
        }
    }
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.col_idx[i] != rhs.col_idx[i] || lhs.values[i] != rhs.values[i]) {
            return true;
        }
    }
    return false;
}

template bool operator!=(const CPU_matrix_CSR<double>& lhs, const CPU_matrix_CSR<double>& rhs);

template<typename T>
bool openmp_equal(const CPU_matrix_CSR<T>& lhs, const CPU_matrix_CSR<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return false;
    }
    bool equal = true;
    #pragma omp parallel for
    for (unsigned long long i = 0; i <= lhs.rows; ++i) {
        if (lhs.row_ptr[i] != rhs.row_ptr[i]) {
            equal = false;
        }
    }
    #pragma omp parallel for
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.col_idx[i] != rhs.col_idx[i] || lhs.values[i] != rhs.values[i]) {
            equal = false;
        }
    }
    return equal;
}

template bool openmp_equal(const CPU_matrix_CSR<double>& lhs, const CPU_matrix_CSR<double>& rhs);

template<typename T>
bool openmp_not_equal(const CPU_matrix_CSR<T>& lhs, const CPU_matrix_CSR<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return true;
    }
    bool not_equal = false;
    #pragma omp parallel for
    for (unsigned long long i = 0; i <= lhs.rows; ++i) {
        if (lhs.row_ptr[i] != rhs.row_ptr[i]) {
            not_equal = true;
        }
    }
    #pragma omp parallel for
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.col_idx[i] != rhs.col_idx[i] || lhs.values[i] != rhs.values[i]) {
            not_equal = true;
        }
    }
    return not_equal;
}

template bool openmp_not_equal(const CPU_matrix_CSR<double>& lhs, const CPU_matrix_CSR<double>& rhs);

template<typename T>
CPU_matrix_CSR<T>::CPU_matrix_CSR(): matrix_base<T>() {
    this->row_ptr = nullptr;
    this->col_idx = nullptr;
    this->values = nullptr;
    this->nnz = 0;
}

template<typename T>
CPU_matrix_CSR<T>::CPU_matrix_CSR(unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols) {
    this->row_ptr = new unsigned long long[rows + 1]();
    this->col_idx = nullptr;
    this->values = nullptr;
    this->nnz = 0;
}

template<typename T>
CPU_matrix_CSR<T>::CPU_matrix_CSR(CPU_matrix_CSR<T>&& other): matrix_base<T>(other.rows, other.cols) {
    this->row_ptr = other.row_ptr;
    this->col_idx = other.col_idx;
    this->values = other.values;
    this->nnz = other.nnz;
    other.row_ptr = nullptr;
    other.col_idx = nullptr;
    other.values = nullptr;
    other.nnz = 0;
}

template<typename T>
CPU_matrix_CSR<T>::~CPU_matrix_CSR() {
    if (this->row_ptr != nullptr) {
        delete [] this->row_ptr;
        this->row_ptr = nullptr;
    }
    if (this->col_idx != nullptr) {
        delete [] this->col_idx;
        this->col_idx = nullptr;
    }
    if (this->values != nullptr) {
        delete [] this->values;
        this->values = nullptr;
    }
    this->nnz = 0;
}

template class CPU_matrix_CSR<double>;
