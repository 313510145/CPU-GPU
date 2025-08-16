#include "CPU_matrix_CSC.h"

template<typename T>
void CPU_matrix_CSC<T>::print_all(std::ostream& os) const {
    os << "rows: " << this->rows << ", cols: " << this->cols << ", nnz: " << this->nnz << std::endl;
    os << "col_ptr:";
    for (unsigned long long i = 0; i <= this->cols; ++i) {
        os << " " << this->col_ptr[i];
    }
    os << std::endl;
    os << "row_idx:";
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        os << " " << this->row_idx[i];
    }
    os << std::endl;
    os << "values:";
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        os << " " << this->values[i];
    }
    os << std::endl;
}

template<typename T>
bool operator==(const CPU_matrix_CSC<T>& lhs, const CPU_matrix_CSC<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return false;
    }
    for (unsigned long long i = 0; i <= lhs.cols; ++i) {
        if (lhs.col_ptr[i] != rhs.col_ptr[i]) {
            return false;
        }
    }
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.row_idx[i] != rhs.row_idx[i] || lhs.values[i] != rhs.values[i]) {
            return false;
        }
    }
    return true;
}

template bool operator==(const CPU_matrix_CSC<double>& lhs, const CPU_matrix_CSC<double>& rhs);

template<typename T>
bool operator!=(const CPU_matrix_CSC<T>& lhs, const CPU_matrix_CSC<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return true;
    }
    for (unsigned long long i = 0; i <= lhs.cols; ++i) {
        if (lhs.col_ptr[i] != rhs.col_ptr[i]) {
            return true;
        }
    }
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.row_idx[i] != rhs.row_idx[i] || lhs.values[i] != rhs.values[i]) {
            return true;
        }
    }
    return false;
}

template bool operator!=(const CPU_matrix_CSC<double>& lhs, const CPU_matrix_CSC<double>& rhs);

template<typename T>
bool openmp_equal(const CPU_matrix_CSC<T>& lhs, const CPU_matrix_CSC<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return false;
    }
    bool equal = true;
    #pragma omp parallel for
    for (unsigned long long i = 0; i <= lhs.cols; ++i) {
        if (lhs.col_ptr[i] != rhs.col_ptr[i]) {
            equal = false;
        }
    }
    #pragma omp parallel for
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.row_idx[i] != rhs.row_idx[i] || lhs.values[i] != rhs.values[i]) {
            equal = false;
        }
    }
    return equal;
}

template bool openmp_equal(const CPU_matrix_CSC<double>& lhs, const CPU_matrix_CSC<double>& rhs);

template<typename T>
bool openmp_not_equal(const CPU_matrix_CSC<T>& lhs, const CPU_matrix_CSC<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.nnz != rhs.nnz) {
        return true;
    }
    bool not_equal = false;
    #pragma omp parallel for
    for (unsigned long long i = 0; i <= lhs.cols; ++i) {
        if (lhs.col_ptr[i] != rhs.col_ptr[i]) {
            not_equal = true;
        }
    }
    #pragma omp parallel for
    for (unsigned long long i = 0; i < lhs.nnz; ++i) {
        if (lhs.row_idx[i] != rhs.row_idx[i] || lhs.values[i] != rhs.values[i]) {
            not_equal = true;
        }
    }
    return not_equal;
}

template bool openmp_not_equal(const CPU_matrix_CSC<double>& lhs, const CPU_matrix_CSC<double>& rhs);

template<typename T>
CPU_matrix_CSC<T>::CPU_matrix_CSC(): matrix_base<T>() {
    this->col_ptr = nullptr;
    this->row_idx = nullptr;
    this->values = nullptr;
    this->nnz = 0;
}

template<typename T>
CPU_matrix_CSC<T>::CPU_matrix_CSC(unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols) {
    this->col_ptr = new unsigned long long[cols + 1]();
    this->row_idx = nullptr;
    this->values = nullptr;
    this->nnz = 0;
}

template<typename T>
CPU_matrix_CSC<T>::CPU_matrix_CSC(CPU_matrix_CSC<T>&& other): matrix_base<T>(other.rows, other.cols) {
    this->col_ptr = other.col_ptr;
    this->row_idx = other.row_idx;
    this->values = other.values;
    this->nnz = other.nnz;
    other.col_ptr = nullptr;
    other.row_idx = nullptr;
    other.values = nullptr;
    other.nnz = 0;
}

template<typename T>
CPU_matrix_CSC<T>::~CPU_matrix_CSC() {
    if (this->col_ptr != nullptr) {
        delete [] this->col_ptr;
        this->col_ptr = nullptr;
    }
    if (this->row_idx != nullptr) {
        delete [] this->row_idx;
        this->row_idx = nullptr;
    }
    if (this->values != nullptr) {
        delete [] this->values;
        this->values = nullptr;
    }
    this->nnz = 0;
}

template class CPU_matrix_CSC<double>;
