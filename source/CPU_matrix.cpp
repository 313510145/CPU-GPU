#include "CPU_matrix.h"

template<typename T>
void CPU_matrix<T>::print_all(std::ostream& os) const {
    os << "rows: " << this->rows << ", cols: " << this->cols << std::endl;
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            os << " " << this->data[i * this->cols + j];
        }
        os << std::endl;
    }
}

template<typename T>
bool operator==(const CPU_matrix<T>& lhs, const CPU_matrix<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return false;
    }
    unsigned long long size = lhs.rows * lhs.cols;
    for (unsigned long long i = 0; i < size; ++i) {
        if (lhs.data[i] != rhs.data[i]) {
            return false;
        }
    }
    return true;
}

template bool operator==(const CPU_matrix<double>& lhs, const CPU_matrix<double>& rhs);

template<typename T>
bool operator!=(const CPU_matrix<T>& lhs, const CPU_matrix<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return true;
    }
    unsigned long long size = lhs.rows * lhs.cols;
    for (unsigned long long i = 0; i < size; ++i) {
        if (lhs.data[i] != rhs.data[i]) {
            return true;
        }
    }
    return false;
}

template bool operator!=(const CPU_matrix<double>& lhs, const CPU_matrix<double>& rhs);

template<typename T>
bool openmp_equal(const CPU_matrix<T>& lhs, const CPU_matrix<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return false;
    }
    unsigned long long size = lhs.rows * lhs.cols;
    bool equal = true;
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        if (lhs.data[i] != rhs.data[i]) {
            equal = false;
        }
    }
    return equal;
}

template bool openmp_equal(const CPU_matrix<double>& lhs, const CPU_matrix<double>& rhs);

template<typename T>
bool openmp_not_equal(const CPU_matrix<T>& lhs, const CPU_matrix<T>& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return true;
    }
    unsigned long long size = lhs.rows * lhs.cols;
    bool not_equal = false;
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        if (lhs.data[i] != rhs.data[i]) {
            not_equal = true;
        }
    }
    return not_equal;
}

template bool openmp_not_equal(const CPU_matrix<double>& lhs, const CPU_matrix<double>& rhs);

template<typename T>
CPU_matrix<T>::CPU_matrix(): matrix_base<T>() {
    this->data = nullptr;
}

template<typename T>
CPU_matrix<T>::CPU_matrix(CPU_matrix<T>&& other) {
    this->data = other.data;
    this->rows = other.rows;
    this->cols = other.cols;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

template<typename T>
CPU_matrix<T>::~CPU_matrix() {
    if (this->data != nullptr) {
        delete [] this->data;
        this->data = nullptr;
    }
}

template class CPU_matrix<double>;
