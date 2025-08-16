#include <stdexcept>
#include <omp.h>

template<typename T>
void CPU_matrix<T>::allocate(unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = new T[this->rows * this->cols]();
}

template<typename T>
CPU_matrix<T>& CPU_matrix<T>::operator=(const CPU_matrix<T>& other) {
    if (this != &other) {
        if (this->data != nullptr) {
            delete [] this->data;
        }
        this->rows = other.rows;
        this->cols = other.cols;
        unsigned long long size = this->rows * this->cols;
        this->data = new T[size]();
        for (unsigned long long i = 0; i < size; ++i) {
            this->data[i] = other.data[i];
        }
    }
    return *this;
}

template<typename T>
CPU_matrix<T> operator+(const CPU_matrix<T>& summand, const T& addend) {
    CPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = summand.data[i] + addend;
    }
    return result;
}

template<typename T>
CPU_matrix<T> operator+(const CPU_matrix<T>& summand, const CPU_matrix<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    CPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = summand.data[i] + addend.data[i];
    }
    return result;
}

template<typename T>
CPU_matrix<T> operator-(const CPU_matrix<T>& minuend, const T& subtrahend) {
    CPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = minuend.data[i] - subtrahend;
    }
    return result;
}

template<typename T>
CPU_matrix<T> operator-(const CPU_matrix<T>& minuend, const CPU_matrix<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    CPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = minuend.data[i] - subtrahend.data[i];
    }
    return result;
}

template<typename T>
CPU_matrix<T> operator*(const CPU_matrix<T>& multiplicand, const T& multiplier) {
    CPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplicand.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = multiplicand.data[i] * multiplier;
    }
    return result;
}

template<typename T>
CPU_matrix<T> operator*(const CPU_matrix<T>& multiplicand, const CPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.data = new T[result.rows * result.cols]();
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < result.cols; ++j) {
            for (unsigned long long k = 0; k < multiplicand.cols; ++k) {
                result.data[i * result.cols + j] += multiplicand.data[i * multiplicand.cols + k] * multiplier.data[k * multiplier.cols + j];
            }
        }
    }
    return result;
}

template<typename T>
CPU_matrix<T> transpose(const CPU_matrix<T>& matrix) {
    CPU_matrix<T> result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.data = new T[result.rows * result.cols]();
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        for (unsigned long long j = 0; j < matrix.cols; ++j) {
            result.data[j * result.rows + i] = matrix.data[i * matrix.cols + j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_add(const CPU_matrix<T>& summand, const T& addend) {
    CPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = summand.data[i] + addend;
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_add(const CPU_matrix<T>& summand, const CPU_matrix<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    CPU_matrix<T> result;
    result.rows = summand.rows;
    result.cols = summand.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = summand.data[i] + addend.data[i];
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_subtract(const CPU_matrix<T>& minuend, const T& subtrahend) {
    CPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = minuend.data[i] - subtrahend;
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_subtract(const CPU_matrix<T>& minuend, const CPU_matrix<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    CPU_matrix<T> result;
    result.rows = minuend.rows;
    result.cols = minuend.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = minuend.data[i] - subtrahend.data[i];
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_multiply(const CPU_matrix<T>& multiplicand, const T& multiplier) {
    CPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplicand.cols;
    unsigned long long size = result.rows * result.cols;
    result.data = new T[size]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < size; ++i) {
        result.data[i] = multiplicand.data[i] * multiplier;
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_multiply(const CPU_matrix<T>& multiplicand, const CPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.data = new T[result.rows * result.cols]();
    #pragma omp parallel for collapse(2)
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < result.cols; ++j) {
            for (unsigned long long k = 0; k < multiplicand.cols; ++k) {
                result.data[i * result.cols + j] += multiplicand.data[i * multiplicand.cols + k] * multiplier.data[k * multiplier.cols + j];
            }
        }
    }
    return result;
}

template<typename T>
CPU_matrix<T> openmp_transpose(const CPU_matrix<T>& matrix) {
    CPU_matrix<T> result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.data = new T[result.rows * result.cols]();
    #pragma omp parallel for collapse(2)
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        for (unsigned long long j = 0; j < matrix.cols; ++j) {
            result.data[j * result.rows + i] = matrix.data[i * matrix.cols + j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix<T>::CPU_matrix(const CPU_matrix<T>& other) {
    this->rows = other.rows;
    this->cols = other.cols;
    unsigned long long size = this->rows * this->cols;
    this->data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        this->data[i] = other.data[i];
    }
}

template<typename T>
CPU_matrix<T>::CPU_matrix(const T* data, unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols) {
    unsigned long long size = this->rows * this->cols;
    this->data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        this->data[i] = data[i];
    }
}

template<typename T>
CPU_matrix<T>::CPU_matrix(const T** data, unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols) {
    this->data = new T[this->rows * this->cols]();
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            this->data[i * this->cols + j] = data[i][j];
        }
    }
}

template<typename T>
CPU_matrix<T>::CPU_matrix(const std::vector<T>& vec, unsigned long long rows, unsigned long long cols) {
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("Size of vector does not match specified dimensions.");
    }
    this->rows = rows;
    this->cols = cols;
    unsigned long long size = this->rows * this->cols;
    this->data = new T[size]();
    for (unsigned long long i = 0; i < size; ++i) {
        this->data[i] = vec[i];
    }
}

template<typename T>
CPU_matrix<T>::CPU_matrix(const std::vector<std::vector<T>>& vec, unsigned long long rows, unsigned long long cols) {
    if (vec.size() != rows) {
        throw std::invalid_argument("Number of rows in vector does not match specified rows.");
    }
    for (const auto& row : vec) {
        if (row.size() != cols) {
            throw std::invalid_argument("Number of columns in vector does not match specified columns.");
        }
    }
    this->rows = rows;
    this->cols = cols;
    this->data = new T[this->rows * this->cols]();
    for (unsigned long long i = 0; i < this->rows; ++i) {
        for (unsigned long long j = 0; j < this->cols; ++j) {
            this->data[i * this->cols + j] = vec[i][j];
        }
    }
}
