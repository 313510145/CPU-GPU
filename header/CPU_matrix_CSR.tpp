#include <map>
#include <stdexcept>
#include <omp.h>

template<typename T>
CPU_matrix_CSR<T>& CPU_matrix_CSR<T>::operator=(const CPU_matrix_CSR<T>& other) {
    if (this != &other) {
        if (this->row_ptr != nullptr) {
            delete [] this->row_ptr;
        }
        if (this->col_idx != nullptr) {
            delete [] this->col_idx;
        }
        if (this->values != nullptr) {
            delete [] this->values;
        }
        this->rows = other.rows;
        this->cols = other.cols;
        this->nnz = other.nnz;
        this->row_ptr = new unsigned long long[this->rows + 1]();
        for (unsigned long long i = 0; i <= this->rows; ++i) {
            this->row_ptr[i] = other.row_ptr[i];
        }
        this->col_idx = new unsigned long long[this->nnz]();
        for (unsigned long long i = 0; i < this->nnz; ++i) {
            this->col_idx[i] = other.col_idx[i];
        }
        this->values = new T[this->nnz]();
        for (unsigned long long i = 0; i < this->nnz; ++i) {
            this->values[i] = other.values[i];
        }
    }
    return *this;
}

template<typename T>
CPU_matrix_CSR<T> operator+(const CPU_matrix_CSR<T>& summand, const T& addend) {
    CPU_matrix_CSR<T> result(summand.rows, summand.cols);
    std::vector<unsigned long long> temp_col_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.rows; ++i) {
        unsigned long long summand_end = summand.row_ptr[i + 1];
        unsigned long long non_zero_count = 0;
        for (unsigned long long j = summand.row_ptr[i]; j < summand_end; ++j) {
            T value = summand.values[j] + addend;
            if (value != 0) {
                temp_col_idx.emplace_back(summand.col_idx[j]);
                temp_values.emplace_back(value);
                ++non_zero_count;
            }
        }
        result.row_ptr[i + 1] = temp_col_idx.size();
    }
    result.nnz = temp_values.size();
    result.col_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.col_idx[i] = temp_col_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> operator+(const CPU_matrix_CSR<T>& summand, const CPU_matrix_CSR<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    CPU_matrix_CSR<T> result(summand.rows, summand.cols);
    std::vector<unsigned long long> temp_col_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.rows; ++i) {
        unsigned long long summand_start = summand.row_ptr[i];
        unsigned long long summand_end = summand.row_ptr[i + 1];
        unsigned long long addend_start = addend.row_ptr[i];
        unsigned long long addend_end = addend.row_ptr[i + 1];
        unsigned long long j = summand_start;
        unsigned long long k = addend_start;
        while (j < summand_end && k < addend_end) {
            unsigned long long j_col = summand.col_idx[j];
            unsigned long long k_col = addend.col_idx[k];
            if (j_col == k_col) {
                T value = summand.values[j] + addend.values[k];
                if (value != 0) {
                    temp_col_idx.emplace_back(j_col);
                    temp_values.emplace_back(value);
                }
                ++j;
                ++k;
            } else if (j_col < k_col) {
                temp_col_idx.emplace_back(j_col);
                temp_values.emplace_back(summand.values[j]);
                ++j;
            } else {
                temp_col_idx.emplace_back(k_col);
                temp_values.emplace_back(addend.values[k]);
                ++k;
            }
        }
        while (j < summand_end) {
            temp_col_idx.emplace_back(summand.col_idx[j]);
            temp_values.emplace_back(summand.values[j]);
            ++j;
        }
        while (k < addend_end) {
            temp_col_idx.emplace_back(addend.col_idx[k]);
            temp_values.emplace_back(addend.values[k]);
            ++k;
        }
        result.row_ptr[i + 1] = temp_col_idx.size();
    }
    result.nnz = temp_values.size();
    result.col_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.col_idx[i] = temp_col_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> operator-(const CPU_matrix_CSR<T>& minuend, const T& subtrahend) {
    CPU_matrix_CSR<T> result(minuend.rows, minuend.cols);
    std::vector<unsigned long long> temp_col_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.rows; ++i) {
        unsigned long long minuend_end = minuend.row_ptr[i + 1];
        unsigned long long non_zero_count = 0;
        for (unsigned long long j = minuend.row_ptr[i]; j < minuend_end; ++j) {
            T value = minuend.values[j] - subtrahend;
            if (value != 0) {
                temp_col_idx.emplace_back(minuend.col_idx[j]);
                temp_values.emplace_back(value);
                ++non_zero_count;
            }
        }
        result.row_ptr[i + 1] = temp_col_idx.size();
    }
    result.nnz = temp_values.size();
    result.col_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.col_idx[i] = temp_col_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> operator-(const CPU_matrix_CSR<T>& minuend, const CPU_matrix_CSR<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    CPU_matrix_CSR<T> result(minuend.rows, minuend.cols);
    std::vector<unsigned long long> temp_col_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.rows; ++i) {
        unsigned long long minuend_start = minuend.row_ptr[i];
        unsigned long long minuend_end = minuend.row_ptr[i + 1];
        unsigned long long subtrahend_start = subtrahend.row_ptr[i];
        unsigned long long subtrahend_end = subtrahend.row_ptr[i + 1];
        unsigned long long j = minuend_start;
        unsigned long long k = subtrahend_start;
        while (j < minuend_end && k < subtrahend_end) {
            unsigned long long j_col = minuend.col_idx[j];
            unsigned long long k_col = subtrahend.col_idx[k];
            if (j_col == k_col) {
                T value = minuend.values[j] - subtrahend.values[k];
                if (value != 0) {
                    temp_col_idx.emplace_back(j_col);
                    temp_values.emplace_back(value);
                }
                ++j;
                ++k;
            } else if (j_col < k_col) {
                temp_col_idx.emplace_back(j_col);
                temp_values.emplace_back(minuend.values[j]);
                ++j;
            } else {
                temp_col_idx.emplace_back(k_col);
                temp_values.emplace_back(-subtrahend.values[k]);
                ++k;
            }
        }
        while (j < minuend_end) {
            temp_col_idx.emplace_back(minuend.col_idx[j]);
            temp_values.emplace_back(minuend.values[j]);
            ++j;
        }
        while (k < subtrahend_end) {
            temp_col_idx.emplace_back(subtrahend.col_idx[k]);
            temp_values.emplace_back(-subtrahend.values[k]);
            ++k;
        }
        result.row_ptr[i + 1] = temp_col_idx.size();
    }
    result.nnz = temp_values.size();
    result.col_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.col_idx[i] = temp_col_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> operator*(const CPU_matrix_CSR<T>& multiplicand, const T& multiplier) {
    CPU_matrix_CSR<T> result(multiplicand.rows, multiplicand.cols);
    std::vector<unsigned long long> temp_col_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.rows; ++i) {
        unsigned long long multiplicand_end = multiplicand.row_ptr[i + 1];
        for (unsigned long long j = multiplicand.row_ptr[i]; j < multiplicand_end; ++j) {
            T value = multiplicand.values[j] * multiplier;
            if (value != 0) {
                temp_col_idx.emplace_back(multiplicand.col_idx[j]);
                temp_values.emplace_back(value);
            }
        }
        result.row_ptr[i + 1] = temp_col_idx.size();
    }
    result.nnz = temp_values.size();
    result.col_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.col_idx[i] = temp_col_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> operator*(const CPU_matrix_CSR<T>& multiplicand, const CPU_matrix_CSC<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix_CSR<T> result(multiplicand.rows, multiplier.cols);
    std::vector<unsigned long long> temp_col_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < multiplicand.rows; ++i) {
        std::vector<unsigned long long> row_col_idx;
        std::vector<T> row_values;
        for (unsigned long long j = 0; j < multiplier.cols; ++j) {
            T sum = T();
            unsigned long long multiplicand_start = multiplicand.row_ptr[i];
            unsigned long long multiplicand_end = multiplicand.row_ptr[i + 1];
            unsigned long long multiplier_start = multiplier.col_ptr[j];
            unsigned long long multiplier_end = multiplier.col_ptr[j + 1];
            while (multiplicand_start < multiplicand_end && multiplier_start < multiplier_end) {
                unsigned long long multiplicand_col = multiplicand.col_idx[multiplicand_start];
                unsigned long long multiplier_row = multiplier.row_idx[multiplier_start];
                if (multiplicand_col == multiplier_row) {
                    sum += multiplicand.values[multiplicand_start] * multiplier.values[multiplier_start];
                    ++multiplicand_start;
                    ++multiplier_start;
                } else if (multiplicand_col < multiplier_row) {
                    ++multiplicand_start;
                } else {
                    ++multiplier_start;
                }
            }
            if (sum != 0) {
                row_col_idx.emplace_back(j);
                row_values.emplace_back(sum);
            }
        }
        result.row_ptr[i + 1] = result.row_ptr[i] + row_col_idx.size();
        temp_col_idx.insert(temp_col_idx.end(), row_col_idx.begin(), row_col_idx.end());
        temp_values.insert(temp_values.end(), row_values.begin(), row_values.end());
    }
    result.nnz = temp_values.size();
    result.col_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.col_idx[i] = temp_col_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> operator*(const CPU_matrix_CSR<T>& multiplicand, const CPU_matrix_CSR<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix_CSC<T> multiplier_csc = to_csc(multiplier);
    CPU_matrix_CSR<T> result = multiplicand * multiplier_csc;
    return result;
}

template<typename T>
CPU_matrix_CSC<T> operator*(const CPU_matrix_CSC<T>& multiplicand, const CPU_matrix_CSC<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix_CSR<T> multiplicand_csr = to_csr(multiplicand);
    CPU_matrix_CSR<T> result = to_csc(multiplicand_csr * multiplier);
    return result;
}

template<typename T>
CPU_matrix<T> operator*(const CPU_matrix_CSR<T>& multiplicand, const CPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.data = new T[result.rows * result.cols]();
    for (unsigned long long i = 0; i < multiplicand.rows; ++i) {
        unsigned long long multiplicand_end = multiplicand.row_ptr[i + 1];
        for (unsigned long long j = multiplicand.row_ptr[i]; j < multiplicand_end; ++j) {
            unsigned long long col_idx = multiplicand.col_idx[j];
            T value = multiplicand.values[j];
            for (unsigned long long k = 0; k < multiplier.cols; ++k) {
                result.data[i * multiplier.cols + k] += value * multiplier.data[col_idx * multiplier.cols + k];
            }
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> to_csc(const CPU_matrix_CSR<T>& matrix) {
    CPU_matrix_CSC<T> result_csc(matrix.rows, matrix.cols);
    result_csc.nnz = matrix.nnz;
    result_csc.row_idx = new unsigned long long[matrix.nnz]();
    result_csc.values = new T[matrix.nnz]();
    for (unsigned long long i = 0; i < matrix.nnz; ++i) {
        result_csc.col_ptr[matrix.col_idx[i] + 1]++;
    }
    std::vector<unsigned long long> temp_col_ptr(matrix.cols + 1, 0);
    for (unsigned long long i = 0; i < matrix.cols; ++i) {
        result_csc.col_ptr[i + 1] += result_csc.col_ptr[i];
        temp_col_ptr[i + 1] = result_csc.col_ptr[i + 1];
    }
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        for (unsigned long long j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j) {
            unsigned long long col = matrix.col_idx[j];
            unsigned long long dest = temp_col_ptr[col];
            result_csc.row_idx[dest] = i;
            result_csc.values[dest] = matrix.values[j];
            temp_col_ptr[col]++;
        }
    }
    return result_csc;
}

template<typename T>
CPU_matrix_CSR<T> to_csr(const CPU_matrix_CSC<T>& matrix) {
    CPU_matrix_CSR<T> result_csr(matrix.rows, matrix.cols);
    result_csr.nnz = matrix.nnz;
    result_csr.col_idx = new unsigned long long[matrix.nnz]();
    result_csr.values = new T[matrix.nnz]();
    for (unsigned long long i = 0; i < matrix.nnz; ++i) {
        result_csr.row_ptr[matrix.row_idx[i] + 1]++;
    }
    std::vector<unsigned long long> temp_row_ptr(matrix.rows + 1, 0);
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        result_csr.row_ptr[i + 1] += result_csr.row_ptr[i];
        temp_row_ptr[i + 1] = result_csr.row_ptr[i + 1];
    }
    for (unsigned long long i = 0; i < matrix.cols; ++i) {
        for (unsigned long long j = matrix.col_ptr[i]; j < matrix.col_ptr[i + 1]; ++j) {
            unsigned long long row = matrix.row_idx[j];
            unsigned long long dest = temp_row_ptr[row];
            result_csr.col_idx[dest] = i;
            result_csr.values[dest] = matrix.values[j];
            temp_row_ptr[row]++;
        }
    }
    return result_csr;
}

template<typename T>
CPU_matrix_CSR<T> transpose(const CPU_matrix_CSR<T>& matrix) {
    CPU_matrix_CSR<T> result(matrix.cols, matrix.rows);
    result.nnz = matrix.nnz;
    result.col_idx = new unsigned long long[result.nnz]();
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.row_ptr[matrix.col_idx[i] + 1]++;
    }
    std::vector<unsigned long long> temp_row_ptr(matrix.rows + 1, 0);
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        result.row_ptr[i + 1] += result.row_ptr[i];
        temp_row_ptr[i + 1] = result.row_ptr[i + 1];
    }
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        for (unsigned long long j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j) {
            unsigned long long col = matrix.col_idx[j];
            unsigned long long dest = temp_row_ptr[col];
            result.col_idx[dest] = i;
            result.values[dest] = matrix.values[j];
            temp_row_ptr[col]++;
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_add(const CPU_matrix_CSR<T>& summand, const T& addend) {
    CPU_matrix_CSR<T> result(summand.rows, summand.cols);
    std::vector<std::vector<unsigned long long>> local_col_idx(result.rows);
    std::vector<std::vector<T>> local_values(result.rows);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long summand_end = summand.row_ptr[i + 1];
        for (unsigned long long j = summand.row_ptr[i]; j < summand_end; ++j) {
            T value = summand.values[j] + addend;
            if (value != 0) {
                accumulator[summand.col_idx[j]] += value;
            }
        }
        for (const auto& pair: accumulator) {
            local_col_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.row_ptr[i + 1] = local_col_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.rows; ++i) {
        result.row_ptr[i] += result.row_ptr[i - 1];
    }
    result.nnz = result.row_ptr[result.rows];
    result.col_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_col_idx[i].size(); ++j) {
            result.col_idx[result.row_ptr[i] + j] = local_col_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.row_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_add(const CPU_matrix_CSR<T>& summand, const CPU_matrix_CSR<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    CPU_matrix_CSR<T> result(summand.rows, summand.cols);
    std::vector<std::vector<unsigned long long>> local_col_idx(result.rows);
    std::vector<std::vector<T>> local_values(result.rows);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long summand_start = summand.row_ptr[i];
        unsigned long long summand_end = summand.row_ptr[i + 1];
        unsigned long long addend_start = addend.row_ptr[i];
        unsigned long long addend_end = addend.row_ptr[i + 1];
        unsigned long long j = summand_start;
        unsigned long long k = addend_start;
        while (j < summand_end && k < addend_end) {
            unsigned long long j_col = summand.col_idx[j];
            unsigned long long k_col = addend.col_idx[k];
            if (j_col == k_col) {
                accumulator[j_col] += summand.values[j] + addend.values[k];
                ++j;
                ++k;
            } else if (j_col < k_col) {
                accumulator[j_col] += summand.values[j];
                ++j;
            } else {
                accumulator[k_col] += addend.values[k];
                ++k;
            }
        }
        while (j < summand_end) {
            accumulator[summand.col_idx[j]] += summand.values[j];
            ++j;
        }
        while (k < addend_end) {
            accumulator[addend.col_idx[k]] += addend.values[k];
            ++k;
        }
        for (const auto& pair: accumulator) {
            if (pair.second != 0) {
                local_col_idx[i].emplace_back(pair.first);
                local_values[i].emplace_back(pair.second);
            }
        }
        result.row_ptr[i + 1] = local_col_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.rows; ++i) {
        result.row_ptr[i] += result.row_ptr[i - 1];
    }
    result.nnz = result.row_ptr[result.rows];
    result.col_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_col_idx[i].size(); ++j) {
            result.col_idx[result.row_ptr[i] + j] = local_col_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.row_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_subtract(const CPU_matrix_CSR<T>& minuend, const T& subtrahend) {
    CPU_matrix_CSR<T> result(minuend.rows, minuend.cols);
    std::vector<std::vector<unsigned long long>> local_col_idx(result.rows);
    std::vector<std::vector<T>> local_values(result.rows);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long minuend_end = minuend.row_ptr[i + 1];
        for (unsigned long long j = minuend.row_ptr[i]; j < minuend_end; ++j) {
            T value = minuend.values[j] - subtrahend;
            if (value != 0) {
                accumulator[minuend.col_idx[j]] += value;
            }
        }
        for (const auto& pair: accumulator) {
            local_col_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.row_ptr[i + 1] = local_col_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.rows; ++i) {
        result.row_ptr[i] += result.row_ptr[i - 1];
    }
    result.nnz = result.row_ptr[result.rows];
    result.col_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_col_idx[i].size(); ++j) {
            result.col_idx[result.row_ptr[i] + j] = local_col_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.row_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_subtract(const CPU_matrix_CSR<T>& minuend, const CPU_matrix_CSR<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    CPU_matrix_CSR<T> result(minuend.rows, minuend.cols);
    std::vector<std::vector<unsigned long long>> local_col_idx(result.rows);
    std::vector<std::vector<T>> local_values(result.rows);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long minuend_start = minuend.row_ptr[i];
        unsigned long long minuend_end = minuend.row_ptr[i + 1];
        unsigned long long subtrahend_start = subtrahend.row_ptr[i];
        unsigned long long subtrahend_end = subtrahend.row_ptr[i + 1];
        unsigned long long j = minuend_start;
        unsigned long long k = subtrahend_start;
        while (j < minuend_end && k < subtrahend_end) {
            unsigned long long j_col = minuend.col_idx[j];
            unsigned long long k_col = subtrahend.col_idx[k];
            if (j_col == k_col) {
                accumulator[j_col] += minuend.values[j] - subtrahend.values[k];
                ++j;
                ++k;
            } else if (j_col < k_col) {
                accumulator[j_col] += minuend.values[j];
                ++j;
            } else {
                accumulator[k_col] -= subtrahend.values[k];
                ++k;
            }
        }
        while (j < minuend_end) {
            accumulator[minuend.col_idx[j]] += minuend.values[j];
            ++j;
        }
        while (k < subtrahend_end) {
            accumulator[subtrahend.col_idx[k]] -= subtrahend.values[k];
            ++k;
        }
        for (const auto& pair: accumulator) {
            if (pair.second != 0) {
                local_col_idx[i].emplace_back(pair.first);
                local_values[i].emplace_back(pair.second);
            }
        }
        result.row_ptr[i + 1] = local_col_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.rows; ++i) {
        result.row_ptr[i] += result.row_ptr[i - 1];
    }
    result.nnz = result.row_ptr[result.rows];
    result.col_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_col_idx[i].size(); ++j) {
            result.col_idx[result.row_ptr[i] + j] = local_col_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.row_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_multiply(const CPU_matrix_CSR<T>& multiplicand, const T& multiplier) {
    CPU_matrix_CSR<T> result(multiplicand.rows, multiplicand.cols);
    std::vector<std::vector<unsigned long long>> local_col_idx(result.rows);
    std::vector<std::vector<T>> local_values(result.rows);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long multiplicand_end = multiplicand.row_ptr[i + 1];
        for (unsigned long long j = multiplicand.row_ptr[i]; j < multiplicand_end; ++j) {
            T value = multiplicand.values[j] * multiplier;
            if (value != 0) {
                accumulator[multiplicand.col_idx[j]] += value;
            }
        }
        for (const auto& pair: accumulator) {
            local_col_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.row_ptr[i + 1] = local_col_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.rows; ++i) {
        result.row_ptr[i] += result.row_ptr[i - 1];
    }
    result.nnz = result.row_ptr[result.rows];
    result.col_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_col_idx[i].size(); ++j) {
            result.col_idx[result.row_ptr[i] + j] = local_col_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.row_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_multiply(const CPU_matrix_CSR<T>& multiplicand, const CPU_matrix_CSC<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix_CSR<T> result(multiplicand.rows, multiplier.cols);
    std::vector<std::vector<unsigned long long>> local_col_idx(result.rows);
    std::vector<std::vector<T>> local_values(result.rows);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < multiplicand.rows; ++i) {
        std::map<unsigned long long, T> accumulator;
        for (unsigned long long j = 0; j < multiplier.cols; ++j) {
            T sum = T();
            unsigned long long multiplicand_start = multiplicand.row_ptr[i];
            unsigned long long multiplicand_end = multiplicand.row_ptr[i + 1];
            unsigned long long multiplier_start = multiplier.col_ptr[j];
            unsigned long long multiplier_end = multiplier.col_ptr[j + 1];
            while (multiplicand_start < multiplicand_end && multiplier_start < multiplier_end) {
                unsigned long long multiplicand_col = multiplicand.col_idx[multiplicand_start];
                unsigned long long multiplier_row = multiplier.row_idx[multiplier_start];
                if (multiplicand_col == multiplier_row) {
                    sum += multiplicand.values[multiplicand_start] * multiplier.values[multiplier_start];
                    ++multiplicand_start;
                    ++multiplier_start;
                } else if (multiplicand_col < multiplier_row) {
                    ++multiplicand_start;
                } else {
                    ++multiplier_start;
                }
            }
            if (sum != 0) {
                accumulator[j] += sum;
            }
        }
        for (const auto& pair: accumulator) {
            local_col_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.row_ptr[i + 1] = local_col_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.rows; ++i) {
        result.row_ptr[i] += result.row_ptr[i - 1];
    }
    result.nnz = result.row_ptr[result.rows];
    result.col_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_col_idx[i].size(); ++j) {
            result.col_idx[result.row_ptr[i] + j] = local_col_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.rows; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.row_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T> openmp_multiply(const CPU_matrix_CSR<T>& multiplicand, const CPU_matrix_CSR<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix_CSC<T> multiplier_csc = to_csc(multiplier);
    CPU_matrix_CSR<T> result = openmp_multiply(multiplicand, multiplier_csc);
    return result;
}

template<typename T>
CPU_matrix_CSC<T> openmp_multiply(const CPU_matrix_CSC<T>& multiplicand, const CPU_matrix_CSC<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix_CSR<T> multiplicand_csr = to_csr(multiplicand);
    CPU_matrix_CSR<T> result = to_csc(openmp_multiply(multiplicand_csr, multiplier));
    return result;
}

template<typename T>
CPU_matrix<T> openmp_multiply(const CPU_matrix_CSR<T>& multiplicand, const CPU_matrix<T>& multiplier) {
    if (multiplicand.cols != multiplier.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    CPU_matrix<T> result;
    result.rows = multiplicand.rows;
    result.cols = multiplier.cols;
    result.data = new T[result.rows * result.cols]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < multiplicand.rows; ++i) {
        unsigned long long multiplicand_end = multiplicand.row_ptr[i + 1];
        #pragma omp simd
        for (unsigned long long j = multiplicand.row_ptr[i]; j < multiplicand_end; ++j) {
            unsigned long long col_idx = multiplicand.col_idx[j];
            T value = multiplicand.values[j];
            for (unsigned long long k = 0; k < multiplier.cols; ++k) {
                result.data[i * multiplier.cols + k] += value * multiplier.data[col_idx * multiplier.cols + k];
            }
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSR<T>::CPU_matrix_CSR(const CPU_matrix_CSR<T>& other): matrix_base<T>(other.rows, other.cols) {
    this->nnz = other.nnz;
    this->row_ptr = new unsigned long long[this->rows + 1]();
    for (unsigned long long i = 0; i <= this->rows; ++i) {
        this->row_ptr[i] = other.row_ptr[i];
    }
    this->col_idx = new unsigned long long[this->nnz]();
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        this->col_idx[i] = other.col_idx[i];
    }
    this->values = new T[this->nnz]();
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        this->values[i] = other.values[i];
    }
}

template<typename T>
CPU_matrix_CSR<T>::CPU_matrix_CSR(const unsigned long long* row_ptr, const unsigned long long* col_idx, const T* values, unsigned long long nnz, unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols), nnz(nnz) {
    if (row_ptr == nullptr || col_idx == nullptr || values == nullptr) {
        throw std::invalid_argument("Null point passed to CPU_matrix_CSR constructor.");
    }
    this->row_ptr = new unsigned long long[rows + 1]();
    for (unsigned long long i = 0; i <= rows; ++i) {
        this->row_ptr[i] = row_ptr[i];
    }
    this->col_idx = new unsigned long long[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->col_idx[i] = col_idx[i];
    }
    this->values = new T[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->values[i] = values[i];
    }
}

template<typename T>
CPU_matrix_CSR<T>::CPU_matrix_CSR(const std::vector<unsigned long long>& row_ptr, const std::vector<unsigned long long>& col_idx, const std::vector<T>& values, unsigned long long nnz, unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols), nnz(nnz) {
    if (row_ptr.empty() || col_idx.empty() || values.empty()) {
        throw std::invalid_argument("Empty vector passed to CPU_matrix_CSR constructor.");
    }
    this->row_ptr = new unsigned long long[rows + 1]();
    for (unsigned long long i = 0; i <= rows; ++i) {
        this->row_ptr[i] = row_ptr[i];
    }
    this->col_idx = new unsigned long long[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->col_idx[i] = col_idx[i];
    }
    this->values = new T[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->values[i] = values[i];
    }
}
