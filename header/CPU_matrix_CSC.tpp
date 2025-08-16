#include <stdexcept>
#include <omp.h>
#include <map>

template<typename T>
CPU_matrix_CSC<T>& CPU_matrix_CSC<T>::operator=(const CPU_matrix_CSC<T>& other) {
    if (this != &other) {
        if (this->col_ptr != nullptr) {
            delete [] this->col_ptr;
        }
        if (this->row_idx != nullptr) {
            delete [] this->row_idx;
        }
        if (this->values != nullptr) {
            delete [] this->values;
        }
        this->rows = other.rows;
        this->cols = other.cols;
        this->nnz = other.nnz;
        this->col_ptr = new unsigned long long[this->cols + 1]();
        for (unsigned long long i = 0; i <= this->cols; ++i) {
            this->col_ptr[i] = other.col_ptr[i];
        }
        this->row_idx = new unsigned long long[this->nnz]();
        for (unsigned long long i = 0; i < this->nnz; ++i) {
            this->row_idx[i] = other.row_idx[i];
        }
        this->values = new T[this->nnz]();
        for (unsigned long long i = 0; i < this->nnz; ++i) {
            this->values[i] = other.values[i];
        }
    }
    return *this;
}

template<typename T>
CPU_matrix_CSC<T> operator+(const CPU_matrix_CSC<T>& summand, const T& addend) {
    CPU_matrix_CSC<T> result(summand.rows, summand.cols);
    std::vector<unsigned long long> temp_row_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.cols; ++i) {
        unsigned long long summand_end = summand.col_ptr[i + 1];
        unsigned long long non_zero_count = 0;
        for (unsigned long long j = summand.col_ptr[i]; j < summand_end; ++j) {
            T value = summand.values[j] + addend;
            if (value != 0) {
                temp_row_idx.emplace_back(summand.row_idx[j]);
                temp_values.emplace_back(value);
                ++non_zero_count;
            }
        }
        result.col_ptr[i + 1] = result.col_ptr[i] + non_zero_count;
    }
    result.nnz = temp_values.size();
    result.row_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.row_idx[i] = temp_row_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> operator+(const CPU_matrix_CSC<T>& summand, const CPU_matrix_CSC<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    CPU_matrix_CSC<T> result(summand.rows, summand.cols);
    std::vector<unsigned long long> temp_row_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.cols; ++i) {
        unsigned long long summand_end = summand.col_ptr[i + 1];
        unsigned long long addend_end = addend.col_ptr[i + 1];
        unsigned long long j = summand.col_ptr[i];
        unsigned long long k = addend.col_ptr[i];
        while (j < summand_end && k < addend_end) {
            unsigned long long j_row = summand.row_idx[j];
            unsigned long long k_row = addend.row_idx[k];
            if (j_row == k_row) {
                temp_row_idx.emplace_back(j_row);
                temp_values.emplace_back(summand.values[j] + addend.values[k]);
                ++j;
                ++k;
            } else if (j_row < k_row) {
                temp_row_idx.emplace_back(j_row);
                temp_values.emplace_back(summand.values[j]);
                ++j;
            } else {
                temp_row_idx.emplace_back(k_row);
                temp_values.emplace_back(addend.values[k]);
                ++k;
            }
        }
        while (j < summand_end) {
            temp_row_idx.emplace_back(summand.row_idx[j]);
            temp_values.emplace_back(summand.values[j]);
            ++j;
        }
        while (k < addend_end) {
            temp_row_idx.emplace_back(addend.row_idx[k]);
            temp_values.emplace_back(addend.values[k]);
            ++k;
        }
        result.col_ptr[i + 1] = temp_row_idx.size();
    }
    result.nnz = temp_values.size();
    result.row_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.row_idx[i] = temp_row_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> operator-(const CPU_matrix_CSC<T>& minuend, const T& subtrahend) {
    CPU_matrix_CSC<T> result(minuend.rows, minuend.cols);
    std::vector<unsigned long long> temp_row_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < minuend.cols; ++i) {
        unsigned long long minuend_end = minuend.col_ptr[i + 1];
        unsigned long long non_zero_count = 0;
        for (unsigned long long j = minuend.col_ptr[i]; j < minuend_end; ++j) {
            T value = minuend.values[j] - subtrahend;
            if (value != 0) {
                temp_row_idx.emplace_back(minuend.row_idx[j]);
                temp_values.emplace_back(value);
                ++non_zero_count;
            }
        }
        result.col_ptr[i + 1] = result.col_ptr[i] + non_zero_count;
    }
    result.nnz = temp_values.size();
    result.row_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.row_idx[i] = temp_row_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> operator-(const CPU_matrix_CSC<T>& minuend, const CPU_matrix_CSC<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    CPU_matrix_CSC<T> result(minuend.rows, minuend.cols);
    std::vector<unsigned long long> temp_row_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < result.cols; ++i) {
        unsigned long long minuend_end = minuend.col_ptr[i + 1];
        unsigned long long subtrahend_end = subtrahend.col_ptr[i + 1];
        unsigned long long j = minuend.col_ptr[i];
        unsigned long long k = subtrahend.col_ptr[i];
        while (j < minuend_end && k < subtrahend_end) {
            unsigned long long j_row = minuend.row_idx[j];
            unsigned long long k_row = subtrahend.row_idx[k];
            if (j_row == k_row) {
                T value = minuend.values[j] - subtrahend.values[k];
                if (value != 0) {
                    temp_row_idx.emplace_back(j_row);
                    temp_values.emplace_back(value);
                }
                ++j;
                ++k;
            } else if (j_row < k_row) {
                temp_row_idx.emplace_back(j_row);
                temp_values.emplace_back(minuend.values[j]);
                ++j;
            } else {
                temp_row_idx.emplace_back(k_row);
                temp_values.emplace_back(-subtrahend.values[k]);
                ++k;
            }
        }
        while (j < minuend_end) {
            temp_row_idx.emplace_back(minuend.row_idx[j]);
            temp_values.emplace_back(minuend.values[j]);
            ++j;
        }
        while (k < subtrahend_end) {
            temp_row_idx.emplace_back(subtrahend.row_idx[k]);
            temp_values.emplace_back(-subtrahend.values[k]);
            ++k;
        }
        result.col_ptr[i + 1] = temp_row_idx.size();
    }
    result.nnz = temp_values.size();
    result.row_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.row_idx[i] = temp_row_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> operator*(const CPU_matrix_CSC<T>& multiplicand, const T& multiplier) {
    CPU_matrix_CSC<T> result(multiplicand.rows, multiplicand.cols);
    std::vector<unsigned long long> temp_row_idx;
    std::vector<T> temp_values;
    for (unsigned long long i = 0; i < multiplicand.cols; ++i) {
        unsigned long long multiplicand_end = multiplicand.col_ptr[i + 1];
        for (unsigned long long j = multiplicand.col_ptr[i]; j < multiplicand_end; ++j) {
            T value = multiplicand.values[j] * multiplier;
            if (value != 0) {
                temp_row_idx.emplace_back(multiplicand.row_idx[j]);
                temp_values.emplace_back(value);
            }
        }
        result.col_ptr[i + 1] = result.col_ptr[i] + temp_row_idx.size();
    }
    result.nnz = temp_values.size();
    result.row_idx = new unsigned long long[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.row_idx[i] = temp_row_idx[i];
    }
    result.values = new T[result.nnz]();
    for (unsigned long long i = 0; i < result.nnz; ++i) {
        result.values[i] = temp_values[i];
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> transpose(const CPU_matrix_CSC<T>& matrix) {
    CPU_matrix_CSC<T> result(matrix.cols, matrix.rows);
    result.nnz = matrix.nnz;
    result.row_idx = new unsigned long long[matrix.nnz]();
    result.values = new T[matrix.nnz]();
    for (unsigned long long i = 0; i < matrix.nnz; ++i) {
        result.col_ptr[matrix.row_idx[i] + 1]++;
    }
    std::vector<unsigned long long> temp_col_ptr(matrix.cols + 1, 0);
    for (unsigned long long i = 0; i < matrix.cols; ++i) {
        result.col_ptr[i + 1] += result.col_ptr[i];
        temp_col_ptr[i + 1] = result.col_ptr[i + 1];
    }
    for (unsigned long long i = 0; i < matrix.rows; ++i) {
        for (unsigned long long j = matrix.col_ptr[i]; j < matrix.col_ptr[i + 1]; ++j) {
            unsigned long long col = matrix.row_idx[j];
            unsigned long long dest = temp_col_ptr[col];
            result.row_idx[dest] = i;
            result.values[dest] = matrix.values[j];
            temp_col_ptr[col]++;
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> openmp_add(const CPU_matrix_CSC<T>& summand, const T& addend) {
    CPU_matrix_CSC<T> result(summand.rows, summand.cols);
    std::vector<std::vector<unsigned long long>> local_row_idx(result.cols);
    std::vector<std::vector<T>> local_values(result.cols);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long summand_end = summand.col_ptr[i + 1];
        for (unsigned long long j = summand.col_ptr[i]; j < summand_end; ++j) {
            T value = summand.values[j] + addend;
            if (value != 0) {
                accumulator[summand.row_idx[j]] += value;
            }
        }
        for (const auto& pair: accumulator) {
            local_row_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.col_ptr[i + 1] = local_row_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.cols; ++i) {
        result.col_ptr[i] += result.col_ptr[i - 1];
    }
    result.nnz = result.col_ptr[result.cols];
    result.row_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_row_idx[i].size(); ++j) {
            result.row_idx[result.col_ptr[i] + j] = local_row_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.col_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> openmp_add(const CPU_matrix_CSC<T>& summand, const CPU_matrix_CSC<T>& addend) {
    if (summand.rows != addend.rows || summand.cols != addend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    CPU_matrix_CSC<T> result(summand.rows, summand.cols);
    std::vector<std::vector<unsigned long long>> local_row_idx(result.cols);
    std::vector<std::vector<T>> local_values(result.cols);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long summand_start = summand.col_ptr[i];
        unsigned long long summand_end = summand.col_ptr[i + 1];
        unsigned long long addend_start = addend.col_ptr[i];
        unsigned long long addend_end = addend.col_ptr[i + 1];
        unsigned long long j = summand_start;
        unsigned long long k = addend_start;
        while (j < summand_end && k < addend_end) {
            unsigned long long j_row = summand.row_idx[j];
            unsigned long long k_row = addend.row_idx[k];
            if (j_row == k_row) {
                accumulator[j_row] += summand.values[j] + addend.values[k];
                ++j;
                ++k;
            } else if (j_row < k_row) {
                accumulator[j_row] += summand.values[j];
                ++j;
            } else {
                accumulator[k_row] += addend.values[k];
                ++k;
            }
        }
        while (j < summand_end) {
            accumulator[summand.row_idx[j]] += summand.values[j];
            ++j;
        }
        while (k < addend_end) {
            accumulator[addend.row_idx[k]] += addend.values[k];
            ++k;
        }
        for (const auto& pair: accumulator) {
            if (pair.second != 0) {
                local_row_idx[i].emplace_back(pair.first);
                local_values[i].emplace_back(pair.second);
            }
        }
        result.col_ptr[i + 1] = local_row_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.cols; ++i) {
        result.col_ptr[i] += result.col_ptr[i - 1];
    }
    result.nnz = result.col_ptr[result.cols];
    result.row_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_row_idx[i].size(); ++j) {
            result.row_idx[result.col_ptr[i] + j] = local_row_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.col_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> openmp_subtract(const CPU_matrix_CSC<T>& minuend, const T& subtrahend) {
    CPU_matrix_CSC<T> result(minuend.rows, minuend.cols);
    std::vector<std::vector<unsigned long long>> local_row_idx(result.cols);
    std::vector<std::vector<T>> local_values(result.cols);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long minuend_end = minuend.col_ptr[i + 1];
        for (unsigned long long j = minuend.col_ptr[i]; j < minuend_end; ++j) {
            T value = minuend.values[j] - subtrahend;
            if (value != 0) {
                accumulator[minuend.row_idx[j]] += value;
            }
        }
        for (const auto& pair: accumulator) {
            local_row_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.col_ptr[i + 1] = local_row_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.cols; ++i) {
        result.col_ptr[i] += result.col_ptr[i - 1];
    }
    result.nnz = result.col_ptr[result.cols];
    result.row_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_row_idx[i].size(); ++j) {
            result.row_idx[result.col_ptr[i] + j] = local_row_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.col_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> openmp_subtract(const CPU_matrix_CSC<T>& minuend, const CPU_matrix_CSC<T>& subtrahend) {
    if (minuend.rows != subtrahend.rows || minuend.cols != subtrahend.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    CPU_matrix_CSC<T> result(minuend.rows, minuend.cols);
    std::vector<std::vector<unsigned long long>> local_row_idx(result.cols);
    std::vector<std::vector<T>> local_values(result.cols);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long minuend_start = minuend.col_ptr[i];
        unsigned long long minuend_end = minuend.col_ptr[i + 1];
        unsigned long long subtrahend_start = subtrahend.col_ptr[i];
        unsigned long long subtrahend_end = subtrahend.col_ptr[i + 1];
        unsigned long long j = minuend_start;
        unsigned long long k = subtrahend_start;
        while (j < minuend_end && k < subtrahend_end) {
            unsigned long long j_row = minuend.row_idx[j];
            unsigned long long k_row = subtrahend.row_idx[k];
            if (j_row == k_row) {
                accumulator[j_row] += minuend.values[j] - subtrahend.values[k];
                ++j;
                ++k;
            } else if (j_row < k_row) {
                accumulator[j_row] += minuend.values[j];
                ++j;
            } else {
                accumulator[k_row] -= subtrahend.values[k];
                ++k;
            }
        }
        while (j < minuend_end) {
            accumulator[minuend.row_idx[j]] += minuend.values[j];
            ++j;
        }
        while (k < subtrahend_end) {
            accumulator[subtrahend.row_idx[k]] -= subtrahend.values[k];
            ++k;
        }
        for (const auto& pair: accumulator) {
            if (pair.second != 0) {
                local_row_idx[i].emplace_back(pair.first);
                local_values[i].emplace_back(pair.second);
            }
        }
        result.col_ptr[i + 1] = local_row_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.cols; ++i) {
        result.col_ptr[i] += result.col_ptr[i - 1];
    }
    result.nnz = result.col_ptr[result.cols];
    result.row_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_row_idx[i].size(); ++j) {
            result.row_idx[result.col_ptr[i] + j] = local_row_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.col_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T> openmp_multiply(const CPU_matrix_CSC<T>& multiplicand, const T& multiplier) {
    CPU_matrix_CSC<T> result(multiplicand.rows, multiplicand.cols);
    std::vector<std::vector<unsigned long long>> local_row_idx(result.cols);
    std::vector<std::vector<T>> local_values(result.cols);
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        std::map<unsigned long long, T> accumulator;
        unsigned long long multiplicand_end = multiplicand.col_ptr[i + 1];
        for (unsigned long long j = multiplicand.col_ptr[i]; j < multiplicand_end; ++j) {
            T value = multiplicand.values[j] * multiplier;
            if (value != 0) {
                accumulator[multiplicand.row_idx[j]] += value;
            }
        }
        for (const auto& pair: accumulator) {
            local_row_idx[i].emplace_back(pair.first);
            local_values[i].emplace_back(pair.second);
        }
        result.col_ptr[i + 1] = local_row_idx[i].size();
    }
    for (unsigned long long i = 1; i <= result.cols; ++i) {
        result.col_ptr[i] += result.col_ptr[i - 1];
    }
    result.nnz = result.col_ptr[result.cols];
    result.row_idx = new unsigned long long[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_row_idx[i].size(); ++j) {
            result.row_idx[result.col_ptr[i] + j] = local_row_idx[i][j];
        }
    }
    result.values = new T[result.nnz]();
    #pragma omp parallel for
    for (unsigned long long i = 0; i < result.cols; ++i) {
        for (unsigned long long j = 0; j < local_values[i].size(); ++j) {
            result.values[result.col_ptr[i] + j] = local_values[i][j];
        }
    }
    return result;
}

template<typename T>
CPU_matrix_CSC<T>::CPU_matrix_CSC(const CPU_matrix_CSC<T>& other): matrix_base<T>(other.rows, other.cols) {
    this->nnz = other.nnz;
    this->col_ptr = new unsigned long long[this->cols + 1]();
    for (unsigned long long i = 0; i <= this->cols; ++i) {
        this->col_ptr[i] = other.col_ptr[i];
    }
    this->row_idx = new unsigned long long[this->nnz]();
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        this->row_idx[i] = other.row_idx[i];
    }
    this->values = new T[this->nnz]();
    for (unsigned long long i = 0; i < this->nnz; ++i) {
        this->values[i] = other.values[i];
    }
}

template<typename T>
CPU_matrix_CSC<T>::CPU_matrix_CSC(const unsigned long long* col_ptr, const unsigned long long* row_idx, const T* values, unsigned long long nnz, unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols), nnz(nnz) {
    if (col_ptr == nullptr || row_idx == nullptr || values == nullptr) {
        throw std::invalid_argument("Null pointer passed to CPU_matrix_CSC constructor.");
    }
    this->col_ptr = new unsigned long long[cols + 1]();
    for (unsigned long long i = 0; i <= cols; ++i) {
        this->col_ptr[i] = col_ptr[i];
    }
    this->row_idx = new unsigned long long[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->row_idx[i] = row_idx[i];
    }
    this->values = new T[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->values[i] = values[i];
    }
}

template<typename T>
CPU_matrix_CSC<T>::CPU_matrix_CSC(const std::vector<unsigned long long>& col_ptr, const std::vector<unsigned long long>& row_idx, const std::vector<T>& values, unsigned long long nnz, unsigned long long rows, unsigned long long cols): matrix_base<T>(rows, cols), nnz(nnz) {
    if (col_ptr.empty() || row_idx.empty() || values.empty()) {
        throw std::invalid_argument("Empty vector passed to CPU_matrix_CSC constructor.");
    }
    this->col_ptr = new unsigned long long[cols + 1]();
    for (unsigned long long i = 0; i <= cols; ++i) {
        this->col_ptr[i] = col_ptr[i];
    }
    this->row_idx = new unsigned long long[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->row_idx[i] = row_idx[i];
    }
    this->values = new T[nnz]();
    for (unsigned long long i = 0; i < nnz; ++i) {
        this->values[i] = values[i];
    }
}
