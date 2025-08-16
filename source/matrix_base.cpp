#include "matrix_base.h"

template<typename T>
unsigned long long matrix_base<T>::get_rows() const {
    return this->rows;
}

template<typename T>
unsigned long long matrix_base<T>::get_cols() const {
    return this->cols;
}

template<typename T>
matrix_base<T>::matrix_base() {
    this->rows = 0;
    this->cols = 0;
}

template<typename T>
matrix_base<T>::matrix_base(unsigned long long rows, unsigned long long cols) {
    this->rows = rows;
    this->cols = cols;
}

template<typename T>
matrix_base<T>::~matrix_base() {
    this->rows = 0;
    this->cols = 0;
}

template class matrix_base<double>;
