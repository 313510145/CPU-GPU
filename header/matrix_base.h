#ifndef MATRIX_BASE_H
#define MATRIX_BASE_H

const unsigned long long THREAD_NUM = 256;
const unsigned long long THREAD_NUM_SQUARE_ROOT = 16;

template<typename T>
class matrix_base {
    public:
        unsigned long long get_rows() const;
        unsigned long long get_cols() const;
        matrix_base();
        matrix_base(unsigned long long rows, unsigned long long cols);
        virtual ~matrix_base();
    protected:
        unsigned long long rows, cols;
};

#endif  // MATRIX_BASE_H
