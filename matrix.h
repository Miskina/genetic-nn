#ifndef MATRIX_H
#define MATRIX_H

#include <OpenBLAS/cblas.h>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <sstream>
#include <optional>
#include <cassert>


struct matrix
{
    explicit matrix() : data_(nullptr), rows_(0), columns_(0) {}

    matrix(size_t rows, size_t columns) : data_(new double[rows * columns]), rows_(rows), columns_(columns) {}

    matrix(size_t rows, size_t columns, double * values) : data_(values), rows_(rows), columns_(columns) {}

    matrix(matrix&& m) : data_(m.data_), rows_(m.rows_), columns_(m.columns_)
    {
    	
    	m.data_ = nullptr;
    	m.rows_ = 0;
    	m.columns_ = 0;
//            std::swap(data_, m.data_);
//            std::swap(rows_, m.rows_);
//            std::swap(columns_, m.columns_);
    }

    matrix(const matrix& m) : data_(new double[m.rows_ * m.columns_]), rows_(m.rows_), columns_(m.columns_)
    {
        cblas_dcopy(m.rows_ * m.columns_, m.data_, 1, data_, 1);
    }

    matrix& operator=(const matrix&) noexcept;

    matrix& operator=(matrix&&) noexcept;

    ~matrix() noexcept;

    matrix& operator*=(const matrix&);

    matrix& operator*=(double) noexcept;

    friend matrix operator*(matrix, double);

    friend matrix operator*(double, matrix);

    friend matrix operator+(matrix lhs, const matrix& rhs);

    friend matrix operator-(matrix lhs, const matrix& rhs);

    friend matrix operator*(matrix m, const matrix& rhs);

    matrix& operator+=(const matrix&);

    matrix& operator-=(const matrix&);

    struct matrix_proxy
    {

        matrix_proxy(double * arr, const size_t columns) : arr_(arr), columns_(columns)
        {}

        double& operator[](size_t);

        double operator[](size_t) const;

        operator double &()
        {
            return *(this->arr_);
        }

    private:
        double * arr_;
        size_t columns_;
    };

    matrix_proxy operator[](size_t);

    matrix_proxy operator[](size_t) const;

    matrix broadcast(size_t) const;

    size_t rows() const noexcept;

    size_t columns() const noexcept;

    double *& data();

    const double * data() const;

    size_t size() const;

    matrix& add(const matrix&, double);

    matrix& matmul(const matrix&) noexcept;

    matrix transpose() const noexcept;

    matrix& transpose_in_place() noexcept;
    
    matrix& inverse();

    template<bool TransposeA = false, bool TransposeB = false>
    static matrix matmul(const matrix& A, const matrix& B)
    {
        auto m = TransposeA ? A.columns_ : A.rows_;
        auto n = TransposeB ? B.rows_ : B.columns_;
        auto k = TransposeA ? A.rows_ : A.columns_;
        matrix result(m, n);
        cblas_dgemm(CBLAS_ORDER::CblasRowMajor,
                    TransposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                    TransposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k,
                    1.0f, A.data_, A.columns_,
                    B.data_, B.columns_,
                    0.0f, result.data_, n);
        return result;
    }
    
    template<bool TransposeA = false, bool TransposeB = false>
    static matrix& matmul(const matrix& A, const matrix& B, matrix& result)
    {
        auto m = TransposeA ? A.columns_ : A.rows_;
        auto n = TransposeB ? B.rows_ : B.columns_;
        auto k = TransposeA ? A.rows_ : A.columns_;
        assert(m == result.rows_ && n == result.columns_ && "The given matrix for storing result should be of appropriate dimensions");
        cblas_dgemm(CBLAS_ORDER::CblasRowMajor,
                    TransposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                    TransposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k,
                    1.0, A.data_, A.columns_,
                    B.data_, B.columns_,
                    0.0, result.data_, n);
        return result;
    }

    template<bool TransposeA = false, bool TransposeB = false>
    static matrix fma(const matrix& A, const matrix& B, const matrix& C)
    {
        auto m = TransposeA ? A.columns_ : A.rows_;
        auto n = TransposeB ? B.rows_ : B.columns_;
        auto k = TransposeA ? A.rows_ : A.columns_;
        auto result = C.broadcast(m);
        cblas_dgemm(CBLAS_ORDER::CblasRowMajor,
                    TransposeA ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                    TransposeB ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k,
                    1.0, A.data_, A.columns_,
                    B.data_, B.columns_,
                    1.0, result.data_, n);
        return result;
    }

    static double asum(const matrix& m)
    {
        return cblas_dasum(m.rows_ * m.columns_, m.data_, 1);
    }

    template<uint8_t Dimension>
    static matrix sum(const matrix& m);

private:
    double * data_{nullptr};
    size_t rows_{0};
    size_t columns_{0};
};

inline void print(const matrix& m)
{
    for (size_t i = 0; i < m.rows(); ++i)
    {
        for (size_t j = 0; j < m.columns(); ++j)
        {
            printf("%.5lf\t", m[i][j]);
        }

        printf("\n");
    }
}


std::optional<matrix> load_matrix(const char * file_name);


#endif
