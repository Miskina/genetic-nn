#include <OpenBLAS/cblas.h>
#include <OpenBLAS/lapacke.h>
#include <algorithm>

#include "matrix.h"

matrix::~matrix() noexcept
{
	if(this->data_ != nullptr) delete [] data_;
}

matrix& matrix::operator=(matrix&& other) noexcept
{
	if(&other == this) return *this;

	if(data_ != nullptr) delete [] data_;

    data_ = other.data_;
	other.data_ = nullptr;
	this->rows_ = std::exchange(other.rows_, 0);
	this->columns_ = std::exchange(other.columns_, 0);
	return *this;
}

matrix& matrix::operator=(const matrix& other) noexcept
{
	if(&other == this) return *this;

	if(data_ != nullptr) delete [] data_;

    data_ = new double[other.rows_ * other.columns_];

	cblas_dcopy(other.rows_ * other.columns_, other.data_, 1, this->data_, 1);

	this->rows_ = other.rows_;
	this->columns_ = other.columns_;
	return *this;
}

matrix& matrix::operator*=(const matrix& m)
{
    size_t n = m.rows() * m.columns();
    size_t n_ = this->rows_ * this->columns_;
    n = n > n_ ? n_ : n;
    for(size_t i = 0; i < n; ++i)
    {
        this->data_[i] *= m.data_[i];
    }
    return *this;
}

matrix operator*(matrix m, const matrix& rhs)
{
    return m *= rhs;
}

matrix& matrix::operator*=(double scalar) noexcept
{
    // Mnozenje matrice sa skalarom.
    cblas_dscal(columns_ * rows_, scalar, data_, 1);
	return *this;
}

matrix operator*(matrix m, double scalar)
{
    return m *= scalar;
}

matrix operator*(double scalar, matrix m)
{
    return m *= scalar;
}

matrix& matrix::operator+=(const matrix& m)
{
	assert(m.rows_ == this->rows_ || m.columns_ == this->columns_ && "Matrices must have same dimensions!");
	// saxpy - y = alpha * x + y
    cblas_daxpy(rows_ * columns_, 1.f, m.data_, 1, data_, 1);
	return *this;
}

matrix operator+(matrix lhs, const matrix& rhs)
{
    return lhs += rhs;
}

matrix &matrix::operator-=(const matrix& m)
{
    assert(m.rows_ == this->rows_ || m.columns_ == this->columns_ && "Matrices must have same dimensions!");
    // saxpy - y = alpha * x + y s time da aplha = -1
    cblas_daxpy(rows_ * columns_, -1.f, m.data_, 1, data_, 1);
    return *this;
}

matrix operator-(matrix lhs, const matrix& rhs)
{
    return lhs -= rhs;
}

matrix::matrix_proxy matrix::operator[](size_t row)
{
	return {&this->data_[row * columns_], columns_};
}

matrix::matrix_proxy matrix::operator[](size_t row) const
{
	return {&this->data_[row * columns_], columns_};
}

double matrix::matrix_proxy::operator[](size_t column) const
{
	return this->arr_[column];
}

double& matrix::matrix_proxy::operator[](size_t column)
{
	return this->arr_[column];
}

matrix matrix::broadcast(size_t to_size) const
{
    assert((columns_ == 1) || (rows_ == 1) && "Can't broadcast matrix with both dimensions higher then 1!");

    auto brcast_columns = false;
	size_t rows = this->rows_;
	size_t columns = this->columns_;
	size_t n = rows;

    if(rows_ == 1)
    {
        brcast_columns = true;
        n = to_size;
    }

	if(brcast_columns)
	{
		rows = to_size;
	}
	else
	{
		columns = to_size;
	}

	matrix result(rows, columns);

	for(size_t i = 0; i < n; ++i)
	{
        if(brcast_columns)
        {
            std::copy(data_, data_ + columns_, result.data_ + i * columns_);
        }
        else
        {
		    std::fill_n(result.data_ + (i * columns), columns, data_[i]);
        }
	}

	return result;
}

size_t matrix::rows() const noexcept
{
	return this->rows_;
}

size_t matrix::columns() const noexcept
{
	return this->columns_;
}

matrix& matrix::matmul(const matrix& m) noexcept
{
    cblas_dgemm(CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                this->rows_, m.columns_, this->columns_,
                1.0f, this->data_, this->columns_,
                m.data_, m.columns_,
                0.0f, this->data_, m.columns_);
    return *this;
}

matrix matrix::transpose() const noexcept
{
    matrix result(this->columns_, this->rows_);
    cblas_domatcopy(CBLAS_ORDER::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    rows_, columns_, 1.f, data_, columns_, result.data_, rows_);

    return result;

}

matrix& matrix::transpose_in_place() noexcept
{
    cblas_dimatcopy(CBLAS_ORDER::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasTrans,
                    rows_, columns_, 1.0f, data_, columns_, rows_);
    std::swap(rows_, columns_);
    return *this;
}

template<>
matrix matrix::sum<0>(const matrix& m)
{
    matrix result(1, m.columns_);
    // Prvi red kopiramo u rezultat
    cblas_dcopy(m.columns_, m.data_, 1, result.data_, 1);
    for (size_t i = 1; i < m.rows_; ++i)
    {
        // Ostale redove dodaje na taj rezultat.
        cblas_daxpy(m.columns_, 1.f, m.data_ + i * m.columns_, 1, result.data_, 1);
    }
    return result;
}

template<>
matrix matrix::sum<1>(const matrix& m)
{
    matrix result(m.rows_, 1);
    // Prvi stupac kopiramo u rezultat. (incx = velicina retka).
    cblas_dcopy(m.rows_, m.data_, m.columns_, result.data_, 1);
    for (size_t i = 1; i < m.columns_; ++i)
    {
        // Ostale stupce dodajemo u rezultat. (incx = velicina retka).
        cblas_daxpy(m.rows_, 1.f, m.data_ + i, m.columns_, result.data_, 1);
    }
    return result;
}

double *& matrix::data()
{
    return data_;
}

const double * matrix::data() const
{
    return data_;
}

size_t matrix::size() const
{
    return rows_ * columns_;
}

matrix& matrix::add(const matrix& m, double multiplier)
{
    assert(m.rows_ == this->rows_ && m.columns_ == this->columns_ && "Matrices must have same dimensions!");
    cblas_daxpy(rows_ * columns_, multiplier, m.data_, 1, data_, 1);
    return *this;
}

std::optional<matrix> load_matrix(const char * file_name)
{
	std::ifstream file_stream(file_name);

	if(!file_stream.is_open()) return std::nullopt;

	size_t row_counter = 0;
	size_t column_counter = 0;

	std::vector<double> data{};

	std::string line;

	if(std::getline(file_stream, line))
	{

		std::istringstream line_stream(line);
		std::for_each(std::istream_iterator<double>(line_stream),
					  std::istream_iterator<double>(),
					  [&column_counter, &data](const double& val)
					  {
							data.push_back(val);
							column_counter++;
					  });
		row_counter++;
	}
	else
	{
		return std::nullopt;
	}

	while(std::getline(file_stream, line))
	{
		std::istringstream line_stream(line);

		std::for_each(std::istream_iterator<double>(line_stream),
					  std::istream_iterator<double>(),
					  [&data](const double& val)
					  {
					  		data.push_back(std::move(val));
					  });

		row_counter++;
	}

	if(data.size() > row_counter * column_counter)
	{
		return std::nullopt;
	}


	matrix result(row_counter, column_counter);
	cblas_dcopy(data.size(), data.data(), 1, result.data(), 1);
	
	return {std::move(result)};
}

matrix& matrix::inverse()
{
	assert(rows_ == columns_ && "Cannot find the inverse of a non-square matrix");
//		print(*this);
	int * ipiv = new int[rows_ + 1];
	int err = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, rows_, columns_, data_, columns_, ipiv);
//    	print(*this);
	assert(!err && "There was an error while trying to compute the LU decomposition of the matrix");
	
	err = LAPACKE_dgetri(LAPACK_ROW_MAJOR, rows_, data_, columns_, ipiv);
	assert(!err && "Error while trying to compute the inverse of the matrix");
	
	delete [] ipiv;
	
	return *this;
}
