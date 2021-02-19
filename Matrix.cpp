#include <Matrix.hpp>

using namespace std;


/* initialize matrix with a given row and column
 * @param  row		the number of row
 * @param  col		the number of column
 * @param  val		initial value
 */
Matrix::Matrix(int row, int col, double val) {
	this->row = row;
	this->col = col;
	this->data = vector<vector<double>>(row, vector<double>(col, val));
}

/* initialize matrix with a given vector
 * @param  src		vector representing a matrix
 */
Matrix::Matrix(const vector<vector<double>>& src) {
	try {
		this->row = src.size();
		this->col = src[0].size();
		this->data = src;

	} catch (exception e) {
		this->row = this->col = DIM;
		this->data = vector<vector<double>>(DIM, vector<double>(DIM, 0.0));
	}
}

/* initialize matrix with a given matrix
 * @param  src		a given matrix
 */
Matrix::Matrix(const Matrix& src) {
	this->row = src.row;
	this->col = src.col;
	this->data = src.data;
}

/* assign the matrix with a given matrix
 * @overload		operator=
 * @param			a given matrix
 * @return			the current matrix
 */
Matrix& Matrix::operator=(const Matrix& src) {
	this->row = src.row;
	this->col = src.col;
	this->data = src.data;

	return *this;
}

/* get the number of row of the matrix
 * @return			the number of row
 */
int Matrix::getRow() const {
	return this->row;
}

/* get the number of column of the matrix
 * @return			the number of column
 */
int Matrix::getCol() const {
	return this->col;
}

/* get a row from the matrix
 * @param  index	a row index of the matrix
 * @return			the index(th) row of the matrix
 */
vector<double>& Matrix::at(int index) {
	return this->data[index];
}

/* get a row from the matrix
 * @overload		operator[]
 * @param  index	a row index of the matrix
 * @return			the index(th) row of the matrix
 */
vector<double>& Matrix::operator[](int index) {
	return this->data[index];
}

/* get the inverse of the current matrix
 * @return res		the inverse of the current matrix
 */
Matrix Matrix::inverse() const {
	if (this->row != this->col)
		return Matrix();

	int n = this->row;
	vector<vector<double>> res(this->data);
	vector<int> r(n, 0), c(n, 0);

	for (int k = 0; k < n; ++k) {
		double d = 0.0;
		for (int y = k; y < n; ++y) {
			for (int x = k; x < n; ++x) {
				if (fabs(res[y][x]) > d) {
					d = fabs(res[y][x]);
					r[k] = y;
					c[k] = x;
				}
			}
		}

		if (fabs(d) < 1e-5)
			return Matrix(this->row, this->col);
		if (r[k] != k) {
			for (int j = 0; j < n; ++j) {
				swap(res[k][j], res[r[k]][j]);
			}
		}
		if (c[k] != k) {
			for (int i = 0; i < n; ++i) {
				swap(res[i][k], res[i][c[k]]);
			}
		}
		
		res[k][k] = 1.0 / res[k][k];
		for (int j = 0; j < n; ++j) {
			if (j != k) {
				res[k][j] *= res[k][k];
			}
		}
		for (int i = 0; i < n; ++i) {
			if (i != k) {
				for (int j = 0; j < n; ++j) {
					if (j != k) {
						res[i][j] -= res[i][k] * res[k][j];
					}
				}
			}
		}
		for (int i = 0; i < n; ++i) {
			if (i != k) {
				res[i][k] *= -res[k][k];
			}
		}
	}

	for (int k = n-1; k >= 0; --k) {
		if (c[k] != k) {
			for (int j = 0; j < n; ++j) {
				swap(res[k][j], res[c[k]][j]);
			}
		}
		if (r[k] != k) {
			for (int i = 0; i < n; ++i) {
				swap(res[i][k], res[i][r[k]]);
			}
		}
	}

	return Matrix(res);
}

/* get the transpose of the current matrix
 * @return res		the transpose of the current matrix
 */
Matrix Matrix::transpose() const {
	Matrix res(this->col, this->row);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			res[j][i] = this->data[i][j];
		}
	}

	return res;
}

/* get the scalar product of a given matrix and a constant
 * @param  a		constant
 * @return res 		scalar product of two given matrix
 */
Matrix Matrix::time(double a) const {
	Matrix res(this->row, this->col);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			res[i][j] = a * this->data[i][j];
		}
	}

	return res;
}

/* get the scalar product of two given matrix
 * @param  other	another matrix
 * @return res 		scalar product of two given matrix
 */
Matrix Matrix::time(const Matrix& other) const {
	if (this->row != other.row || this->col != other.col)
		return Matrix();

	Matrix res(this->row, this->col);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			res[i][j] = this->data[i][j] * other.data[i][j];
		}
	}

	return res;
}

/* get the dot product of two given matrix
 * @param  other	another matrix
 * @return res 		dot product of two given matrix
 */
Matrix Matrix::dot(const Matrix& other) const {
	if (this->col != other.row)
		return Matrix();

	Matrix res(this->row, other.col);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < other.col; ++j) {
			for (int k = 0; k < this->col; ++k) {
				res[i][j] += this->data[i][k] * other.data[k][j];
			}
			if (fabs(res[i][j]) < 1e-5) {
				res[i][j] = 0;
			}
		}
	}

	return res;
}

/* get the dot product of two given matrix
 * @overload		operator*
 * @param  other	another matrix
 * @return res 		dot product of two given matrix
 */
Matrix Matrix::operator*(const Matrix& other) const {
	if (this->col != other.row)
		return Matrix();

	Matrix res(this->row, other.col);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < other.col; ++j) {
			for (int k = 0; k < this->col; ++k) {
				res[i][j] += this->data[i][k] * other.data[k][j];
			}
			if (fabs(res[i][j]) < 1e-5) {
				res[i][j] = 0;
			}
		}
	}

	return res;
}

/* get the sum of two given matrix
 * @overload		operator+
 * @param  other	another matrix
 * @return res 		sum of two given matrix
 */
Matrix Matrix::operator+(const Matrix& other) const {
	if (this->row != other.row || this->col != other.col)
		return Matrix();

	Matrix res(this->row, this->col);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			res[i][j] = this->data[i][j] + other.data[i][j];
		}
	}

	return res;
}

/* get the difference of two given matrix
 * @overload		operator-
 * @param  other	another matrix
 * @return res 		difference of two given matrix
 */
Matrix Matrix::operator-(const Matrix& other) const {
	if (this->row != other.row || this->col != other.col)
		return Matrix();

	Matrix res(this->row, this->col);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			res[i][j] = this->data[i][j] - other.data[i][j];
		}
	}

	return res;
}

/* get a diagonal matrix with given shape
 * @param  n		the shape of matrix
 * @return res		a diagonal matrix
 */
Matrix Matrix::diagonal(int n) {
	Matrix res(n, n);
	for (int i = 0; i < n; ++i) {
		res[i][i] = 1.0;
	}

	return res;
}

/* ostream for printing the matrix
 * @overload		operator<<
 * @param  src 		a given matrix
 * @return out 		an ostream object
 */
ostream& operator<<(ostream& out, const Matrix& src) {
	for (int i = 0; i < src.row; ++i) {
		for (int j = 0; j < src.col; ++j) {
			out << src.data[i][j] << " ";
		}
		if (i != src.row-1) {
			out << endl;
		}
	}

	return out;
}
