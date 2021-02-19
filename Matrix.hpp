#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#define DIM 3		// default value of row and col


class Matrix {
public:
	Matrix(int = DIM, int = DIM, double = 0.0);
	Matrix(const std::vector<std::vector<double>>&);
	Matrix(const Matrix&);
	Matrix& operator=(const Matrix&);
	~Matrix() = default;

	int getRow() const;
	int getCol() const;
	std::vector<double>& at(int);
	std::vector<double>& operator[](int);

	Matrix inverse() const;
	Matrix transpose() const;
	Matrix time(double) const;
	Matrix time(const Matrix&) const;
	Matrix dot(const Matrix&) const;
	Matrix operator*(const Matrix&) const;
	Matrix operator+(const Matrix&) const;
	Matrix operator-(const Matrix&) const;

	static Matrix diagonal(int = DIM);

	friend std::ostream& operator<<(std::ostream&, const Matrix&);

private:
	int row, col;
	std::vector<std::vector<double>> data;
};


#endif