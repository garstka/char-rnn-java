package pl.edu.agh.po.rnn.math;


import java.util.Random;
import java.util.function.UnaryOperator;

// MxN Matrix
// If M=1 or N=1, the matrix is treated as a k-vector v. Additionally,
// v is treated as equivalent to v.T() in matrix multiplication.
public class Matrix
{

	/* Create */

	// Copy other
	public Matrix(Matrix other)
	{
		//
	}

	// All zeros, M rows, N cols
	public static Matrix zeros(int M, int N)
	{
		//
	}

	// All zeros, k-vector
	public static Matrix zeros(int k)
	{
		//
	}

	// Returns matrix shaped like m.
	public static Matrix zerosLike(Matrix other)
	{
		//
	}

	// All ones, M rows, N cols
	public static Matrix ones(int M, int N)
	{
		//
	}

	// All ones, k-vector
	public static Matrix ones(int k)
	{
		//
	}

	// All ones, shaped like m.
	public static Matrix onesLike(Matrix other)
	{
		//
	}

	// Returns a one-hot vector (v.at(i) = 1, v.at(j) = 0 ; j != i)
	public static Matrix oneHot(int k, int i)
	{
		//
	}

	// Returns the matrix product (a x b)
	public static Matrix dot(Matrix a, Matrix b)
	{
		//
	}

	/* Other static */

	// Applies tanh element-wise.
	public static Matrix tanh(Matrix a)
	{
		//
	}

	// Returns the sum of elements.
	public static double sum(Matrix a)
	{
		//
	}

	/* Operators */

	// Applies the softmax function with temperature = 1.0
	public static Matrix softmax(Matrix yAtt)
	{
	}

	// Applies the softmax function.
	public static Matrix softmax(Matrix yAtt, double temperature)
	{
	}


	// Multiplies all elements by x.
	public Matrix mul(double x)
	{
		//
	}

	// Multiplies element-wise.
	public Matrix mul(Matrix other)
	{
		//
	}

	// Adds element-wise.
	public Matrix add(Matrix other)
	{
		//
	}

	// Negates all elements
	public Matrix neg()
	{
		//
	}

	/* Element access */

	// Accesses the vector element at i.
	public double at(int i)
	{
		//
	}

	// Accesses the matrix element at i,j.
	public double at(int i, int j)
	{
		//
	}

	// Accesses the matrix element at m.oneHotIndex()
	public double at(Matrix m)
	{
		//
	}

	// Iff m is a one-hot vector, returns the 1.0 element's index
	public int oneHotIndex()
	{
		//
	}

	public void setAt(int i, double x)
	{
		//
	}

	public void setAt(int i, int j, double x)
	{
		//
	}

	public void setAt(Matrix m, double x)
	{
		//
	}

	// Returns the transpose of the matrix.
	public Matrix T()
	{
		//
	}

	public Matrix clip(double x_a, double x_b)
	{
		//
	}

	// For each elem

	public Matrix apply(UnaryOperator<Double> f)
	{
		//
		return this;
	}
}
