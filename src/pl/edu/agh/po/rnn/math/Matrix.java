package pl.edu.agh.po.rnn.math;


import java.util.Random;

public class Matrix
{


	public Matrix(Matrix other)
	{
	}

	// M rows, N cols
	public static Matrix zeros(int M, int N) throws BadMatrixException
	{
	}

	public static Matrix zeros(int M) throws BadMatrixException
	{
	}

	public static Matrix zerosLike(Matrix m)
	{
	}

	public void zero()
	{
	}

	public void randomize(Random r)
	{
	}

	public Matrix dot(Matrix other) throws BadDotException
	{
	}

	public Matrix dot(double x)
	{
	}
}
