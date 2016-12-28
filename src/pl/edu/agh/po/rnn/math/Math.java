package pl.edu.agh.po.rnn.math;

import java.util.Random;

public class Math
{

	public static final Random rand = new Random();

	// Create matrix

	public static Matrix randn(int M, int N) throws BadMatrixException
	{
	}

	public static Matrix randn(int M) throws BadMatrixException
	{
	}

	public static Matrix randomLike(Matrix m)
	{
	}


	// Matrix ops

	public static Matrix dot(Matrix a, Matrix b) throws BadDotException
	{
	}

	public static Matrix dot(double a, Matrix b)
	{
	}

	public static Matrix dot(Matrix a, double b)
	{
	}
}
