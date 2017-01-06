package io.github.garstka.rnn.math;

import java.util.Random;

// Math helper functions
public class Math
{
	public static Random rand = new Random();
	public static double compareEpsilon = 1E-6;

	// Change seed

	public static void seed(long seed)
	{
		rand = new Random(seed);
	}


	// Create matrix

	public static Matrix randn(int M, int N)
	{
		//
	}

	// Random vector, components from the normal distribution
	public static Matrix randn(int M)
	{
		//
	}

	public static Matrix randomLike(Matrix m)
	{
		//
	}

	public static boolean close(double a, double b)
	{
		return java.lang.Math.abs(a - b) <= compareEpsilon;
	}

	// Applies tanh element-wise.
	public static Matrix tanh(Matrix m)
	{
		//
	}

	// Applies the softmax function with temperature = 1.0
	public static Matrix softmax(Matrix yAtt)
	{
		//
	}

	// Applies the softmax function with the given temperature.
	public static Matrix softmax(Matrix yAtt, double temperature)
	{
		//
	}
}
