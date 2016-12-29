package pl.edu.agh.po.rnn.math;

import java.util.Random;

// Math helper functions
public class Math
{
	public static Random rand = new Random();

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
}
