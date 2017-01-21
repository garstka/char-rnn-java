package io.github.garstka.rnn.math;

// Math helper functions
public class Math
{
	public static double compareEpsilon = 1E-6;

	/* Double epsilon compare*/

	public static boolean close(double a, double b)
	{
		return java.lang.Math.abs(a - b) <= compareEpsilon;
	}

	public static boolean close(double a, double b, double eps)
	{
		return java.lang.Math.abs(a - b) <= eps;
	}

	// return the comparison epsilon
	public static double eps()
	{
		return compareEpsilon;
	}

	/* Useful Matrix functions */

	// Applies the softmax function with temperature = 1.0
	public static Matrix softmax(Matrix yAtt)
	{
		Matrix e_to_x = new Matrix(yAtt).exp();
		e_to_x = e_to_x.div(e_to_x.sum());
		return e_to_x;
	}

	// Applies the softmax function with the given temperature.
	// Temperature can't be close to 0.
	public static Matrix softmax(Matrix yAtt, double temperature)
	{
		Matrix e_to_x = new Matrix(yAtt).div(temperature).exp();
		e_to_x = e_to_x.div(e_to_x.sum());
		return e_to_x;
	}
}
