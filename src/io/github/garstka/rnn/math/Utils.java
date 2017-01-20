package io.github.garstka.rnn.math;

import io.github.garstka.rnn.math.exceptions.NotRectangularArrayException;

import java.util.Arrays;

// Utility helper funtions.
public class Utils
{
	/* Utilities for 2D arrays. */

	// Performs a deep copy of a 2D array of doubles.
	static public double[][] deepCopyOf(double[][] src)
	{
		if (src == null)
			return null;

		double[][] dst = new double[src.length][];
		for (int i = 0; i < src.length; i++)
			if (src[i] != null)
				dst[i] = Arrays.copyOf(src[i], src[i].length);

		return dst;
	}

	// Returns the row count of a 2D array of doubles. Treats null as size-0
	// array.
	static public int arrayRows(double[][] array)
	{
		if (array == null)
			return 0;
		return array.length;
	}

	// Returns the col count of a 2D array of doubles.
	// Throws, if array is not rectangular. Treats null as size-0 array.
	static public int arrayCols(double[][] array)
	    throws NotRectangularArrayException
	{
		if (array == null)
			return 0;

		// store expected length
		int length = 0;
		if (array[0] != null)
			length = array[0].length;

		// verify if all rows have that many cols
		for (int i = 1; i < array.length; i++)
			if ((array[i] == null && length != 0) || array[i].length != length)
				throw new NotRectangularArrayException();

		return length;
	}
}
