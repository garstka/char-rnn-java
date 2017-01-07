package io.github.garstka.rnn.math;

// Helper functions for randomness.
public class Random
{
	private static java.util.Random rand = new java.util.Random();

	/* Initialize */

	// Change seed.
	public static void reseed(long seed)
	{
		rand = new java.util.Random(seed);
	}

	/* Random matrix*/

	// Returns an MxN matrix filled with numbers drawn from a standard normal
	// distribution.
	// Requires that M > 0 and N > 0.
	public static Matrix randn(int M, int N)
	{
		if (!(M > 0 && N > 0))
			throw new IllegalArgumentException(
			    "M,N > 0 expected for matrix dimensions.");

		Matrix m = Matrix.zeros(M, N);
		m.apply((d) -> rand.nextGaussian());
		return m;
	}

	// Returns an k-dimensional vector filled with numbers drawn from a standard
	// normal distribution.
	// Requires that k > 0.
	public static Matrix randn(int k)
	{
		if (!(k > 0))
			throw new IllegalArgumentException(
			    "k > 0 expected for vector size.");

		return randn(1, k);
	}

	// Returns a matrix shaped like m filled with numbers drawn from a standard
	// normal distribution.
	// Requires that m != null.
	public static Matrix randomLike(Matrix m)
	{
		if (m == null)
			throw new NullPointerException("Non-null m expected.");

		return randn(m.getM(), m.getN());
	}

	/* Random choice */

	// Samples an index from a random distribution with the given probabilities
	// p. Will work properly, if the sum of probabilities is 1.0.
	// Requires that p != null.
	public static int randomChoice(double[] p)
	{
		if (p == null)
			throw new NullPointerException(
			    "The array of probabilities can't be null.");

		double random = rand.nextDouble();
		double cumulative = 0.0;

		for (int i = 0; i < p.length; i++)
		{
			cumulative += p[i];
			if (cumulative > random)
				return i;
		}
		return p.length - 1; // Fallback: probabilities did not sum up to a 1.0;
	}
}
