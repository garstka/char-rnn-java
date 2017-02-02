package io.github.garstka.rnn.math;

import io.github.garstka.rnn.math.exceptions.NotAVectorException;
import io.github.garstka.rnn.math.exceptions.NotRectangularArrayException;

import java.io.Serializable;
import java.util.function.UnaryOperator;

// MxN Matrix
// If M=1 or N=1, the matrix is treated as a k-vector v.
// v is equivalent to v.T() in scenarios like matrix multiplication,
// or element-wise addition.
public class Matrix implements Serializable
{
	private int M; // rows
	private int N; // cols
	private double[][] data; // MxN

	/* Create */

	// Constructs by copying other. Requires other != null.
	public Matrix(Matrix other)
	{
		if (other == null)
			throw new NullPointerException(
			    "Non-null Matrix expected for copy-construction.");

		this.M = other.M;
		this.N = other.N;
		this.data = Utils.deepCopyOf(other.data);
	}

	// Constructs using an MxN array. Requires M, N > 0.
	private Matrix(double data[][])
	{
		try
		{
			this.M = Utils.arrayRows(data);
			this.N = Utils.arrayCols(data);

			if (this.M == 0 || this.N == 0) // Don't accept 0 as a size.
				throw new IllegalArgumentException(
				    "One of the array dimensions is 0.");

			this.data = data;
		}
		catch (NotRectangularArrayException | IllegalArgumentException e)
		{
			throw new IllegalArgumentException(
			    "MxN array expected, where M != 0 and N != 0.", e);
		}
	}

	// Constructs using an M*N row-major array. Requires M > 0.
	private Matrix(int M, double data[])
	{
		if (!(M > 0))
			throw new IllegalArgumentException(
			    "M > 0 expected as a matrix dimension.");
		if (!(data != null && data.length != 0 && data.length % M == 0))
			throw new IllegalArgumentException("A row-major array expected.");

		this.M = M;
		this.N = data.length % this.M;

		for (int i = 0; i < this.M; i++)
			for (int j = 0; j < this.N; j++)
				this.data[i][j] = data[M * i + j];
	}

	// Returns a matrix constructed using an MxN array. Requires M, N > 0.
	public static Matrix fromRaw(double data[][])
	{
		return new Matrix(Utils.deepCopyOf(data));
	}

	// Constructs using an M*N row-major array. Requires M > 0.
	public static Matrix fromFlat(int M, double data[])
	{
		return new Matrix(M, data);
	}

	// Returns a matrix with all zeros, M rows, N cols. Requires M, N > 0.
	public static Matrix zeros(int M, int N)
	{
		if (!(M > 0 && N > 0))
			throw new IllegalArgumentException(
			    "M > 0 and N > 0 expected as matrix dimensions.");

		return new Matrix(new double[M][N]);
	}

	// Returns a k-dimensional vector with all zeros. Requires k > 0.
	public static Matrix zeros(int k)
	{
		if (!(k > 0))
			throw new IllegalArgumentException(
			    "k > 0 expected for a k-dimensional vector.");

		return zeros(1, k);
	}

	// Returns a matrix shaped like other with all zeros. Requires other !=
	// null.
	public static Matrix zerosLike(Matrix other)
	{
		if (other == null)
			throw new NullPointerException(
			    "Non-null Matrix expected as a shape template.");

		return zeros(other.getM(), other.getN());
	}

	// Returns a matrix with all ones, M rows, N cols. Requires M, N > 0.
	public static Matrix ones(int M, int N)
	{
		if (!(M > 0 && N > 0))
			throw new IllegalArgumentException(
			    "M > 0 and N > 0 expected as matrix dimensions.");

		return zeros(M, N).add(1.0);
	}

	// Returns a k-dimensional vector with all ones. Requires k > 0.
	public static Matrix ones(int k)
	{
		if (!(k > 0))
			throw new IllegalArgumentException(
			    "k > 0 expected for a k-dimensional vector.");

		return ones(1, k);
	}

	// Returns a matrix shaped like other with all ones.
	public static Matrix onesLike(Matrix other)
	{
		if (other == null)
			throw new NullPointerException(
			    "Non-null Matrix expected as a shape template.");

		return ones(other.getM(), other.getN());
	}

	// Returns a one-hot vector (v.at(i) = 1, v.at(j) = 0 ; j != i).
	// Requires 0 <= i < k.
	public static Matrix oneHot(int k, int i)
	{
		if (!(0 <= i && i < k))
			throw new IllegalArgumentException(
			    "0 <= i < k expected to create a one hot k-dimensional vector.");

		Matrix v = zeros(k);
		v.setAt(i, 1.0);
		return v;
	}

	// Returns the matrix product (a x b)
	public static Matrix dot(Matrix a, Matrix b)
	{
		// System.out.println("a: " + a.M + "x" + a.N + "b:" + b.M + "x" + b.N);
		if (a.N != b.M) // if dimensions are not compatible
		{

			if (a.N == 1 && b.N == 1) // if both are vectors
			{
				// b^T is compatible (an outer product)
				b = b.T();
			}
			else if (b.M == 1 // if the second one is a vector,
			    && a.N == b.N) // and the other dimension matches
			{
				// b^T is compatible
				b = b.T();
			}
			else
				throw new RuntimeException(
				    "Incompatible dimensions for matrix multiplication.");
		}

		double[][] c = new double[a.M][b.N];
		for (int i = 0; i < a.M; i++)
			for (int j = 0; j < b.N; j++)
				for (int k = 0; k < a.N; k++)
					c[i][j] += a.data[i][k] * b.data[k][j];

		return new Matrix(c);
	}

	// Returns the transpose of this matrix.
	public Matrix T()
	{
		double[][] new_data = new double[N][M];

		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				new_data[j][i] = data[i][j];

		return new Matrix(new_data);
	}

	/* Operators */

	// Adds x to all elements.
	public Matrix add(double x)
	{
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				row[j] += x;
		return this;
	}

	// Adds element-wise. Requires the matrices to have the same
	// dimensions.
	public Matrix add(Matrix other)
	{
		if (M == other.M && N == other.N) // compatible matrices
		{
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					data[i][j] += other.data[i][j];
		}
		else if (M == other.N && N == 1 && other.M == 1)
		{
			// this is a row vector and other is a column vector
			for (int i = 0; i < M; i++)
				data[i][0] += other.data[0][i];
		}
		else if (N == other.M && M == 1 && other.N == 1)
		{
			// this is a column vector and other is a row vector
			for (int i = 0; i < N; i++)
				data[0][i] += other.data[i][0];
		}
		else
			throw new RuntimeException(
			    "Matrices/vectors incompatible for element-wise addition.");

		return this;
	}

	// Multiplies all elements by x.
	public Matrix mul(double x)
	{
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				row[j] *= x;
		return this;
	}

	// Multiplies element-wise.
	public Matrix mul(Matrix other)
	{
		if (M == other.M && N == other.N) // compatible matrices
		{
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					data[i][j] *= other.data[i][j];
		}
		else if (M == other.N && N == 1 && other.M == 1)
		{
			// this is a row vector and other is a column vector
			for (int i = 0; i < M; i++)
				data[i][0] *= other.data[0][i];
		}
		else if (N == other.M && M == 1 && other.N == 1)
		{
			// this is a column vector and other is a row vector
			for (int i = 0; i < N; i++)
				data[0][i] *= other.data[i][0];
		}
		else
			throw new RuntimeException(
			    "Matrices/vectors incompatible for element-wise multiplication.");

		return this;
	}

	// Multiplies all elements by -1.0.
	public Matrix neg()
	{
		return mul(-1.0);
	}

	// Divides all elements by x.
	public Matrix div(double x)
	{
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				row[j] /= x;
		return this;
	}

	// Divides element-wise.
	public Matrix div(Matrix other)
	{
		if (M == other.M && N == other.N) // compatible matrices
		{
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					data[i][j] /= other.data[i][j];
		}
		else if (M == other.N && N == 1 && other.M == 1)
		{
			// this is a row vector and other is a column vector
			for (int i = 0; i < M; i++)
				data[i][0] /= other.data[0][i];
		}
		else if (N == other.M && M == 1 && other.N == 1)
		{
			// this is a column vector and other is a row vector
			for (int i = 0; i < N; i++)
				data[0][i] /= other.data[i][0];
		}
		else
			throw new RuntimeException(
			    "Matrices/vectors incompatible for element-wise division.");

		return this;
	}

	// Applies e^x element-wise.
	public Matrix exp()
	{
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				row[j] = java.lang.Math.exp(row[j]);
		return this;
	}

	// Applies tanh(x) element-wise.
	public Matrix tanh()
	{
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				row[j] = java.lang.Math.tanh(row[j]);
		return this;
	}

	// Clips all elements to the interval [x_a, x_b]. Requires that x_a < x_b
	public Matrix clip(double x_a, double x_b)
	{
		if (!(x_a < x_b))
			throw new IllegalArgumentException(
			    "An interval [a,b], a < b expected.");

		for (double[] row : data)
			for (int j = 0; j < N; j++)
			{
				if (row[j] < x_a)
					row[j] = x_a;
				else if (x_b < row[j])
					row[j] = x_b;
			}

		return this;
	}

	// Calls f for each element.
	public Matrix apply(UnaryOperator<Double> f)
	{
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				row[j] = f.apply(row[j]).doubleValue();
		return this;
	}

	/* Other of all elements */

	// Returns the sum of elements.
	public double sum()
	{
		double sum = 0.0;
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				sum += row[j];

		return sum;
	}

	// Returns the product of elements.
	public double prod()
	{
		double prod = 0.0;
		for (double[] row : data)
			for (int j = 0; j < N; j++)
				prod *= row[j];

		return prod;
	}

	// Returns a copy of the MxN array.
	public double[][] raw()
	{
		return Utils.deepCopyOf(data);
	}

	// Returns a row-major flattened array.
	public double[] unravel()
	{
		double[] result = new double[M * N];

		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				result[N * i + j] = data[i][j];

		return result;
	}

	/* State */

	// Returns true, if the matrix is a vector.
	public boolean isVector()
	{
		return M == 1 || N == 1;
	}

	// Returns the index with the value 1.0. Requires the matrix to be a one-hot
	// vector.
	public int oneHotIndex()
	{
		if (!(M == 1 || N == 1))
			throw new NotAVectorException(
			    "A vector expected to be able to return the one-hot index.");

		boolean one_already_encountered = false;
		int one_hot_index = 0;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
			{
				if (Math.close(data[i][j], 0.0)) // ignore zeros
				{
					// continue
				}
				else if (Math.close(data[i][j], 1.0)) // allow a single one
				{
					if (one_already_encountered)
						throw new RuntimeException(
						    "A one-hot vector can't have multiple ones.");

					one_already_encountered = true;
					one_hot_index = (M == 1 ? j : i);
				}
				else
					throw new RuntimeException(
					    "A one-hot vector can't have elements other than 0 or 1.");
			}

		if (!one_already_encountered)
			throw new RuntimeException("One-hot vector can't be all zeros.");

		return one_hot_index;
	}

	/* Dimensions */

	// Returns the row count.
	public int getM()
	{
		return M;
	}

	// Returns the column count.
	public int getN()
	{
		return N;
	}

	// Returns the vector length. Requires that the matrix is a vector.
	public int getk()
	{
		if (M == 1)
			return N;
		else if (N == 1)
			return M;
		else
			throw new NotAVectorException("The matrix is not a vector.");
	}

	/* Element access */

	// Returns the vector element at i. Requires i < k
	public double at(int i)
	{
		if (!(M == 1 || N == 1))
			throw new NotAVectorException("The matrix is not a vector.");

		if (!(i >= 0 && (M == 1 ? i < N : i < M)))
			throw new IndexOutOfBoundsException(
			    "Vector element index out of bounds.");

		if (M == 1)
			return data[0][i]; // row vector
		return data[i][0]; // column vector
	}

	// Returns the matrix element at i,j. Requires i < M, j < N.
	public double at(int i, int j)
	{
		if (!(0 <= i && i < M && 0 <= j && j < N))
			throw new IndexOutOfBoundsException(
			    "Matrix element index out of bounds.");

		return data[i][j];
	}

	// Returns the vector element at m.oneHotIndex(). Requires index to be a
	// one-hot vector.
	public double at(Matrix index)
	{
		if (index == null)
			throw new NullPointerException(
			    "Non-null Matrix expected as a one hot index.");

		return at(index.oneHotIndex());
	}

	// Sets the vector element at i to x. Requires 0 <= i < k.
	public void setAt(int i, double x)
	{
		if (!(M == 1 || N == 1))
			throw new NotAVectorException("The matrix is not a vector.");

		if (!(i >= 0 && (M == 1 ? i < N : i < M)))
			throw new IndexOutOfBoundsException(
			    "Vector element index out of bounds.");

		if (M == 1)
			data[0][i] = x; // row vector
		if (N == 1)
			data[i][0] = x; // column vector
	}

	// Sets the matrix element at i,j to x. Requires 0 <= i < M and 0 <= j < N.
	public void setAt(int i, int j, double x)
	{
		if (!(0 <= i && i < M && 0 <= j && j < N))
			throw new IndexOutOfBoundsException(
			    "Matrix element index out of bounds.");

		data[i][j] = x;
	}

	// Sets the vector element at m.oneHotIndex() to x. Requires index to be a
	// one-hot vector, and its index i < k.
	public void setAt(Matrix m, double x)
	{
		setAt(m.oneHotIndex(), x);
	}
}