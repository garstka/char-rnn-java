package pl.edu.agh.po.rnn.net;

import pl.edu.agh.po.rnn.math.*;
import pl.edu.agh.po.rnn.math.Math;

import java.util.Vector;

public class RNNLayer
{
	// Hyperparameters
	private int hiddenCount; // Size of a single hidden layer of neurons.
	private int sequenceLength; // Number of steps to unroll the RNN for backpropagation.
	private double learningRate; // Backpropagation parameter.

	// Defaults
	public static final int defaultHiddenCount = 100;
	public static final int defaultSequenceLength = 25;
	public static final double defaultLearningRate = 0.1;

	private int inputSize; // input vector size

	// Weights
	private Matrix Wxh; // input layer weights
	private Matrix Whh; // hidden layer weights
	private Matrix Why; // output layer weights
	private Matrix bh; // hidden bias
	private Matrix by; // output bias

	private Matrix gWxh; // gradient descent params: input layer
	private Matrix gWhh; // gradient descent params: hidden layer
	private Matrix gWhy; // gradient descent params: output layer
	private Matrix gbh; // gradient descent params: hidden bias
	private Matrix gby; // gradient descent params: output bias


	// Current state

	private boolean initialized;

	private Matrix h; // last hidden state

	// private boolean

	// Creates a net with default parameters.
	public RNNLayer()
	{
		hiddenCount = defaultHiddenCount;
		sequenceLength = defaultSequenceLength;
		learningRate = defaultLearningRate;
	}

	// Creates a net with custom parameters
	public RNNLayer(int hiddenCount, int sequenceLength, double learningRate)
	{
		this.hiddenCount = hiddenCount;
		this.sequenceLength = sequenceLength;
		this.learningRate = learningRate;
	}

	// Initialize the net with random weights.
	public void initialize(int vocabSize)
	{
		if (vocabSize < 0) throw new IllegalArgumentException();

		this.inputSize = vocabSize;

		// create weight matrices

		double scale = 0.1;

		Wxh = Math.randn(hiddenCount, vocabSize).mul(scale);
		Whh = Math.randn(hiddenCount, hiddenCount).mul(scale);
		Why = Math.randn(vocabSize, hiddenCount).mul(scale);
		bh = Matrix.zeros(hiddenCount);
		by = Matrix.zeros(vocabSize);

		Wxh = Matrix.zerosLike(gWxh);
		Whh = Matrix.zerosLike(gWhh);
		Why = Matrix.zerosLike(gWhy);
		bh = Matrix.zerosLike(gbh);
		by = Matrix.zerosLike(gby);

		h = Math.randn(hiddenCount);
	}

	// Train: x's -> y's. Returns the cross-entropy loss.
	// x - indices of inputs through time, starting from t=1
	// y - indices of outputs through time, starting from t=1
	public double train(int ix[], int iy[])
	{
		if (ix.length != iy.length || ix.length != sequenceLength)
			throw new IllegalArgumentException();

		{
			int toCheck[][] = {ix, iy};
			for (int arr[] : toCheck)
				for (int index : arr)
					if (index < 0 || index >= inputSize) throw new IllegalArgumentException();
		}

		/* Initialize forward pass */

		Matrix xAt[] = new Matrix[sequenceLength + 1]; // one-hot input vectors through time
		Matrix hAt[] = new Matrix[sequenceLength + 1]; // one-hot hidden state vectors through time
		Matrix pAt[] =
		    new Matrix[sequenceLength + 1]; // one-hot normalized probability vectors through time
		Matrix yAtt = null; // one-hot unnormalized output probability (now)

		hAt[0] = new Matrix(h); // copy the current state


		/* Forward pass */

		double loss = 0.0;

		for (int t = 1; t < sequenceLength + 1; ++t)
		{
			// Create the one-hot vectors
			xAt[t] = Matrix.oneHot(inputSize, ix[t]);
			Matrix targetYAtt = Matrix.oneHot(inputSize, iy[t]);

			// Find the new hidden state
			hAt[t] = Matrix.tanh(Matrix.dot(Wxh, xAt[t]).add(Matrix.dot(Whh, hAt[t - 1])).add(bh));

			// Find unnormalized output probabilities.
			yAtt = Matrix.dot(Why, hAt[t]).add(by);

			// Normalize output probabilities
			pAt[t] = Matrix.softmax(yAtt);

			// Calulate cross-entropy loss
			loss += -java.lang.Math.log(pAt[t].at(targetYAtt));
		}

		/* Initialize backward pass */

		Matrix dWxh = Matrix.zerosLike(Wxh);
		Matrix dWhh = Matrix.zerosLike(Whh);
		Matrix dWhy = Matrix.zerosLike(Why);
		Matrix dbh = Matrix.zerosLike(bh);
		Matrix dby = Matrix.zerosLike(by);

		Matrix dhNext = Matrix.zerosLike(h);

		/* Backward pass */

		for (int t = sequenceLength; t >= 1; --t)
		{
			// backprop into y, http://cs231n.github.io/neural-networks-case-study/#grad
			Matrix dy = new Matrix(pAt[t]);
			dy.setAt(iy[t], (dy.at(iy[t]) - 1));

			// y updates
			dWhy.add(Matrix.dot(dy, hAt[t].T()));
			dby.add(dy);

			// backprop into h and through tanh nonlinearity
			Matrix dh = Matrix.dot(Why.T(), dy).add(dhNext);
			Matrix dhRaw =
			    Matrix.onesLike(hAt[t]).add(new Matrix(hAt[t]).mul(hAt[t]).neg()).mul(dh);

			// h updates
			dWxh.add(Matrix.dot(dhRaw, xAt[t].T()));
			dWhh.add(Matrix.dot(dhRaw, hAt[t - 1].T()));
			dbh.add(dhRaw);

			// save dhNext for the next iteration
			dhNext = Matrix.dot(Whh.T(), dhRaw);
		}

		// clip exploding gradients

		double clip_a = -5.0;
		Matrix dparams[] = {dWxh, dWhh, dWhy, dbh, dby};

		for (Matrix m : dparams) m.clip(clip_a, -clip_a);


		/* Update weights with Adagrad */

		Matrix params[] = {Wxh, Whh, Why, bh, by};
		Matrix gparams[] = {gWxh, gWhh, gWhy, gbh, gby};

		for (int i = 0; i < dparams.length; ++i)
		{
			gparams[i].add(new Matrix(dparams[i]).mul(dparams[i]));

			Matrix tmp = new Matrix(
			    gparams[i]).apply((gparam) -> 1.0 / (java.lang.Math.sqrt(gparam) + 1e-8));
			params[i].add(new Matrix(dparams[i]).mul(-learningRate).mul(tmp));
		}

		/* Update hidden state */

		h = hAt[ix.length - 1];

		return loss;
	}

	// Samples n indices. Doesn't advance the hidden state.
	public int[] sample(int n, int seedIndex)
	{
		return sample(n, seedIndex, false);
	}

	// Samples n indices.
	// seedIndex is the previous index in the sequence.
	// If keep = true, advances the hidden state.
	public int[] sample(int n, int seedIndex, boolean keep)
	{
		if (n < 1) throw new IllegalArgumentException();

		int indices[] = new int[n];

		Matrix h;
		if (keep)
			h = this.h;
		else
			h = new Matrix(this.h);

		Matrix xAtt = Matrix.oneHot(inputSize, seedIndex);

		for (int i = 0; i < n; ++i)
		{
			h = Matrix.tanh(Matrix.dot(Wxh, xAtt).add(Matrix.dot(Whh, h)).add(bh));
			Matrix y = Matrix.dot(Why, h).add(by);
			Matrix p = Matrix.softmax(y);
		}
	}



	public int getSequenceLength()
	{
		return sequenceLength;
	}
}
