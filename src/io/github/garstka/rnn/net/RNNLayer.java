package io.github.garstka.rnn.net;

import io.github.garstka.rnn.math.Math;
import io.github.garstka.rnn.math.Matrix;
import io.github.garstka.rnn.math.Random;

// An RNN Layer with support for mult-layer networks.
public class RNNLayer
{
	// Hyperparameters

	private double learningRate; // Backpropagation parameter.

	// Dimensions

	private int inputSize; // input vector size
	private int hiddenSize; // hidden state size
	private int outputSize; // input vector size

	// Defaults

	public static final int defaultInputSize = 50;
	public static final int defaultHiddenSize = 100;
	public static final int defaultOutputSize = 50;
	public static final double defaultLearningRate = 0.1;

	// Network state

	private Matrix Wxh; // input layer weights
	private Matrix Whh; // hidden layer weights
	private Matrix Why; // output layer weights
	private Matrix bh; // hidden bias
	private Matrix by; // output bias

	private Matrix h; // last hidden state

	// Training state

	private Matrix gWxh; // gradient descent params: input layer
	private Matrix gWhh; // gradient descent params: hidden layer
	private Matrix gWhy; // gradient descent params: output layer
	private Matrix gbh; // gradient descent params: hidden bias
	private Matrix gby; // gradient descent params: output bias

	private Matrix[] xAt; // input vectors through time
	private Matrix[] hAt; // hidden state vectors through time
	private Matrix[] yAt; // unnormalized output probability vectors through
	// time
	private Matrix[] pAt; // normalized output probability vectors through time
	private Matrix[] dxAt; // output gradient from a backwards pass

	private int lastSequenceLength; // Number of steps in the last forward pass
	// (must match the steps for the backward pass)

	private boolean initialized;

	/* Init */
	// Creates a net with default parameters.
	public RNNLayer()
	{
		inputSize = defaultInputSize;
		hiddenSize = defaultHiddenSize;
		inputSize = defaultOutputSize;
		learningRate = defaultLearningRate;
	}

	// Creates a net with custom parameters
	public RNNLayer(
	    int inputSize, int hiddenSize, int outputSize, double learningRate)
	{
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;
		this.learningRate = learningRate;
	}

	// Set the input size.
	public void setInputSize(int inputSize)
	{
		if (initialized)
			initialized = false;

		if (inputSize <= 0)
			throw new IllegalArgumentException("Illegal input size.");

		this.inputSize = inputSize;
	}

	// Set the hidden size.
	public void setHiddenSize(int hiddenSize)
	{
		if (initialized)
			initialized = false;

		if (hiddenSize <= 0)
			throw new IllegalArgumentException("Illegal hidden size.");

		this.hiddenSize = hiddenSize;
	}

	// Set the output size.
	public void setOutputSize(int outputSize)
	{
		if (initialized)
			initialized = false;

		if (outputSize <= 0)
			throw new IllegalArgumentException("Illegal output size.");

		this.outputSize = outputSize;
	}

	public void setLearningRate(double learningRate)
	{
		if (initialized)
			initialized = false;

		this.learningRate = learningRate;
	}

	// Initialize the net with random weights.
	public void initialize()
	{
		if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0)
			throw new IllegalArgumentException("Illegal layer sizes.");

		// create weight matrices

		double scale = 0.1;

		Wxh = Random.randn(hiddenSize, inputSize).mul(scale);
		Whh = Random.randn(hiddenSize, hiddenSize).mul(scale);
		Why = Random.randn(outputSize, hiddenSize).mul(scale);
		bh = Matrix.zeros(hiddenSize);
		by = Matrix.zeros(outputSize);

		gWxh = Matrix.zerosLike(Wxh);
		gWhh = Matrix.zerosLike(Whh);
		gWhy = Matrix.zerosLike(Why);
		gbh = Matrix.zerosLike(bh);
		gby = Matrix.zerosLike(by);

		h = Random.randn(hiddenSize);

		initialized = true;
	}

	/*
	    Converts the indices (x's) of a sequence to one-hot vectors
	    for use as the x parameter for the first layer of a network.

	    Requirement: ix can't be null, or empty. ix[i] < inputSize
	*/
	Matrix[] ixTox(int ix[])
	{
		if (ix == null)
			throw new NullPointerException("The sequence can't be null.");

		if (ix.length == 0)
			throw new IllegalArgumentException(
			    "The sequence must be non-empty.");

		{
			for (int index : ix)
				if (index < 0 || index >= inputSize)
					throw new IllegalArgumentException(
					    "Illegal index passed as argument.");
		}

		// start at t = 1
		Matrix[] oneHot = new Matrix[ix.length + 1];
		for (int t = 1; t < ix.length + 1; t++)
			oneHot[t] = Matrix.oneHot(inputSize, ix[t - 1]);

		return oneHot;
	}

	// Like ixTox, but a single index instead of an array
	Matrix ixTox(int ix)
	{
		if (ix < 0 || ix >= inputSize)
			throw new IllegalArgumentException(
			    "Illegal index passed as argument.");

		return Matrix.oneHot(inputSize, ix);
	}

	/*
	    Training forward pass.

	    TRAINING

	    Takes x as input. For the first layer, x[t] should be a one-hot vector,
	    for the others it's the output of the previous layer.

	    Yields y: unnormalized probabilities (inputs to next layers)
	          p: normalized probabilities (for sampling)
	    and dy, if provided with targets y's (input for the backward pass)

	    SAMPLING

	    Same as training, but x should be treated as a seed sequence.

	    Requirement:
	    x came from ixTox or the previous layer's forward pass result y.
	*/
	void forward(Matrix x[])
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		if (x == null || x.length < 2) // starting at t = 1
			throw new IllegalArgumentException(
			    "Expected a sequence of length at least 1.");

		{
			for (int t = 1; t < x.length; ++t)
				if (x[t] == null || x[t].getk() != inputSize)
					throw new IllegalArgumentException(
					    "Bad vector passed as argument.");
		}

		/* Initialize the forward pass */

		xAt = x; // save the inputs (needed for backpropagation)

		// Reset outputs

		lastSequenceLength = x.length - 1;

		// hidden state vectors through time
		hAt = new Matrix[lastSequenceLength + 1];

		// normalized probability vectors through time
		pAt = new Matrix[lastSequenceLength + 1];

		// normalized probability vectors through time
		yAt = new Matrix[lastSequenceLength + 1];

		hAt[0] = new Matrix(h); // copy the current state

		/* Forward pass */

		for (int t = 1; t < lastSequenceLength + 1; ++t)
		{
			// find the new hidden state
			hAt[t] = (Matrix.dot(Wxh, xAt[t])
			              .add(Matrix.dot(Whh, hAt[t - 1]))
			              .add(bh))
			             .tanh();

			// find unnormalized output probabilities
			yAt[t] = Matrix.dot(Why, hAt[t]).add(by);

			// normalize output probabilities
			pAt[t] = Math.softmax(yAt[t]);
		}

		/* Update the hidden state */

		h = hAt[lastSequenceLength];
	}

	// Forward pass for a single seed.
	void forward(Matrix x)
	{
		if (x == null)
			throw new NullPointerException("x is null.");

		Matrix[] xa = new Matrix[2];
		xa[1] = x;
		forward(xa);
	}

	/*
	    Calculates the cross-entropy loss of the last forward pass
	    given target outputs.

	    iy - the target indices

	    Requirements: iy must be the size of the last sequence length, iy[i] <
	   outputSize
	*/
	double getLoss(int iy[])
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		if (iy == null || iy.length != lastSequenceLength)
			throw new IllegalArgumentException(
			    "Expected the iy sequence to be the same length as the last x sequence.");

		{
			for (int index : iy)
				if (index < 0 || index >= outputSize)
					throw new IllegalArgumentException(
					    "Bad index passed as argument.");
		}

		double loss = 0.0;
		for (int t = 1; t < lastSequenceLength + 1; ++t) // start at t = 1
			loss += -java.lang.Math.log(
			    pAt[t].at(iy[t - 1])); // calculate the cross-entropy loss

		return loss;
	}

	// Returns y: the unnormalized probabilities - output of the last forward
	// pass, starting at t = 1.
	Matrix[] gety()
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		return yAt;
	}

	// Returns p: the normalized probabilities - output of the last forward
	// pass, starting at t = 1.
	Matrix[] getp()
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		return pAt;
	}

	/*
	    Returns dy: the gradients to be used as input to the last layer's
	    backward pass, starting at t = 1. given iy - the target indices.

	    Requirement:
	    iy must be the size of the last sequence length, iy[i] < outputSize
	*/
	Matrix[] getdy(int iy[])
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		if (iy == null || iy.length != lastSequenceLength)
			throw new IllegalArgumentException(
			    "Expected the iy sequence to be the same length as the last x sequence.");

		{
			for (int index : iy)
				if (index < 0 || index >= outputSize)
					throw new IllegalArgumentException(
					    "Bad index passed as argument.");
		}

		Matrix[] dyAt = new Matrix[lastSequenceLength + 1]; // start at t = 1

		for (int t = 1; t < lastSequenceLength + 1; ++t)
		{
			// backprop into y,
			// http://cs231n.github.io/neural-networks-case-study/#grad
			dyAt[t] = new Matrix(pAt[t]);

			int expected = iy[t - 1];
			dyAt[t].setAt(expected, (dyAt[t].at(expected) - 1));
		}

		return dyAt;
	}



	/*
	    Training backward pass.

	    Takes dy - the gradient to backpropagate.

	    Yields dx, which can be used as input to previous layer's backward pass,
	    if present, and the updated weights.

	    Requirement: x came from this layer's getdy() or the next layer's
	    backward pass result dx.
	*/
	void backward(Matrix dy[])
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		if (dy == null
		    || dy.length != lastSequenceLength + 1) // starts at t = 1
			throw new IllegalArgumentException(
			    "Expected the y sequence to be the same length as the last x sequence.");

		{
			for (int t = 1; t < lastSequenceLength + 1; ++t)
				if (dy[t] == null || dy[t].getk() != outputSize)
					throw new IllegalArgumentException(
					    "Bad index passed as argument.");
		}

		/* Initialize backward pass */

		Matrix[] dyAt = dy; // just an alias

		Matrix dWxh = Matrix.zerosLike(Wxh);
		Matrix dWhh = Matrix.zerosLike(Whh);
		Matrix dWhy = Matrix.zerosLike(Why);
		Matrix dbh = Matrix.zerosLike(bh);
		Matrix dby = Matrix.zerosLike(by);

		Matrix dhNext = Matrix.zerosLike(h);

		// gradients to be passed to the next backwards pass
		dxAt = new Matrix[lastSequenceLength + 1];

		/* Backward pass */

		for (int t = lastSequenceLength; t >= 1; --t)
		{
			// y updates
			dWhy.add(Matrix.dot(dyAt[t], hAt[t].T()));
			dby.add(dyAt[t]);

			// backprop into h and through tanh nonlinearity
			Matrix dh = Matrix.dot(Why.T(), dyAt[t]).add(dhNext);
			Matrix dhRaw = Matrix.onesLike(hAt[t])
			                   .add(new Matrix(hAt[t]).mul(hAt[t]).neg())
			                   .mul(dh);

			// h updates
			dWxh.add(Matrix.dot(dhRaw, xAt[t].T()));
			dWhh.add(Matrix.dot(dhRaw, hAt[t - 1].T()));
			dbh.add(dhRaw);

			// save dhNext for the next iteration
			dhNext = Matrix.dot(Whh.T(), dhRaw);

			// multi-layer only - save dx
			dxAt[t] = Matrix.dot(Wxh.T(), dhRaw);
		}

		// clip exploding gradients

		double clip_a = -5.0;
		Matrix dparams[] = {dWxh, dWhh, dWhy, dbh, dby};

		for (Matrix m : dparams)
			m.clip(clip_a, -clip_a);

		for (int t = 1; t < lastSequenceLength + 1; ++t)
			dxAt[t].clip(clip_a, -clip_a);


		/* Update weights with Adagrad */

		Matrix params[] = {Wxh, Whh, Why, bh, by};
		Matrix gparams[] = {gWxh, gWhh, gWhy, gbh, gby};

		for (int i = 0; i < dparams.length; ++i)
		{
			Matrix param = params[i];
			Matrix dparam = dparams[i];
			Matrix gparam = gparams[i];

			gparam.add(new Matrix(dparam).mul(dparam));
			Matrix tmp = new Matrix(gparam).apply(
			    (elem) -> java.lang.Math.sqrt(elem) + 1e-8);
			param.add(new Matrix(dparam).mul(-learningRate).div(tmp));
		}
	}

	/*
	    Returns dx: the gradients to be used as input to the previous layer's
	    backward pass.
	*/
	Matrix[] getdx()
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		return dxAt;
	}

	// Samples n indices using only the single layer, and a single seed index.
	// Doesn't advance the hidden state.
	public int[] sample(int n, int seedIndex)
	{
		return sample(n, seedIndex, false);
	}

	// Samples n indices using only the single layer, and a single seed index.
	// seedIndex is the previous index in the sequence.
	// If keep = true, advances the hidden state.
	public int[] sample(int n, int seedIndex, boolean keep)
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		if (n < 1)
			throw new IllegalArgumentException();

		int[] indices = new int[n];

		Matrix h;
		if (keep)
			h = this.h;
		else
			h = new Matrix(this.h);

		Matrix xAtt = Matrix.oneHot(inputSize, seedIndex);

		for (int i = 0; i < n; ++i)
		{
			// calculate the next hidden state
			h = (Matrix.dot(Wxh, xAtt).add(Matrix.dot(Whh, h)).add(bh)).tanh();

			// calculate output
			Matrix y = Matrix.dot(Why, h).add(by);

			// calculate probabilities
			Matrix p = Math.softmax(y);

			// choose one of the outputs based on the probabilities
			int ix = Random.randomChoice(p.unravel());
			indices[i] = ix;

			// use the chosen index as the new input
			xAtt = Matrix.oneHot(inputSize, ix);
		}

		return indices;
	}

	// Save the hidden state before sampling.
	Matrix saveHiddenState()
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		return new Matrix(h);
	}

	// Restore the hidden state after sampling.
	void restoreHiddenState(Matrix h)
	{
		if (!initialized)
			throw new IllegalStateException("Network was not initialized.");

		if (h.getk() != hiddenSize)
			throw new IllegalArgumentException(
			    "The hidden state has the wrong size.");
		this.h = h;
	}


	int getInputSize()
	{
		return inputSize;
	}

	int getHiddenSize()
	{
		return hiddenSize;
	}

	int getOutputSize()
	{
		return outputSize;
	}

	double getLearningRate()
	{
		return learningRate;
	}
}
