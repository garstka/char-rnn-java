package io.github.garstka.rnn.net;

import io.github.garstka.rnn.math.Matrix;
import io.github.garstka.rnn.math.Random;

// Multi layer RNN.
public class MultiLayerRNN extends RNN
{
	// Layers

	protected RNNLayer[] layer; // All RNN layers

	// Init data

	private int[] hiddenSize;
	private double learningRate;

	protected boolean initialized;

	/*** Construct ***/

	// Creates a net with default parameters.
	public MultiLayerRNN()
	{
		learningRate = RNNLayer.defaultLearningRate;
	}

	// Creates a net with default parameters and initializes immediately.
	public MultiLayerRNN(int vocabularySize)
	{
		this();
		initialize(vocabularySize);
	}

	/*** Hyperparameters ***/

	/*
	    Sets the hidden layer sizes per RNN layer

	    hiddenSize.length > 0
	    each size > 1

	    Network must be initialized again.
	*/
	public void setHiddenSize(int[] hiddenSize)
	{
		if (hiddenSize.length == 0)
			throw new IllegalArgumentException("At least one layer required.");

		for (int size : hiddenSize)
			if (size <= 0)
				throw new IllegalArgumentException(
				    "Hidden size can't be less than 1.");

		this.hiddenSize = hiddenSize;
		initialized = false;
	}

	// Sets the learning rate for each layer.
	public void setLearningRate(double learningRate)
	{
		if (layer == null)
			this.learningRate = learningRate;
		else
			for (RNNLayer layer : layer)
				layer.setLearningRate(learningRate);
	}

	/*** Initialize ***/

	// Initializes the net for this vocabulary size.
	// Requires vocabularySize > 0.
	public void initialize(int vocabularySize)
	{
		if (vocabularySize < 1)
			throw new IllegalArgumentException("Vocabulary size must be > 0.");

		// Create layers

		if (hiddenSize == null) // default: single layer
		{
			hiddenSize = new int[1];
			hiddenSize[0] = RNNLayer.defaultHiddenSize;
		}

		layer = new RNNLayer[hiddenSize.length];

		for (int i = 0; i < layer.length; i++)
		{
			layer[i] = new RNNLayer();

			if (i == 0)
				layer[i].setInputSize(vocabularySize);
			else
				layer[i].setInputSize(hiddenSize[i - 1]);

			layer[i].setHiddenSize(hiddenSize[i]);
			layer[i].setLearningRate(learningRate);


			if (i == layer.length - 1)
				layer[i].setOutputSize(vocabularySize);
			else
				layer[i].setOutputSize(hiddenSize[i + 1]);

			layer[i].initialize();
		}

		initialized = true;
	}

	/*** Train ***/

	/*
	    Performs a forward-backward pass for the given indices.

	    ix.length and iy.length lengths must match, can't be empty.
	    All indices must be less than the vocabulary size.

	    Returns the cross-entropy loss.
	*/
	public double forwardBackward(int[] ix, int[] iy)
	{
		if (!initialized)
			throw new IllegalStateException("Network is uninitialized.");

		if (ix == null || iy == null)
			throw new NullPointerException("Arrays can't be null.");

		if (ix.length != iy.length)
			throw new IllegalArgumentException(
			    "Inputs and outputs must match.");

		if (ix.length == 0)
			throw new IllegalArgumentException(
			    "Can't perform a pass on an empty sequence.");

		// forward pass
		layer[0].forward(layer[0].ixTox(ix));
		for (int i = 1; i < layer.length; i++)
			layer[i].forward(layer[i - 1].gety());

		// calculate loss and get dy
		double loss = layer[layer.length - 1].getLoss(iy);

		// last layer backward pass
		layer[layer.length - 1].backward(layer[layer.length - 1].getdy(iy));

		// rest of the backward pass
		for (int i = layer.length - 2; i >= 0; i--)
			layer[i].backward(layer[i + 1].getdx());

		return loss;
	}

	/*** Sample ***/

	// Samples n indices, single seed, advance the state.
	public int[] sampleIndices(int n, int seed)
	{
		return sampleIndices(n, seed, true);
	}

	// Samples n indices, single seed, choose whether to advance the state.
	public int[] sampleIndices(int n, int seed, boolean advance)
	{
		return sampleIndices(n, new int[] {seed}, advance);
	}

	// Samples n indices, sequence seed, advance the state.
	public int[] sampleIndices(int n, int[] seed)
	{
		return sampleIndices(n, seed, true);
	}

	// Samples n indices, sequence seed, choose whether to advance the state.
	public int[] sampleIndices(int n, int[] seed, boolean advance)
	{
		if (!initialized)
			throw new IllegalStateException("Network is uninitialized.");

		if (n < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		if (seed == null)
			throw new NullPointerException("Non-null seed expected.");

		if (n < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		Matrix[] savedState = null;

		if (!advance)
		{
			savedState = new Matrix[layer.length];
			for (int i = 0; i < layer.length; i++)
				savedState[i] = layer[i].saveHiddenState();
		}

		int[] sampled = new int[n];

		// Seed forward pass.
		layer[0].forward(layer[0].ixTox(seed));
		for (int i = 1; i < layer.length; i++)
			layer[i].forward(layer[i - 1].gety());

		sampled[0] = Random.randomChoice(layer[layer.length
		    - 1].getp()[1].unravel()); // first choice given seed

		// Sample.

		Matrix seedVec = layer[0].ixTox(sampled[0]);
		for (int t = 1; t < n; t++)
		{
			layer[0].forward(seedVec);
			for (int i = 1; i < layer.length; i++)
				layer[i].forward(layer[i - 1].gety());

			sampled[t] = Random.randomChoice(
			    layer[layer.length - 1].getp()[1].unravel());
			seedVec = layer[0].ixTox(sampled[t]);
		}

		if (!advance)
		{
			for (int i = 0; i < layer.length; i++)
				layer[i].restoreHiddenState(savedState[i]);
		}

		return sampled;
	}

	// Returns true if the net was initialized.
	public boolean isInitialized()
	{
		return initialized;
	}

	// Returns the vocabulary size - max index + 1.
	public int getVocabularySize()
	{
		if (!initialized)
			throw new IllegalStateException("Network is uninitialized.");
		return layer[0].getInputSize();
	}
}
