package io.github.garstka.rnn.net;

import io.github.garstka.rnn.math.Matrix;
import io.github.garstka.rnn.math.Random;

// Single layer RNN.
public class SingleLayerRNN extends RNN
{
	// The layer

	protected RNNLayer layer; // The single RNN layer

	protected boolean initialized;

	/*** Construct ***/

	// Creates a net with default parameters.
	public SingleLayerRNN()
	{
		layer = new RNNLayer();
	}

	// Creates a net with default parameters and initializes immediately.
	public SingleLayerRNN(int vocabularySize)
	{
		this();
		initialize(vocabularySize);
	}

	/*** Hyperparameters ***/

	// Sets the hidden layer size. Network must be initialized again.
	public void setHiddenSize(int hiddenSize)
	{
		if (hiddenSize < 1)
			throw new IllegalArgumentException(
			    "Hidden size can't be less than 1.");
		layer.setHiddenSize(hiddenSize);
		initialized = false;
	}

	// Sets the learning rate.
	public void setLearningRate(double learningRate)
	{
		layer.setLearningRate(learningRate);
	}

	/*** Initialize ***/

	// Initializes the net for this vocabulary size. Requires vocabularySize >
	// 0.
	public void initialize(int vocabularySize)
	{
		if (vocabularySize < 1)
			throw new IllegalArgumentException("Vocabulary size must be > 0.");

		// Set the layer parameters.

		layer.setInputSize(vocabularySize);
		layer.setOutputSize(vocabularySize);

		layer.initialize();

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

		layer.forward(layer.ixTox(ix));
		double loss = layer.getLoss(iy);
		layer.backward(layer.getdy(iy));

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

		Matrix savedState = !advance ? layer.saveHiddenState() : null;

		int[] sampled = new int[n];

		layer.forward(layer.ixTox(seed));
		sampled[0] = Random.randomChoice(
		    layer.getp()[1].unravel()); // first choice given seed

		Matrix seedVec = layer.ixTox(sampled[0]);
		for (int t = 1; t < n; t++)
		{
			layer.forward(seedVec);
			sampled[t] = Random.randomChoice(layer.getp()[1].unravel());
			seedVec = layer.ixTox(sampled[t]);
		}

		if (!advance)
			layer.restoreHiddenState(savedState);

		// int[] sampled = layer.sample(n,seed[0]);

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
		return layer.getInputSize();
	}
}
