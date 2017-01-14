package io.github.garstka.rnn.net;

import io.github.garstka.rnn.math.Matrix;
import io.github.garstka.rnn.math.Random;

// Single layer RNN.
public class SingleLayerRNN extends RNN
{
	// The layer

	protected RNNLayer layer; // The single RNN layer

	// Training

	protected int
	    sequenceLength; // Steps to unroll the RNN for during training.

	protected TrainingSet trainingSet; // The training set
	protected int dataTrainedIndex; // Current index into training data.

	protected int[] ix; // Training sequence inputs by time.
	protected int[] iy; // Training sequence inputs by time.
	protected double smoothLoss; // Loss for training evaluation.
	protected int totalSteps; // Total step count.

	protected boolean debugMessagesOn;
	protected boolean initialized;

	// Creates a net with default parameters.
	public SingleLayerRNN(int sequenceLength)
	{
		layer = new RNNLayer();
		this.sequenceLength = sequenceLength;
	}

	// Creates a net with custom parameters.
	public SingleLayerRNN(
	    int hiddenCount, int sequenceLength, double learningRate)
	{
		if (hiddenCount <= 0)
			throw new IllegalArgumentException("Illegal hidden layer size.");
		if (sequenceLength <= 1)
			throw new IllegalArgumentException("Illegal sequence length.");

		this.sequenceLength = sequenceLength;
		layer = new RNNLayer();
		layer.setHiddenSize(hiddenCount);
		layer.setLearningRate(learningRate);
	}

	// Initializes the net for training. Requires trainingSet != null.
	public void initialize(TrainingSet trainingSet)
	    throws BadTrainingSetException
	{
		if (trainingSet == null)
			throw new NullPointerException("Training set shouldn't be null.");

		this.trainingSet = trainingSet;

		int vocabSize = trainingSet.vocabularySize();

		if (vocabSize == 0)
			throw new BadTrainingSetException("Vocabulary can't be empty.");

		if (trainingSet.size() < sequenceLength)
			throw new BadTrainingSetException(
			    "Data is too small for even a single pass.");

		layer.setInputSize(vocabSize);
		layer.setOutputSize(vocabSize);
		layer.initialize();

		// get the temporary index arrays for sequences

		ix = new int[sequenceLength];
		iy = new int[sequenceLength];

		// initialize smooth loss
		smoothLoss = -Math.log(1.0 / vocabSize) * sequenceLength;

		initialized = true;
	}

	// Trains the network until there's no more data.
	public void train() throws NoMoreTrainingDataException
	{
		train(Integer.MAX_VALUE);
	}

	// Trains the net for a few steps. Requires steps >= 0.
	public void train(int steps) throws NoMoreTrainingDataException
	{
		if (!initialized)
			throw new IllegalStateException("Network is uninitialized.");

		if (steps < 0)
			throw new IllegalArgumentException("Non-negative int expected.");

		for (int i = 0; i < steps; i++)
		{
			// try to extract
			trainingSet.extract(dataTrainedIndex, ix, iy);

			// train

			layer.forward(layer.ixTox(ix));
			double loss = layer.getLoss(iy);
			layer.backward(layer.getdy(iy));

			// calculate smooth loss

			smoothLoss = smoothLoss * 0.999 + loss * 0.001;

			dataTrainedIndex += sequenceLength; // shift the training offset

			totalSteps++;
		}

		// print debug
		if (debugMessagesOn)
			System.out.println(
			    "Step: " + totalSteps + ", loss: " + getSmoothLoss());
	}

	// Returns the cross-entropy loss.
	public double getSmoothLoss()
	{
		return smoothLoss / sequenceLength * 100;
	}

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

	// Print debug messages.
	public void printDebug(boolean on)
	{
		debugMessagesOn = on;
	}
}
