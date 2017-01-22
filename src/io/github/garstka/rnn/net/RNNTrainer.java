package io.github.garstka.rnn.net;

import io.github.garstka.rnn.net.exceptions.BadTrainingSetException;
import io.github.garstka.rnn.net.exceptions.NoMoreTrainingDataException;
import io.github.garstka.rnn.net.interfaces.Trainable;
import io.github.garstka.rnn.net.interfaces.TrainingSet;

import java.io.Serializable;

// Trains a recurrent neural net on a training set.
public class RNNTrainer
{
	// Defaults

	public static final int defaultSequenceLength = 50;

	private Trainable net;

	// Training

	protected int
	    sequenceLength; // Steps to unroll the RNN for during training.

	protected TrainingSet trainingSet; // The training set.
	protected int dataTrainedIndex; // Current index into training data.

	protected int[] ix; // Training sequence inputs by time.
	protected int[] iy; // Training sequence inputs by time.
	protected double smoothLoss; // Loss for training evaluation.
	protected int totalSteps; // Total step count.

	protected boolean debugMessagesOn;

	protected boolean initialized;

	/*** Construct ***/

	// Constructs without initializing
	public RNNTrainer()
	{
		this.setSequenceLength(defaultSequenceLength);
	}

	// Constructs and initializes.
	public RNNTrainer(Trainable net, TrainingSet data)
	    throws BadTrainingSetException
	{
		this();
		initialize(net, data);
	}

	/*** Params ***/

	// Set a different sequence length.
	public void setSequenceLength(int length)
	{
		if (length <= 1)
			throw new IllegalArgumentException("Illegal sequence length.");

		this.sequenceLength = length;
		if (initialized)
		{
			ix = new int[sequenceLength];
			iy = new int[sequenceLength];
		}
	}

	/*** Initialize ***/

	// Initializes training. Requires trainingSet != null.
	public void initialize(Trainable net, TrainingSet trainingSet)
	    throws BadTrainingSetException
	{
		if (trainingSet == null)
			throw new NullPointerException("Training set shouldn't be null.");

		if (net == null)
			throw new NullPointerException("Network set shouldn't be null.");

		this.net = net;
		this.trainingSet = trainingSet;

		int vocabSize = trainingSet.vocabularySize();

		if (vocabSize == 0)
			throw new BadTrainingSetException("Vocabulary can't be empty.");

		if (trainingSet.size() < sequenceLength)
			throw new BadTrainingSetException(
			    "Data is too small for even a single pass.");

		// get the temporary index arrays for sequences

		ix = new int[sequenceLength];
		iy = new int[sequenceLength];

		// initialize smooth loss
		smoothLoss = -Math.log(1.0 / vocabSize) * sequenceLength;

		initialized = true;
	}

	/*** Train ***/

	// Trains the network until there's no more data.
	public void train() throws NoMoreTrainingDataException
	{
		train(Integer.MAX_VALUE);
	}

	// Trains the net for a few steps. Requires steps >= 0.
	public void train(int steps) throws NoMoreTrainingDataException
	{
		if (!initialized)
			throw new IllegalStateException("Training is uninitialized.");

		if (steps < 0)
			throw new IllegalArgumentException(
			    "Non-negative step count expected.");

		for (int i = 0; i < steps; i++)
		{
			// try to extract
			trainingSet.extract(dataTrainedIndex, ix, iy);

			// train

			double loss = net.forwardBackward(ix, iy);

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

	// Reset the data pointer to the beginning.
	public void loopAround()
	{
		dataTrainedIndex = 0;
	}

	/*** Get ***/

	// Returns the step count.
	public int getTotalSteps()
	{
		return totalSteps;
	}

	// Returns the smooth cross-entropy loss.
	public double getSmoothLoss()
	{
		if (!initialized)
			throw new IllegalStateException("Training is uninitialized.");

		return smoothLoss / sequenceLength * 100;
	}

	/*** Set ***/

	// Print debug messages.
	public void printDebug(boolean on)
	{
		debugMessagesOn = on;
	}
}
