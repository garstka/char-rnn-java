package io.github.garstka.rnn.net;

import io.github.garstka.rnn.Alphabet;
import io.github.garstka.rnn.CharacterNotInAlphabetException;
import io.github.garstka.rnn.TrainingSet;

/*
 Single layer RNN.
 */
public class SingleLayerNet
{
	private RNNLayer layer; // RNN layer

	private TrainingSet trainingSet; // The training set

	private int dataTrainedIndex; // Index into training data.
	private int[] ix; // Training sequence inputs by time (starting from t=1).
	private int[] iy; // Training sequence inputs by time (starting from t=1).
	private double smoothLoss; // Loss for training evaluation.
	int totalSteps; // Total step count.

	private boolean initialized;

	// Creates a net with default parameters.
	public SingleLayerNet()
	{
		layer = new RNNLayer();
	}

	// Creates a net with custom parameters.
	public SingleLayerNet(
	    int hiddenCount, int sequenceLength, double learningRate)
	{
		layer = new RNNLayer(hiddenCount, sequenceLength, learningRate);
	}

	// Initializes the net for training. Requires trainingSet != null.
	public void initialize(TrainingSet trainingSet)
	    throws BadTrainingSetException
	{
		if (trainingSet == null)
			throw new NullPointerException("Training set shouldn't be null.");

		this.trainingSet = trainingSet;

		int vocabSize = trainingSet.getAlphabet().size();

		if (vocabSize == 0)
			throw new BadTrainingSetException("Vocabulary can't be empty.");

		if (trainingSet.getData().length() < layer.getSequenceLength())
			throw new BadTrainingSetException(
			    "Data is too small for even a single pass.");

		layer.initialize(vocabSize);

		// get the temporary index arrays for sequences

		ix = new int[layer.getSequenceLength() + 1];
		iy = new int[layer.getSequenceLength() + 1];

		// initialize smooth loss
		smoothLoss = -Math.log(1.0 / vocabSize) * layer.getSequenceLength();

		initialized = true;
	}

	// Trains the net for a few steps. Requires steps >= 0.
	public void train(int steps, boolean printDebug)
	    throws NoMoreTrainingDataException
	{
		if (!initialized)
			return;

		if (steps < 0)
			throw new IllegalArgumentException("Non-negative int expected.");

		try
		{
			String data = trainingSet.getData();
			Alphabet alphabet = trainingSet.getAlphabet();

			for (int i = 0; i < steps; i++)
			{
				// set up the bounds
				int lowerBound = dataTrainedIndex;
				int upperBound = lowerBound + layer.getSequenceLength() + 1;

				if (upperBound > data.length())
					throw new NoMoreTrainingDataException();

				// prepare the input/output arrays (shifted by one)
				int firstCharI;
				int secondCharI = alphabet.charToIndex(data.charAt(lowerBound));
				int t = 1; // start at t = 1.
				for (int j = lowerBound + t; j < upperBound; j++, t++)
				{
					firstCharI = secondCharI;
					secondCharI = alphabet.charToIndex(data.charAt(j));
					ix[t] = firstCharI;
					iy[t] = secondCharI;
				}

				// train
				double loss = layer.train(ix, iy);

				// print debug

				if (printDebug)
				{
					smoothLoss = smoothLoss * 0.999 + loss * 0.001;
					System.out.println(
					    "Step: " + totalSteps + ", loss: " + smoothLoss);
				}
				dataTrainedIndex = upperBound - 1; // shift the trained boundary

				totalSteps++;
			}
		}
		catch (CharacterNotInAlphabetException e)
		{
			// Shouldn't happen.
			throw new RuntimeException("Unexpected state of alphabet.");
		}
	}

	// Samples a string of the given length.
	// Throws, if seedChar is not part of the alphabet.
	// Requires length >= 0.
	public String sample(int length, char seedChar)
	    throws CharacterNotInAlphabetException
	{
		if (length < 0)
			throw new IllegalArgumentException("Non-negative int expected.");

		Alphabet alphabet = trainingSet.getAlphabet();
		int seedIndex = alphabet.charToIndex(seedChar);
		int[] sampledIndices = layer.sample(length, seedIndex);
		char[] sampled = new char[sampledIndices.length];

		for (int i = 0; i < sampledIndices.length; i++)
			sampled[i] = alphabet.indexToChar(sampled[i]);

		return new String(sampled);
	}
}
