package io.github.garstka.rnn.net;

import io.github.garstka.rnn.TrainingSet;

/*
 Single layer RNN.
 */
public class SingleLayerNet
{
	// RNN layer

	private RNNLayer layer;

	// State

	private boolean initialized;

	private TrainingSet trainingSet;

	// Creates a net with default parameters.
	public SingleLayerNet()
	{
		layer = new RNNLayer();
	}

	// Creates a net with custom parameters
	public SingleLayerNet(int hiddenCount, int sequenceLength, double learningRate)
	{
		layer = new RNNLayer(hiddenCount, sequenceLength, learningRate);
	}

	public void initialize(TrainingSet trainingSet) throws BadTrainingSetException
	{
		int vocabSize = trainingSet.vocabSize();
		if (vocabSize == 0 || trainingSet.dataSize() < layer.getSequenceLength())
			throw new BadTrainingSetException();

		layer.initialize(vocabSize);
		initialized = true;
	}

	public void train(int steps)
	{
	}
}
