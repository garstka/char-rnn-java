package pl.edu.agh.po.rnn.net;

import pl.edu.agh.po.rnn.TrainingSet;
import pl.edu.agh.po.rnn.math.BadMatrixException;
import pl.edu.agh.po.rnn.math.Matrix;
import pl.edu.agh.po.rnn.math.Math;

/*
 3 layer neural net
 */
public class SingleLayerNet
{
	// Hyperparameters
	private int hiddenCount; // Size of a single hidden layer of neurons.
	private int sequenceLength; // Number of steps to unroll the RNN for backpropagation.
	private double learningRate; // Backpropagation parameter.

	public static final int defaultHiddenCount = 100;
	public static final int defaultSequenceLength = 25;
	public static final double defaultLearningRate = 0.1;

	// Creates a net with default parameters.
	public SingleLayerNet()
	{
		hiddenCount = defaultHiddenCount;
		sequenceLength = defaultSequenceLength;
		learningRate = defaultLearningRate;
	}

	// Creates a net with custom parameters
	public SingleLayerNet(int hiddenCount, int sequenceLength, double learningRate)
	{
		this.hiddenCount = hiddenCount;
		this.sequenceLength = sequenceLength;
		this.learningRate = learningRate;
	}

	// State

	private boolean initialized;

	private TrainingSet trainingSet;
	private int dataSize;
	private int vocabSize;
	private Matrix Wih; // input layer weights
	private Matrix Whh; // hidden layer weights
	private Matrix Whi; // output layer weights
	private Matrix bh; // hidden bias
	private Matrix by; // output bias

	public void initialize(TrainingSet trainingSet) throws BadTrainingSetException
	{
		dataSize = trainingSet.dataSize();
		vocabSize = trainingSet.vocabSize();
		if (vocabSize == 0 || dataSize < defaultSequenceLength) throw new BadTrainingSetException();

		// create weight matrices

		try
		{

			double scale = 0.1;

			// hidden rows, input size cols
			Wih = Math.randn(hiddenCount, trainingSet.dataSize()).dot(scale);
			Whh = Math.randn(hiddenCount, hiddenCount).dot(scale);
			Whi = Math.randn(trainingSet.dataSize(), hiddenCount).dot(scale);

			bh = Matrix.zeros(hiddenCount);
			by = Matrix.zeros(trainingSet.vocabSize());
		}
		catch (BadMatrixException e)
		{
			throw new RuntimeException("Bad matrix.");
		}

		initialized = true;
	}
}
