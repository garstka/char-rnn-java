package io.github.garstka.rnn.net;

// Single layer character level RNN.
public class MultiLayerCharLevelRNN extends CharLevelRNN
{
	protected Alphabet alphabet; // The alphabet for sampling.

	protected MultiLayerRNN internal; // Basic network.

	/*** Construct ***/

	// Constructs without initialization.
	public MultiLayerCharLevelRNN()
	{
		internal = new MultiLayerRNN();
	}

	// Constructs and initializes immediately.
	// Requires that alphabet != null.
	public MultiLayerCharLevelRNN(Alphabet alphabet)
	{
		this();
		initialize(alphabet);
	}

	/*** Hyperparameters ***/

	// Sets the hidden layer size. Network must be initialized again.
	public void setHiddenSize(int[] hiddenSize)
	{
		internal.setHiddenSize(hiddenSize);
	}

	// Sets the learning rate.
	public void setLearningRate(double learningRate)
	{
		internal.setLearningRate(learningRate);
	}

	/*** Initialize ***/

	// Initializes the net. alphabet != null.
	public void initialize(Alphabet alphabet)
	{
		if (alphabet == null)
			throw new NullPointerException("Alphabet can't be null.");

		this.alphabet = alphabet;
		internal.initialize(alphabet.size());
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
		return internal.forwardBackward(ix, iy);
	}

	/*** Sample ***/

	// Samples n indices, sequence seed, advance the state.
	public int[] sampleIndices(int n, int[] seed, double temp)
	{
		return internal.sampleIndices(n, seed, temp);
	}

	// Samples n indices, sequence seed, choose whether to advance the state.
	public int[] sampleIndices(int n, int[] seed, double temp, boolean advance)
	{
		return internal.sampleIndices(n, seed, temp, advance);
	}

	/*** Get ***/

	// Returns the alphabet, if initialized.
	public Alphabet getAlphabet()
	{
		return alphabet;
	}

	// Returns true if the net was initialized.
	public boolean isInitialized()
	{
		return internal.isInitialized();
	}

	// Returns the vocabulary size (the alphabet size), if initialized.
	public int getVocabularySize()
	{
		return internal.getVocabularySize();
	}
}
