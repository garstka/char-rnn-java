package io.github.garstka.rnn.net;

import io.github.garstka.rnn.Alphabet;
import io.github.garstka.rnn.net.CharacterSampleable;
import io.github.garstka.rnn.net.SingleLayerRNN;
import io.github.garstka.rnn.CharacterNotInAlphabetException;

// Single layer character level RNN
public class SingleLayerCharLevelRNN
    extends SingleLayerRNN implements CharacterSampleable
{
	// Defaults

	public static final int defaultSequenceLength = 50;

	// Creates a net with default parameters.
	public SingleLayerCharLevelRNN()
	{
		super(defaultSequenceLength);
	}

	// Creates a net with custom parameters.
	public SingleLayerCharLevelRNN(
	    int hiddenCount, int sequenceLength, double learningRate)
	{
		super(hiddenCount, sequenceLength, learningRate);
	}

	// Initializes the net for training. Requires trainingSet != null.
	public void initialize(StringTrainingSet trainingSet)
	    throws BadTrainingSetException
	{
		super.initialize(trainingSet);
	}

	// Samples length indices, single seed, advance the state.
	// Throws, if seed is not part of the alphabet.
	public String sampleString(int length, char seed)
	    throws CharacterNotInAlphabetException
	{
		return sampleString(length, seed, true);
	}

	// Samples length indices, single seed, choose whether to advance the state.
	// Throws, if seed is not part of the alphabet.
	public String sampleString(int length, char seed, boolean advance)
	    throws CharacterNotInAlphabetException
	{
		if (!initialized)
			throw new IllegalStateException("Network is uninitialized.");

		if (length < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		Alphabet alphabet = ((StringTrainingSet) trainingSet).getAlphabet();
		int seedIndex = alphabet.charToIndex(seed);


		int[] sampledIndices = this.sampleIndices(length, seedIndex, advance);
		char[] sampledChars = alphabet.indicesToChars(sampledIndices);

		return new String(sampledChars);
	}

	// Samples length indices, string seed, advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	public String sampleString(int length, String seed)
	    throws CharacterNotInAlphabetException
	{
		return sampleString(length, seed, true);
	}

	// Samples length indices, string seed, choose whether to advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	public String sampleString(int length, String seed, boolean advance)
	    throws CharacterNotInAlphabetException
	{
		if(!initialized)
			throw new IllegalStateException("Network uninitialized.");

		if (length < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		if (seed == null)
			throw new NullPointerException("Non-null seed expected.");

		Alphabet alphabet = ((StringTrainingSet) trainingSet).getAlphabet();
		int[] seedIndices = alphabet.charsToIndices(seed);

		int[] sampledIndices = this.sampleIndices(length, seedIndices, advance);
		char[] sampledChars = alphabet.indicesToChars(sampledIndices);

		return new String(sampledChars);
	}

}
