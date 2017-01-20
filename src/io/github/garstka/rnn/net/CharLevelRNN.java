package io.github.garstka.rnn.net;

import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;
import io.github.garstka.rnn.net.interfaces.CharacterSampleable;

// RNN that can use both indices, and characters as inputs/outputs.
public abstract class CharLevelRNN extends RNN implements CharacterSampleable
{
	/*** Initialize ***/

	// Initializes the net. Requires that alphabet != null.
	public abstract void initialize(Alphabet alphabet);

	/*** Get ***/

	// Returns the alphabet, if initialized.
	public abstract Alphabet getAlphabet();

	/*** Sample ***/

	// Samples length characters, single seed, advance the state.
	// Throws, if seed is not part of the alphabet.
	public String sampleString(int length, char seed)
	    throws CharacterNotInAlphabetException
	{
		return sampleString(length, seed, true);
	}

	// Samples length characters, single seed, choose whether to advance the
	// state. Throws, if seed is not part of the alphabet.
	public String sampleString(int length, char seed, boolean advance)
	    throws CharacterNotInAlphabetException
	{
		if (!isInitialized())
			throw new IllegalStateException("Network is uninitialized.");

		if (length < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		int seedIndex = getAlphabet().charToIndex(seed);

		int[] sampledIndices = sampleIndices(length, seedIndex, advance);
		char[] sampledChars = getAlphabet().indicesToChars(sampledIndices);

		return new String(sampledChars);
	}

	// Samples n characters, sequence seed, advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	public String sampleString(int length, String seed)
	    throws CharacterNotInAlphabetException
	{
		return sampleString(length, seed, true);
	}

	// Samples n characters, sequence seed, choose whether to advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	public String sampleString(int length, String seed, boolean advance)
	    throws CharacterNotInAlphabetException
	{
		if (!isInitialized())
			throw new IllegalStateException("Network uninitialized.");

		if (length < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		if (seed == null)
			throw new NullPointerException("Non-null seed expected.");

		int[] seedIndices = getAlphabet().charsToIndices(seed);

		int[] sampledIndices = sampleIndices(length, seedIndices, advance);
		char[] sampledChars = getAlphabet().indicesToChars(sampledIndices);

		return new String(sampledChars);
	}
}
