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

	public String sampleString(int length, String seed, double temp)
	    throws CharacterNotInAlphabetException
	{
		return sampleString(length, seed, temp, true);
	}

	public String sampleString(int length, String seed, double temp,
	    boolean advance) throws CharacterNotInAlphabetException
	{
		if (!isInitialized())
			throw new IllegalStateException("Network uninitialized.");

		if (length < 0)
			throw new IllegalArgumentException("Non-negative length expected.");

		if (seed == null)
			throw new NullPointerException("Non-null seed expected.");

		if (seed.length() == 0)
			throw new NullPointerException("Non-empty seed expected.");

		int[] seedIndices = getAlphabet().charsToIndices(seed);

		int[] sampledIndices =
		    sampleIndices(length, seedIndices, temp, advance);

		char[] sampledChars = getAlphabet().indicesToChars(sampledIndices);

		return new String(sampledChars);
	}
}
