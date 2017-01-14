package io.github.garstka.rnn.net;

import io.github.garstka.rnn.CharacterNotInAlphabetException;

public interface CharacterSampleable {

	// Samples n characters, single seed, advance the state.
	// Throws, if seed is not part of the alphabet.
	String sampleString(int n, char seed)
	    throws CharacterNotInAlphabetException;

	// Samples n characters, single seed, choose whether to advance the state.
	// Throws, if seed is not part of the alphabet.
	String sampleString(int n, char seed, boolean advance)
	    throws CharacterNotInAlphabetException;

	// Samples n characters, sequence seed, advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	String sampleString(int n, String seed)
	    throws CharacterNotInAlphabetException;

	// Samples n characters, sequence seed, choose whether to advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	String sampleString(int n, String seed, boolean advance)
	    throws CharacterNotInAlphabetException;
}
