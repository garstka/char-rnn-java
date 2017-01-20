package io.github.garstka.rnn.net.interfaces;

import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;

public interface CharacterSampleable {

	// Samples length characters, single seed, advance the state.
	// Throws, if seed is not part of the alphabet.
	String sampleString(int length, char seed)
	    throws CharacterNotInAlphabetException;

	// Samples length characters, single seed, choose whether to advance the
	// state. Throws, if seed is not part of the alphabet.
	String sampleString(int length, char seed, boolean advance)
	    throws CharacterNotInAlphabetException;

	// Samples n characters, sequence seed, advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	String sampleString(int length, String seed)
	    throws CharacterNotInAlphabetException;

	// Samples n characters, sequence seed, choose whether to advance the state.
	// Throws, if any character in seed is not part of the alphabet.
	String sampleString(int length, String seed, boolean advance)
	    throws CharacterNotInAlphabetException;
}
