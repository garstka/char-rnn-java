package io.github.garstka.rnn.net.interfaces;

import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;

public interface CharacterSampleable {

	// Samples length characters, advances the state.
	// Seed must be at least one character.
	// temp is the must be in (0.0,1.0]. Lower temp means more conservative
	// predictions.
	// Throws, if any character in seed is not part of the alphabet.
	String sampleString(int length, String seed, double temp)
	    throws CharacterNotInAlphabetException;

	// Samples length characters, choose whether to advance the state.
	// Seed must be at least one character.
	// temp is the must be in (0.0,1.0]. Lower temp means more conservative
	// predictions.
	// Throws, if any character in seed is not part of the alphabet.
	String sampleString(int length, String seed, double temp, boolean advance)
	    throws CharacterNotInAlphabetException;
}
