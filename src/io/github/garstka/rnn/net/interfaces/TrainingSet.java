package io.github.garstka.rnn.net.interfaces;


import io.github.garstka.rnn.net.exceptions.NoMoreTrainingDataException;

// Training set for sequences.
public interface TrainingSet {
	// Extracts out.length indices starting at index.
	// ix - input sequence
	// iy - expected output sequence (shifted by 1)
	void extract(int lowerBound, int[] ix, int[] iy)
	    throws NoMoreTrainingDataException;

	// Returns the data size.
	int size();

	// Returns the max index + 1.
	int vocabularySize();
}
