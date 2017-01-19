package io.github.garstka.rnn.net;

// Trainable neural network.
public interface Trainable {

	/*
		Performs a forward-backward pass for the given indices.

		ix.length and iy.length lengths must match.
		All indices must be less than the vocabulary size.

		Returns the cross-entropy loss.
	*/
	double forwardBackward(int[] ix, int[] iy);
}
