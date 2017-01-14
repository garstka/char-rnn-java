package io.github.garstka.rnn.net;

// Trainable neural network.
public interface Trainable {

	// Trains the network until there's no more data.
	void train() throws NoMoreTrainingDataException;

	// Trains the network for some steps.
	void train(int steps) throws NoMoreTrainingDataException;

	// Returns the smooth cross-entropy loss.
	double getSmoothLoss();
}
