package io.github.garstka.rnn.net;

// RNN that uses integer indices as inputs and outputs.
public abstract class BasicRNN extends RNN
{
	// Initializes the net for this vocabulary size.
	// Requires vocabularySize > 0.
	public abstract void initialize(int vocabularySize);
}
