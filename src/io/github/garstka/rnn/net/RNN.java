package io.github.garstka.rnn.net;

import io.github.garstka.rnn.net.interfaces.IntegerSampleable;
import io.github.garstka.rnn.net.interfaces.Trainable;

// A recurrent neural network.
public abstract class RNN implements IntegerSampleable, Trainable
{
	/*** Get ***/

	// Returns true if the net was initialized.
	public abstract boolean isInitialized();

	// Returns the vocabulary size (max index + 1), if initialized.
	public abstract int getVocabularySize();
}
