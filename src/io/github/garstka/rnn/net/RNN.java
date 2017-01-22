package io.github.garstka.rnn.net;

import io.github.garstka.rnn.net.interfaces.IntegerSampleable;
import io.github.garstka.rnn.net.interfaces.Trainable;

import java.io.Serializable;

// A recurrent neural network.
public abstract class RNN implements IntegerSampleable, Trainable, Serializable
{
	/*** Get ***/

	// Returns true if the net was initialized.
	public abstract boolean isInitialized();

	// Returns the vocabulary size (max index + 1), if initialized.
	public abstract int getVocabularySize();
}
