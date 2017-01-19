package io.github.garstka.rnn.net;

// A recurrent neural network.
public abstract class RNN implements IntegerSampleable, Trainable
{
    // Initializes the net for this vocabulary size. Requires vocabularySize > 0.
    abstract public void initialize(int vocabularySize);
}
