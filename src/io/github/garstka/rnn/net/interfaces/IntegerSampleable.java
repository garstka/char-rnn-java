package io.github.garstka.rnn.net.interfaces;

// Network that can be sampled for a sequence of integers.
public interface IntegerSampleable {

	// Samples n indices, single seed, advance the state.
	int[] sampleIndices(int n, int seed);

	// Samples n indices, single seed, choose whether to advance the state.
	int[] sampleIndices(int n, int seed, boolean advance);

	// Samples n indices, sequence seed, advance the state.
	int[] sampleIndices(int n, int[] seed);

	// Samples n indices, sequence seed, choose whether to advance the state.
	int[] sampleIndices(int n, int[] seed, boolean advance);
}