package io.github.garstka.rnn.net.interfaces;

// Network that can be sampled for a sequence of integers.
public interface IntegerSampleable {

	// Samples n indices, advances the state.
	// Seed must be at least one index.
	// temp is the must be in (0.0,1.0]. Lower temp means more conservative
	// predictions.
	int[] sampleIndices(int n, int[] seed, double temp);

	// Samples n indices, choose whether to advance the state.
	// Seed must be at least one index.
	// temp is the must be in (0.0,1.0]. Lower temp means more conservative
	// predictions.
	int[] sampleIndices(int n, int[] seed, double temp, boolean advance);
}