package io.github.garstka.rnn;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

// Immutable training set for a recurrent neural net - string and its
// alphabet.
public class TrainingSet
{
	private String data; // Data from file.
	private Alphabet alphabet; // Alphabet extracted from data.

	// Constructs from data. Treats null as an empty string.
	private TrainingSet(String data)
	{
		if (data == null)
			data = new String();

		this.data = data;
		this.alphabet = Alphabet.fromString(data);
	}

	// Returns a training set with data from file (UTF-8). Requires fileName !=
	// null.
	public static TrainingSet fromFile(String fileName) throws IOException
	{
		if (fileName == null)
			throw new NullPointerException("File path can't be null.");

		String data =
		    Files.newBufferedReader(Paths.get(fileName), StandardCharsets.UTF_8)
		        .lines()
		        .collect(Collectors.joining("\n"));
		return new TrainingSet(data);
	}

	// Returns a training set created from a string.
	public static TrainingSet fromString(String data)
	{
		return new TrainingSet(data);
	}

	// Returns the loaded data.
	public String getData()
	{
		return data;
	}

	// Returns the alphabet.
	public Alphabet getAlphabet()
	{
		return alphabet;
	}
}
