package io.github.garstka.rnn.net;

import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;
import io.github.garstka.rnn.net.exceptions.NoMoreTrainingDataException;
import io.github.garstka.rnn.net.interfaces.TrainingSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

// Immutable training set for a character level RNN.
public class StringTrainingSet implements TrainingSet
{
	private String data; // Data from file.
	private Alphabet alphabet; // Alphabet extracted from data.

	// Constructs from data. Treats null as an empty string.
	private StringTrainingSet(String data)
	{
		if (data == null)
			data = "";

		this.data = data;
		this.alphabet = Alphabet.fromString(data);
	}

	/* Create */

	// Returns a training set with data from file (UTF-8).
	// Requires fileName != null.
	public static StringTrainingSet fromFile(String fileName) throws IOException
	{
		if (fileName == null)
			throw new NullPointerException("File path can't be null.");

		String data =
		    Files.newBufferedReader(Paths.get(fileName), StandardCharsets.UTF_8)
		        .lines()
		        .collect(Collectors.joining("\n"));
		return new StringTrainingSet(data);
	}

	// Returns a training set created from a string.
	public static StringTrainingSet fromString(String data)
	{
		return new StringTrainingSet(data);
	}

	/* Main functionality */

	// Extracts out.length indices starting at index.
	// ix - input sequence
	// iy - expected output sequence (shifted by 1)
	public void extract(int lowerBound, int[] ix, int[] iy)
	    throws NoMoreTrainingDataException
	{
		try
		{
			if (ix == null || iy == null)
				throw new NullPointerException("Output arrays can't be null.");

			if (ix.length != iy.length)
				throw new IllegalArgumentException(
				    "Arrays must be the same size.");

			if (lowerBound < 0)
				throw new IllegalArgumentException("Illegal lower bound.");

			// fetch one more symbol than the length.
			int upperBound = lowerBound + iy.length + 1;
			if (upperBound >= data.length())
				throw new NoMoreTrainingDataException();

			// prepare the input/output arrays
			int firstCharI;
			int secondCharI = alphabet.charToIndex(data.charAt(lowerBound));
			int t = 0;
			for (int j = lowerBound + t + 1; j < upperBound; j++, t++)
			{
				firstCharI = secondCharI;
				secondCharI = alphabet.charToIndex(data.charAt(j));
				ix[t] = firstCharI;
				iy[t] = secondCharI;
			}
		}
		catch (CharacterNotInAlphabetException e)
		{
			throw new RuntimeException(
			    "Data doesn't match the alphabet."); // shouldn't happen
		}
	}

	/* Getters */

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

	// Returns data size.
	public int size()
	{
		return data.length();
	}

	// Returns the alphabet size.
	public int vocabularySize()
	{
		return alphabet.size();
	}
}
