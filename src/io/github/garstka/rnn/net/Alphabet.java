package io.github.garstka.rnn.net;

import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeSet;

// Immutable set of symbols mapped indices.
public class Alphabet implements Serializable
{
	private char[] indexToChar;
	private HashMap<Character, Integer> charToIndex;

	// Constructs an alphabet containing symbols extracted from the string.
	// Treats null as an empty string.
	private Alphabet(String data)
	{
		if (data == null)
		{
			indexToChar = new char[0];
			charToIndex = new HashMap<Character, Integer>();
			return;
		}

		// find the alphabet
		TreeSet<Character> chars = new TreeSet<Character>();
		charToIndex = new HashMap<Character, Integer>();

		for (int i = 0; i < data.length(); i++)
			chars.add(data.charAt(i));

		indexToChar = new char[chars.size()];

		int i = 0;
		for (Character c : chars)
		{
			indexToChar[i] = c;
			charToIndex.put(c, i++);
		}
	}

	// Returns alphabet containing symbols extracted from the string.
	// Treats null as an empty string.
	public static Alphabet fromString(String data)
	{
		return new Alphabet(data);
	}

	// Returns the alphabet size.
	public int size()
	{
		return indexToChar.length;
	}

	// Converts a character to the corresponding index.
	public int charToIndex(char c) throws CharacterNotInAlphabetException
	{
		Integer index = charToIndex.get(c);
		if (index == null)
			throw new CharacterNotInAlphabetException(
			    "Character is not a part of the alphabet.");

		return index;
	}

	// Converts an index to the corresponding character.
	// Index must be an index returned by charToIndex.
	public char indexToChar(int index)
	{
		if (!(index >= 0 && index < size()))
			throw new IndexOutOfBoundsException(
			    "Index does not correspond to a character.");
		return indexToChar[index];
	}

	// Converts all indices to chars using indexToChar.
	public char[] indicesToChars(int[] indices)
	{
		if (indices == null)
			throw new NullPointerException("Indices can't be null.");

		char[] out = new char[indices.length];

		for (int i = 0; i < indices.length; i++)
			out[i] = indexToChar(indices[i]);

		return out;
	}

	// Converts the string to indices using charToIndex.
	public int[] charsToIndices(String chars)
	    throws CharacterNotInAlphabetException
	{
		if (chars == null)
			throw new NullPointerException("Array can't be null.");

		int[] out = new int[chars.length()];

		for (int i = 0; i < chars.length(); i++)
			out[i] = charToIndex(chars.charAt(i));

		return out;
	}
}
