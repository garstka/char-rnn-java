package io.github.garstka.rnn;

import java.util.HashMap;
import java.util.TreeSet;

// Immutable set of symbols mapped indices.
public class Alphabet
{
	private char[] index_to_char;
	private HashMap<Character, Integer> char_to_index;

	// Constructs an alphabet containing symbols extracted from the string.
	// Treats null as an empty string.
	private Alphabet(String data)
	{
		if (data == null)
		{
			index_to_char = new char[0];
			char_to_index = new HashMap<Character, Integer>();
			return;
		}

		// find the alphabet
		TreeSet<Character> chars = new TreeSet<Character>();
		char_to_index = new HashMap<Character, Integer>();

		for (int i = 0; i < data.length(); i++)
			chars.add(data.charAt(i));

		index_to_char = new char[chars.size()];

		int i = 0;
		for (Character c : chars)
		{
			index_to_char[i] = c;
			char_to_index.put(c, i);
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
		return index_to_char.length;
	}

	// Converts a character to the corresponding index.
	public int charToIndex(char c) throws CharacterNotInAlphabetException
	{
		Integer index = char_to_index.get(c);
		if (index == null)
			throw new CharacterNotInAlphabetException(
			    "Character is not a part of the alphabet.");

		return index.intValue();
	}

	// Converts an index to the corresponding character.
	// Index must be an index returned by charToIndex.
	public char indexToChar(int index)
	{
		if (!(index >= 0 && index < size()))
			throw new IndexOutOfBoundsException(
			    "Index does not correspond to a character.");
		return index_to_char[index];
	}
}
