package io.github.garstka.rnn.net.exceptions;

public class CharacterNotInAlphabetException extends Exception
{
	public CharacterNotInAlphabetException()
	{
	}

	public CharacterNotInAlphabetException(String message)
	{
		super(message);
	}

	public CharacterNotInAlphabetException(Throwable cause)
	{
		super(cause);
	}

	public CharacterNotInAlphabetException(String message, Throwable cause)
	{
		super(message, cause);
	}
}
