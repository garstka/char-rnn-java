package io.github.garstka.rnn.net.exceptions;

public class NoMoreTrainingDataException extends Exception
{
	public NoMoreTrainingDataException()
	{
	}

	public NoMoreTrainingDataException(String message)
	{
		super(message);
	}

	public NoMoreTrainingDataException(Throwable cause)
	{
		super(cause);
	}

	public NoMoreTrainingDataException(String message, Throwable cause)
	{
		super(message, cause);
	}
}
