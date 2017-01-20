package io.github.garstka.rnn.math.exceptions;

public class NotAVectorException extends RuntimeException
{
	public NotAVectorException()
	{
	}

	public NotAVectorException(String message)
	{
		super(message);
	}

	public NotAVectorException(Throwable cause)
	{
		super(cause);
	}

	public NotAVectorException(String message, Throwable cause)
	{
		super(message, cause);
	}
}
