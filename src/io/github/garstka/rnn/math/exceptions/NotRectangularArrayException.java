package io.github.garstka.rnn.math.exceptions;

public class NotRectangularArrayException extends Exception
{
	public NotRectangularArrayException()
	{
	}

	public NotRectangularArrayException(String message)
	{
		super(message);
	}

	public NotRectangularArrayException(Throwable cause)
	{
		super(cause);
	}

	public NotRectangularArrayException(String message, Throwable cause)
	{
		super(message, cause);
	}
}
