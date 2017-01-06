package io.github.garstka.rnn.math;

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
