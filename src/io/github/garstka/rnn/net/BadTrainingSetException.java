package io.github.garstka.rnn.net;

// Exception thrown when the training set is too small.
public class BadTrainingSetException extends Exception
{
	public BadTrainingSetException()
	{
	}

	public BadTrainingSetException(String message)
	{
		super(message);
	}

	public BadTrainingSetException(Throwable cause)
	{
		super(cause);
	}

	public BadTrainingSetException(String message, Throwable cause)
	{
		super(message, cause);
	}
}
