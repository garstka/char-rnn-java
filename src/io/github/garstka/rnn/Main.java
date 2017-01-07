package io.github.garstka.rnn;

import io.github.garstka.rnn.net.BadTrainingSetException;
import io.github.garstka.rnn.net.NoMoreTrainingDataException;
import io.github.garstka.rnn.net.SingleLayerNet;

import java.io.FileNotFoundException;
import java.io.IOException;

public class Main
{

	public static void main(String[] args)
	{
		// Load the training set.

		String inputFile = "input.txt";
		TrainingSet trainingSet;
		try
		{
			trainingSet = TrainingSet.fromFile(inputFile);
		}
		catch (IOException e)
		{
			System.out.println("Couldn't open the file.");
			return;
		}

		System.out.println("Data size: " + trainingSet.getData().length()
		    + ", vocabulary size: " + trainingSet.getAlphabet().size());

		// Initialize the network.

		SingleLayerNet net = new SingleLayerNet();

		try
		{
			net.initialize(trainingSet);
		}
		catch (BadTrainingSetException e)
		{
			System.out.println("Bad training set.");
			return;
		}

		try
		{
			char seedChar = trainingSet.getData().charAt(0);
			while (true)
			{
				net.train(100, true);
				System.out.println(net.sample(200, seedChar));
			}
		}
		catch (CharacterNotInAlphabetException e)
		{
			// Shouldn't happen.
		}
		catch (NoMoreTrainingDataException e)
		{
			System.out.println("Out of training data.");
		}
	}
}
