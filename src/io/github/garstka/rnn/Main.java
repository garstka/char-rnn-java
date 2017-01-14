package io.github.garstka.rnn;

import io.github.garstka.rnn.net.*;

import java.io.IOException;

public class Main
{

	public static void main(String[] args)
	{
		// Load the training set.

		String inputFile = "input.txt";
		StringTrainingSet trainingSet;
		try
		{
			trainingSet = StringTrainingSet.fromFile(inputFile);
		}
		catch (IOException e)
		{
			System.out.println("Couldn't open the file.");
			return;
		}

		System.out.println("Data size: " + trainingSet.getData().length()
		    + ", vocabulary size: " + trainingSet.getAlphabet().size());

		// Initialize the network.

		SingleLayerCharLevelRNN net = new SingleLayerCharLevelRNN();
		net.printDebug(true);
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
				System.out.println("___________________");
				net.train(100);
				System.out.println(net.sampleString(200, seedChar,false));
			}
		}
		catch (CharacterNotInAlphabetException e)
		{
			// Shouldn't happen.
			throw new RuntimeException("Unexpected alphabet state.", e);
		}
		catch (NoMoreTrainingDataException e)
		{
			System.out.println("Out of training data.");
		}
	}
}
