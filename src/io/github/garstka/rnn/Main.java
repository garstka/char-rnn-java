package io.github.garstka.rnn;

import io.github.garstka.rnn.net.*;

import java.io.IOException;

public class Main
{

	public static void main(String[] args)
	{
		try
		{
			// Load the training set.

			String inputFile = "input.txt";

			StringTrainingSet trainingSet = StringTrainingSet.fromFile(inputFile);

			System.out.println("Data size: " + trainingSet.size()
					+ ", vocabulary size: " + trainingSet.vocabularySize());

			// Initialize the network and its trainer.

			SingleLayerCharLevelRNN net = new SingleLayerCharLevelRNN(trainingSet.getAlphabet());
			RNNTrainer trainer = new RNNTrainer(net,trainingSet);
			trainer.printDebug(true);

			char seedChar = trainingSet.getData().charAt(0);
			while (true)
			{
				System.out.println("___________________");
				trainer.train(100);
				System.out.println(net.sampleString(200, seedChar,false));
			}
		}
		catch (IOException e)
		{
			System.out.println("Couldn't open the file.");
		}
		catch (BadTrainingSetException e)
		{
			System.out.println("Bad training set.");
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
