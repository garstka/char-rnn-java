package io.github.garstka.rnn;

import io.github.garstka.rnn.net.*;
import io.github.garstka.rnn.net.exceptions.BadTrainingSetException;
import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;
import io.github.garstka.rnn.net.exceptions.NoMoreTrainingDataException;

import java.io.IOException;

public class Main
{

	public static void main(String[] args)
	{
		boolean useMultilayer = true;
		try
		{
			// Load the training set.

			String inputFile = "input.txt";

			StringTrainingSet trainingSet =
			    StringTrainingSet.fromFile(inputFile);

			System.out.println("Data size: " + trainingSet.size()
			    + ", vocabulary size: " + trainingSet.vocabularySize());

			// Initialize the network and its trainer.

			CharLevelRNN net = null;

			if (!useMultilayer)
			{
				SingleLayerCharLevelRNN tmp = new SingleLayerCharLevelRNN();
				tmp.setHiddenSize(100);
				tmp.setLearningRate(0.1);
				net = tmp;
			}
			else
			{
				MultiLayerCharLevelRNN tmp = new MultiLayerCharLevelRNN();
				tmp.setHiddenSize(new int[] {50, 50});
				tmp.setLearningRate(0.1);
				net = tmp;
			}

			net.initialize(trainingSet.getAlphabet());

			RNNTrainer trainer = new RNNTrainer();
			trainer.setSequenceLength(50);
			trainer.initialize(net, trainingSet);
			trainer.printDebug(true);

			char seedChar = trainingSet.getData().charAt(0);
			while (true)
			{
				System.out.println("___________________");
				trainer.train(100);
				System.out.println(net.sampleString(200, seedChar, false));
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
