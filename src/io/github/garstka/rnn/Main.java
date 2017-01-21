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
		String configFile = "config.properties";
		Options options = null;
		try
		{
			options = new Options(configFile);
		}
		catch (IOException e)
		{
			options = new Options();
			System.out.println("Using the defaults.");

			// Save config.
			try
			{
				options.save(configFile);
			}
			catch (IOException e2)
			{
				System.out.println("Couldn't save the options.");
			}
		}

		if (options.getPrintOptions()) // Print options
			options.print();

		boolean useMultilayer = true;
		try
		{
			/* Load the training set. */

			StringTrainingSet trainingSet =
			    StringTrainingSet.fromFile(options.getInputFile());

			System.out.println("Data size: " + trainingSet.size()
			    + ", vocabulary size: " + trainingSet.vocabularySize());

			// Initialize the network and its trainer.

			CharLevelRNN net = null;

			if (options.getUseSingleLayerNet()) // legacy network, single layer only
			{
				SingleLayerCharLevelRNN tmp = new SingleLayerCharLevelRNN();
				tmp.setHiddenSize(options.getHiddenSize());
				tmp.setLearningRate(options.getLearningRate());
				net = tmp;
			}
			else // Multi layer network.
			{
				MultiLayerCharLevelRNN tmp = new MultiLayerCharLevelRNN();

				// Use the same hidden size for all layers.
				int hiddenSize = options.getHiddenSize();
				int[] hidden = new int[options.getLayers()];
				for(int i = 0; i < hidden.length; i++)
					hidden[i] = hiddenSize;
				tmp.setHiddenSize(hidden);
				tmp.setLearningRate(options.getLearningRate());
				net = tmp;
			}

			net.initialize(trainingSet.getAlphabet());

			RNNTrainer trainer = new RNNTrainer();
			trainer.setSequenceLength(options.getSequenceLength());
			trainer.initialize(net, trainingSet);
			trainer.printDebug(true);

			String seed = Character.toString(trainingSet.getData().charAt(0));
			double samplingTemperature = options.getSamplingTemp();
			int sampleLength = options.getTrainingSampleLength();
			while(true)
			{
				try {
					while (true) {
						System.out.println("___________________");
						trainer.train(100);
						System.out.println(net.sampleString(sampleLength, seed, samplingTemperature, false));
					}
				} catch (NoMoreTrainingDataException e) {
					System.out.println("Out of training data.");
				}

				trainer.loopAround();
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
	}
}
