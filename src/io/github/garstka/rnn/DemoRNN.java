package io.github.garstka.rnn;

import io.github.garstka.rnn.net.*;
import io.github.garstka.rnn.net.exceptions.BadTrainingSetException;
import io.github.garstka.rnn.net.exceptions.CharacterNotInAlphabetException;
import io.github.garstka.rnn.net.exceptions.NoMoreTrainingDataException;

import java.io.*;
import java.util.Scanner;

public class DemoRNN
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

		Scanner scanner = new Scanner(System.in);
		while (true)
		{
			System.out.println("Choose an action:");
			System.out.println(
			    "1. Create a new network and start training it.");
			System.out.println(
			    "2. Restore a snapshot and continue training it.");
			System.out.println(
			    "3. Restore a snapshot and sample it (generate text).");
			System.out.println("(anything else to quit)");

			try
			{
				CharLevelRNN net = null;
				String networkName = null;

				int nextChar = Integer.parseInt(scanner.nextLine());
				if (nextChar == 1) // Create a new network
				{
					System.out.println("New network name: ");
					networkName = scanner.nextLine();
					net = initialize(options);
				}
				else if (nextChar == 2 || nextChar == 3) // From snapshot
				{
					System.out.println(".snapshot file name: ");
					networkName = scanner.nextLine();
					try
					{
						net = loadASnapshot(networkName);
					}
					catch (IOException e)
					{
						System.out.println("Couldn't load from file.");
						continue;
					}
				}
				else // Exit
				{
					break;
				}

				if (nextChar == 1 || nextChar == 2) // train
					train(options, net, networkName);
				else // sample
				{
					double temp = options.getSamplingTemp();
					while (true)
					{
						System.out.println(
						    "How many characters to sample (<1 to exit): ");
						int characters = Integer.parseInt(scanner.nextLine());

						if (characters < 1)
							break;
						System.out.println("Seed string: ");
						String seed = scanner.nextLine();
						if (seed.length() == 0)
						{
							System.out.println("Seed must not be empty.");
							continue;
						}

						sample(characters, seed, temp, net);
					}
				}
			}
			catch (NumberFormatException e)
			{
				System.out.println("Expected a number.");
			}
		}
	}


	// Initialize a network for training.
	private static CharLevelRNN initialize(Options options)
	{
		if (options.getUseSingleLayerNet())
		{
			// legacy network, single layer only
			SingleLayerCharLevelRNN net = new SingleLayerCharLevelRNN();
			net.setHiddenSize(options.getHiddenSize());
			net.setLearningRate(options.getLearningRate());
			return net;
		}
		else // Multi layer network.
		{
			MultiLayerCharLevelRNN net = new MultiLayerCharLevelRNN();

			// Use the same hidden size for all layers.
			int hiddenSize = options.getHiddenSize();
			int[] hidden = new int[options.getLayers()];
			for (int i = 0; i < hidden.length; i++)
				hidden[i] = hiddenSize;
			net.setHiddenSize(hidden);
			net.setLearningRate(options.getLearningRate());
			return net;
		}
	}

	// Train the network.
	private static void train(
	    Options options, CharLevelRNN net, String snapshotName)
	{
		if (options == null || net == null || snapshotName == null)
			throw new IllegalArgumentException("Params can't be null.");

		try
		{

			// Load the training set.

			StringTrainingSet trainingSet =
			    StringTrainingSet.fromFile(options.getInputFile());

			System.out.println("Data size: " + trainingSet.size()
			    + ", vocabulary size: " + trainingSet.vocabularySize());

			// Initialize the network and its trainer.

			if (!net.isInitialized()) // Only if not restored from a snapshot.
				net.initialize(trainingSet.getAlphabet());

			RNNTrainer trainer = new RNNTrainer();
			trainer.setSequenceLength(options.getSequenceLength());
			trainer.initialize(net, trainingSet);
			trainer.printDebug(true);

			// For sampling during training, pick the temperature from options
			// and the first character in the training set as seed.
			String seed = Character.toString(trainingSet.getData().charAt(0));
			double samplingTemperature = options.getSamplingTemp();
			int sampleLength = options.getTrainingSampleLength();


			int loopTimes = options.getLoopAroundTimes();
			int sampleEveryNSteps = options.getSampleEveryNSteps();
			int snapshotEveryNSamples = options.getSnapshotEveryNSamples();
			int nextSnapshotNumber = 1;
			while (true) // Go over the whole training set.
			{

				try
				{

					// Train for some steps and sample a short string
					// for evaluation.
					int batchCount = 0;
					while (true)
					{

						if ((batchCount++) % snapshotEveryNSamples == 0)
							saveASnapshot(
							    snapshotName + "-" + (nextSnapshotNumber++),
							    net);

						System.out.println("___________________");

						trainer.train(sampleEveryNSteps);

						System.out.println(net.sampleString(
						    sampleLength, seed, samplingTemperature, false));
					}
				}
				catch (NoMoreTrainingDataException e)
				{
					System.out.println("Out of training data.");
				}

				saveASnapshot(snapshotName + "-" + (nextSnapshotNumber++), net);

				if (--loopTimes < 0)
					break;

				System.out.println(
				    "Looping around " + (loopTimes + 1) + "more time(s).");

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
			throw new RuntimeException(
			    "Different alphabet - can't train on this dataset.", e);
		}
	}

	// Save a network snapshot with this name to file.
	private static void saveASnapshot(String name, CharLevelRNN net)
	{
		if (name == null || net == null || !net.isInitialized())
			throw new IllegalArgumentException();

		// Take a snapshot
		try (FileOutputStream str = new FileOutputStream(name + ".snapshot"))
		{
			ObjectOutputStream ostr = new ObjectOutputStream(str);
			ostr.writeObject(net);
			ostr.close();
		}
		catch (IOException e)
		{
			System.out.println("Couldn't save a snapshot.");
			return;
		}
		System.out.println("Saved as " + name + ".snapshot");
	}

	// Load a network snapshot with this name from file.
	private static CharLevelRNN loadASnapshot(String name) throws IOException
	{
		if (name == null)
			throw new IllegalArgumentException();

		CharLevelRNN net = null;

		// Take a snapshot
		try (FileInputStream str = new FileInputStream(name + ".snapshot"))
		{
			ObjectInputStream ostr = new ObjectInputStream(str);
			net = (CharLevelRNN) ostr.readObject();
			ostr.close();
		}
		catch (IOException | ClassNotFoundException | ClassCastException e)
		{
			throw new IOException("Couldn't load the snapshot from file.", e);
		}
		return net;
	}


	// Just samples the net for n characters.
	private static void sample(
	    int n, String seed, double temperature, CharLevelRNN net)
	{
		if (n < 1 || net == null || !net.isInitialized())
			throw new IllegalArgumentException();

		try
		{
			System.out.println(
			    net.sampleString(n, seed, temperature)); // sample and advance
		}
		catch (CharacterNotInAlphabetException e)
		{
			System.out.println("Error: Character not in alphabet.");
		}
	}
}
