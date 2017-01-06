package io.github.garstka.rnn;

public class Main
{

	public static void main(String[] args)
	{
		String inputFile = "input.txt";

		TrainingSet trainingSet = new TrainingSet();
		try
		{
			trainingSet.fromFile(inputFile);
		}
		catch (MissingFileException e)
		{
			System.out.println("File not found");
			return;
		}

		System.out.println("Data byte count: " + inputFile.dataSize()
		    + ", vocabulary size: " + inputFile.vocabSize());
	}
}
