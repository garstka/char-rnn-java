package io.github.garstka.rnn;

import java.io.*;
import java.util.Properties;
import io.github.garstka.rnn.math.Math;

// Application options.
public class Options
{
	/*** Model parameters ***/

	private int hiddenSize; // Size of a single RNN layer hidden state.
	public static final int hiddenSizeDefault = 50;

	private int layers; // How many layers in a net?
	public static final int layersDefault = 2;

	/*** Training parameters ***/

	private int sequenceLength; // How many steps to unroll during training?
	public static final int sequenceLengthDefault = 50;


	private double learningRate; // The network learning rate.
	public static final double learningRateDefault = 0.1;

	/*** Sampling parameters ***/

	// Sampling temperature (0.0, 1.0]. Lower
	// temperature means more conservative
	// predictions.
	private double samplingTemp;
	public static final double samplingTempDefault = 1.0;

	/*** Other options ***/

	private boolean printOptions; // Print options at the start.
	public static final boolean printOptionsDefault = true;

	private int trainingSampleLength; // Length of a sample during training.
	public static final int trainingSampleLengthDefault = 200;

	private String inputFile; // The training data.
	public static final String inputFileDefault = "input.txt";

	private boolean useSingleLayerNet; // Use the simple, single layer net.
	public static final boolean useSingleLayerNetDefault = false;

	/*** Load ***/

	private Properties prop;

	// Constructs, uses the defaults.
	Options()
	{
		prop = new Properties();
		setDefaults();
	}

	// Loads the options from the file, or uses defaults if not found.
	Options(String file) throws IOException
	{
		this();
		try (InputStream in = new FileInputStream(file))
		{
			prop.load(in);
			getProperties();
		}
		catch (IOException e)
		{
			throw new IOException(
			    "Loading config from " + file + " failed.", e);
		}
	}

	/*** Print ***/

	// Prints the options.
	void print()
	{
		setProperties();
		prop.list(System.out);
	}

	/*** Save ***/

	// Saves to file.
	void save(String file) throws IOException
	{
		setProperties();
		try (OutputStream out = new FileOutputStream(file))
		{
			prop.store(out, "---RNN properties---");
		}
		catch (IOException e)
		{
			throw new IOException("Saving config to " + file + " failed.", e);
		}
	}

	/*** Get ***/

	public int getHiddenSize()
	{
		return hiddenSize;
	}

	public int getLayers()
	{
		return layers;
	}

	public int getSequenceLength()
	{
		return sequenceLength;
	}

	public double getLearningRate()
	{
		return learningRate;
	}

	public double getSamplingTemp()
	{
		return samplingTemp;
	}

	public boolean getPrintOptions()
	{
		return printOptions;
	}

	public String getInputFile()
	{
		return inputFile;
	}

	public boolean getUseSingleLayerNet()
	{
		return useSingleLayerNet;
	}

	public int getTrainingSampleLength()
	{
		return trainingSampleLength;
	}

	/*** Helper ***/

	// Sets the default values.
	private void setDefaults()
	{
		hiddenSize = hiddenSizeDefault;
		layers = layersDefault;

		sequenceLength = sequenceLengthDefault;
		learningRate = learningRateDefault;

		samplingTemp = samplingTempDefault;

		printOptions = printOptionsDefault;
		trainingSampleLength = trainingSampleLengthDefault;
		inputFile = inputFileDefault;
		useSingleLayerNet = useSingleLayerNetDefault;
	}

	// Validates the properties and sets to default values where failed.
	private void validateProperties()
	{
		if (hiddenSize < 1)
		{
			hiddenSize = hiddenSizeDefault;
			System.out.println("Hidden size must be >= 1. Using default "
			    + Integer.toString(hiddenSize) + ".");
		}

		if (layers < 1)
		{
			layers = layersDefault;
			System.out.println("Layer count must be >= 1. Using default "
			    + Integer.toString(layers) + ".");
		}

		if (sequenceLength < 1)
		{
			sequenceLength = sequenceLengthDefault;
			System.out.println("Sequence length must be >= 1. Using default "
			    + Integer.toString(sequenceLength) + ".");
		}

		if (learningRate < 0.0)
		{
			learningRate = learningRateDefault;
			System.out.println("Learning rate must be >= 0. Using default "
			    + Double.toString(learningRate) + ".");
		}

		if (Math.close(samplingTemp, 0.0) || samplingTemp < 0.0
		    || samplingTemp > 1.0 + Math.eps())
		{
			learningRate = learningRateDefault;
			System.out.println(
			    "Learning rate must be in (0.0,1.0]. Using default "
			    + Double.toString(learningRate) + ".");
		}

		if (trainingSampleLength < 1)
		{
			trainingSampleLength = trainingSampleLengthDefault;
			System.out.println(
			    "Training sample length must be >= 1. Using default "
			    + Integer.toString(trainingSampleLength) + ".");
		}
	}

	// Gets the properties from the Properties class.
	private void getProperties()
	{
		hiddenSize = parseInt("hiddenSize", hiddenSizeDefault);
		layers = parseInt("layers", layersDefault);
		sequenceLength = parseInt("sequenceLength", sequenceLengthDefault);
		learningRate = parseDouble("learningRate", learningRateDefault);
		samplingTemp = parseDouble("samplingTemp", samplingTempDefault);
		printOptions = parseBool("printOptions", printOptionsDefault);
		trainingSampleLength =
		    parseInt("trainingSampleLength", trainingSampleLengthDefault);
		inputFile = prop.getProperty("inputFile");
		useSingleLayerNet =
		    parseBool("useSingleLayerNet", useSingleLayerNetDefault);

		validateProperties();
	}

	// Saves the properties in the Properties class.
	private void setProperties()
	{
		prop.setProperty("hiddenSize", Integer.toString(hiddenSize));
		prop.setProperty("layers", Integer.toString(layers));
		prop.setProperty("sequenceLength", Integer.toString(sequenceLength));
		prop.setProperty("learningRate", Double.toString(learningRate));
		prop.setProperty("samplingTemp", Double.toString(samplingTemp));
		prop.setProperty("printOptions", Boolean.toString(printOptions));
		prop.setProperty(
		    "trainingSampleLength", Integer.toString(trainingSampleLength));
		prop.setProperty("inputFile", inputFile);
		prop.setProperty(
		    "useSingleLayerNet", Boolean.toString(useSingleLayerNet));
	}

	// Parses int, returns the default value if failed.
	private int parseInt(String name, int defaultValue)
	{
		try
		{
			return Integer.parseInt(prop.getProperty(name));
		}
		catch (NumberFormatException e)
		{
			System.out.println("Error parsing " + name + ": "
			    + prop.getProperty(name) + ", defaulting to: "
			    + Integer.toString(defaultValue));
			return defaultValue;
		}
	}

	// Parses double, returns the default value if failed.
	private double parseDouble(String name, double defaultValue)
	{
		try
		{
			return Double.parseDouble(prop.getProperty(name));
		}
		catch (NumberFormatException e)
		{
			System.out.println("Error parsing " + name + ": "
			    + prop.getProperty(name) + ", defaulting to: "
			    + Double.toString(defaultValue));
			return defaultValue;
		}
	}

	// Parses boolean, returns the default value if failed.
	private boolean parseBool(String name, boolean defaultValue)
	{
		try
		{
			return Boolean.parseBoolean(prop.getProperty(name));
		}
		catch (NumberFormatException e)
		{
			System.out.println("Error parsing " + name + ": "
			    + prop.getProperty(name) + ", defaulting to: "
			    + Boolean.toString(defaultValue));
			return defaultValue;
		}
	}
}