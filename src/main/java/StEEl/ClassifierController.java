package StEEl;

import StEEl.run1.TinyImageClassifier;
import StEEl.run2.LinearClassifier;
import StEEl.run3.ComplexClassifier;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Set;

class ClassifierController
{

	private static final String CURRENT_WORKING_DIRECTORY = System.getProperty("user.dir");
	private static final File   RUN_1_FILE                = new File(CURRENT_WORKING_DIRECTORY + "/run1.txt");
	private static final File   RUN_2_FILE                = new File(CURRENT_WORKING_DIRECTORY + "/run2.txt");
	private static final File   RUN_3_FILE                = new File(CURRENT_WORKING_DIRECTORY + "/run3.txt");
	private static final File   TRAINING_DATA_DIRECTORY   = new File(CURRENT_WORKING_DIRECTORY + "/training/");
	//	private static final File   TRAINING_DATA_DIRECTORY   = new File("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\training.zip");
	private static final File   TESTING_DATA_DIRECTORY    = new File(CURRENT_WORKING_DIRECTORY + "/testing/");

	private static final int                                                    TINYIMAGE_ID        = 1;
	private static final int                                                    LINEAR_ID           = 2;
	private static final int                                                    COMPLEX_ID          = 3;
	//	private static final File   TESTING_DATA_DIRECTORY    = new File("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\testing.zip");
	private              GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset     = null;
	private              VFSListDataset<FImage>                                 testDataset         = null;
	private              boolean                                                consoleOutput       = false;
	private              boolean                                                writeSubmissionFile = false;


	/**
	 * @param consoleOutputArg
	 * @param writeSubmissionFileArg
	 */
	@SuppressWarnings("OverlyBroadCatchBlock")
	public final void runClassifiers(boolean consoleOutputArg, boolean writeSubmissionFileArg)
	{
		try
		{
			this.consoleOutput = consoleOutputArg;
			this.writeSubmissionFile = writeSubmissionFileArg;

			initialiseData();

			final IClassifier run1TinyImage = new TinyImageClassifier(1);
			final IClassifier run2LinearClassifier = new LinearClassifier(2);
			final IClassifier run3ComplexClassifier = new ComplexClassifier(3);

			runClassifier(run1TinyImage);
			runClassifier(run2LinearClassifier);
			runClassifier(run3ComplexClassifier);

		}
		catch (final IOException | ClassifierException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * @throws IOException
	 * @throws FileSystemException
	 */
	private void initialiseData() throws IOException, FileSystemException
	{
		if (!TRAINING_DATA_DIRECTORY.exists())
		{
			throw new IOException("Training data missing");
		}
		if (!TESTING_DATA_DIRECTORY.exists())
		{
			throw new IOException("Testing data missing");
		}

		trainingDataset = new VFSGroupDataset<FImage>(TRAINING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
		testDataset = new VFSListDataset<FImage>(TESTING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);

		//		trainingDataset = new VFSGroupDataset<FImage>("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\training.zip",
		//				ImageUtilities.FIMAGE_READER);
		//		testDataset = new VFSGroupDataset<FImage>("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\testing.zip", ImageUtilities.FIMAGE_READER);
	}


	/**
	 * @param instance
	 * @throws ClassifierException
	 */
	private void runClassifier(IClassifier instance) throws ClassifierException
	{

		final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = createTrainingAndValidationData();
		final VFSListDataset<FImage> testData = createTestDataset();

		System.out.println("Training dataset loaded. Staring training...");

		trainClassifer(instance, trainingData);

		System.out.println("Training complete. Now predicting... (progress on stderr, output on stout)");

		testClassifier(instance, testData);
		evaluateClassifier(instance, trainingData);

	}


	/**
	 * @return
	 * @throws Exception
	 */
	private GroupedDataset<String, ListDataset<FImage>, FImage> createTrainingAndValidationData()
	{
		final GroupedUniformRandomisedSampler<String, FImage> groupSampler = new GroupedUniformRandomisedSampler<>(1.0d);

		// Sample all data so we have a GroupedDataset<String, ListDataset<FImage>, FImage>, and not a group with a VFSListDataset.
		GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = GroupSampler.sample(trainingDataset, trainingDataset.size(), false);

		final int trainingDataSize = trainingData.size();
		final GroupedRandomSplitter<String, FImage> trainingSplitter = splitTrainingData(trainingData, (double) trainingDataSize);

		GroupedDataset<String, ListDataset<FImage>, FImage> newTrainingDataset = trainingSplitter.getTrainingDataset();
		final GroupedDataset<String, ListDataset<FImage>, FImage> validationData = trainingSplitter.getValidationDataset();

		System.out.println("Training set size = " + trainingDataSize);
		//			System.out.println("Control set size = " + controlData.size());
		return newTrainingDataset;
	}


	/**
	 * @return
	 * @throws Exception
	 */
	private VFSListDataset<FImage> createTestDataset() throws ClassifierException
	{
		//		UniformRandomisedSampler<FImage> listSampler = new UniformRandomisedSampler<FImage>(0.8d);
		//
		//		final ListDataset<FImage> testData = listSampler.sample(testDataset);

		final VFSListDataset<FImage> testData = testDataset;

		// Loading test dataset
		if (testDataset == null)
		{
			throw new ClassifierException("Error loading test dataset");
		}

		System.out.println("Test set size = " + testData.size());
		return testData;
	}


	/**
	 * @param instance
	 * @param trainingData
	 */
	private static void trainClassifer(final IClassifier instance, final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
	{
		instance.train(trainingData);
	}


	/**
	 * @param instance
	 * @param testData
	 */
	private void testClassifier(final IClassifier instance, final VFSListDataset<FImage> testData)
	{
		try
		{
			File submissionFile = new File(System.getProperty("user.dir") + "default.txt");
			submissionFile = setSubmissionFileLocation(instance, submissionFile);
			Files.write(submissionFile.toPath(), "".getBytes(), StandardOpenOption.CREATE);
			Files.write(submissionFile.toPath(), "".getBytes(), StandardOpenOption.WRITE);

			for (int j = 0; j < testData.size(); j++)
			{
				classifyImage(instance, testData, submissionFile, j);
			}
			System.out.println("\n Done.");
		}
		catch (ClassifierException | IOException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * @param instance
	 * @param trainingData
	 */
	private static void evaluateClassifier(final IClassifier instance, final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
	{
		//			 OpenIMAJ evaluation method.
		final ClassificationEvaluator<CMResult<String>, String, FImage> evaluator = new ClassificationEvaluator<CMResult<String>, String, FImage>(instance, trainingData,
				new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

		final Map<FImage, ClassificationResult<String>> guesses = evaluator.evaluate();
		if (guesses.get(guesses.keySet().toArray()[0]) != null) //Complex null checking
		{
			final CMResult<String> result = evaluator.analyse(guesses);

			System.out.println(result.getDetailReport());
		}
	}


	/**
	 * @param trainingData
	 * @param trainingDataSize
	 * @return
	 */
	private static GroupedRandomSplitter<String, FImage> splitTrainingData(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData,
			final double trainingDataSize)
	{
		final int PERCENT80 = (int) Math.round(trainingDataSize * 0.8);
		final int PERCENT20 = (int) Math.round(trainingDataSize * 0.2);
		return new GroupedRandomSplitter<String, FImage>(trainingData, PERCENT80, PERCENT20, 0);
	}


	/**
	 * @param instance
	 * @param tempSubmissionFile
	 * @return
	 * @throws ClassifierException
	 */
	private File setSubmissionFileLocation(final IClassifier instance, File tempSubmissionFile) throws ClassifierException
	{
		switch (instance.getClassifierID())
		{
			case TINYIMAGE_ID:
				tempSubmissionFile = RUN_1_FILE;
				break;
			case LINEAR_ID:
				tempSubmissionFile = RUN_2_FILE;
				break;
			case COMPLEX_ID:
				tempSubmissionFile = RUN_3_FILE;
				break;
			default:
				throw new ClassifierException("Undefined classifier ID");

		}
		return tempSubmissionFile;
	}


	/**
	 * @param instance
	 * @param testDatasetToClassify
	 * @param submissionFile
	 * @param j
	 */
	private void classifyImage(final IClassifier instance, final VFSListDataset<FImage> testDatasetToClassify, final File submissionFile, final int j)
	{
		final FImage img = testDatasetToClassify.get(j);
		final String filename = testDatasetToClassify.getID(j);

		final ClassificationResult<String> predicted = instance.classify(img);
		if (consoleOutput)
		{
			printTestProgress(testDatasetToClassify, j, filename, predicted);
		}
		if (writeSubmissionFile)
		{
			new Thread(() -> writeResult(submissionFile, filename, predicted)).start();
			//			writeResult(submissionFile, filename, predicted);
		}
	}


	/**
	 * @param classifiedTestDataset
	 * @param j
	 * @param file
	 * @param predicted
	 */
	private static void printTestProgress(final VFSListDataset<FImage> classifiedTestDataset, final int j, final String file, final ClassificationResult<String> predicted)
	{
		if (predicted != null)
		{
			final Set<String> predictedClasses = predicted.getPredictedClasses();
			final String[] classes = predictedClasses.toArray(new String[predictedClasses.size()]);

			System.out.print(file);
			System.out.print(" ");
			for (final String cls : classes)
			{
				System.out.print(cls);
				System.out.print(" ");
			}
			System.out.println();
			System.out.printf("\r %d %%  ", Math.round(((double) (j + 1) * 100.0) / (double) classifiedTestDataset.size()));

		}
	}


	/**
	 * @param file
	 * @param imageName
	 * @param predictedImageClasses
	 */
	private static void writeResult(File file, String imageName, ClassificationResult<String> predictedImageClasses)
	{
		if (predictedImageClasses != null)
		{
			final Set<String> predictedClasses = predictedImageClasses.getPredictedClasses();
			final String[] classes = predictedClasses.toArray(new String[predictedClasses.size()]);

			try
			{
				final StringBuilder sb = new StringBuilder();
				sb.append(imageName).append(' ');
				for (final String cls : classes)
				{
					sb.append(cls);
					sb.append(' ');
				}
				sb.append(System.lineSeparator());
				Files.write(file.toPath(), sb.toString().getBytes(), StandardOpenOption.APPEND);
				//			throw new UnsupportedOperationException();
			}
			catch (final IOException e)
			{
				e.printStackTrace();
			}
		}
		else
		{
		}
	}

}



