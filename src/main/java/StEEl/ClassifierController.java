package StEEl;

import StEEl.run1.TinyImageClassifier;
import StEEl.run2.LinearClassifier;
import StEEl.run3.ComplexClassifier;
import org.apache.commons.vfs2.FileSystemException;
import org.jetbrains.annotations.NotNull;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.transform.AffineSimulation;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static StEEl.ClassifierUtils.parallelAwarePrintln;

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
	private              GroupedDataset<String, ListDataset<FImage>, FImage>    trainingDataset     = null;
	private              VFSListDataset<FImage>                                 testDataset         = null;
	private              boolean                                                consoleOutput       = false;
	private              boolean                                                writeSubmissionFile = false;
	private ExecutorService topExecutor;


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

			//Create an executor service for running the 3 classifiers in parallel
			topExecutor = Executors.newCachedThreadPool();

			//Create training/testing datasets
			initialiseData();

			//Creates instances of the three classifiers, give them each a unique number to identify them
			final IClassifier run1TinyImage = new TinyImageClassifier(1);
			final IClassifier run2LinearClassifier = new LinearClassifier(2);
			final IClassifier run3ComplexClassifier = new ComplexClassifier(3);

			//Construct async tasks(runnables) for each classifier
			final Runnable c1task = () -> runClassifierTask(run1TinyImage);
			final Runnable c2task = () -> runClassifierTask(run2LinearClassifier);
			final Runnable c3task = () -> runClassifierTask(run3ComplexClassifier);

			//Asynchronous execution
/*
			//Use the executor to invoke each classifier task, these are executed in parallel to each other
			topExecutor.execute(c1task);
			topExecutor.execute(c2task);
			topExecutor.execute(c3task);
			//
			//Orders the executor to finish all tasks, will give it 20 minutes to complete execution, at which point is will force shutdown their threads
			topExecutor.shutdown();
			topExecutor.awaitTermination(20, TimeUnit.MINUTES);
*/
			//Synchronous execution
			//Execute each tasks one at a time, waiting for each one to finish before invoking the next
			topExecutor.execute(c1task);
			topExecutor.shutdown();
			topExecutor.awaitTermination(20, TimeUnit.MINUTES);

			topExecutor = Executors.newCachedThreadPool();

			topExecutor.execute(c2task);
			topExecutor.shutdown();
			topExecutor.awaitTermination(20, TimeUnit.MINUTES);

			topExecutor = Executors.newCachedThreadPool();

			topExecutor.execute(c3task);
			topExecutor.shutdown();
			topExecutor.awaitTermination(20, TimeUnit.MINUTES);
		}
		catch (final IOException | InterruptedException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * Creates the data structures containing the testing/training images
	 *
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

		GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingData = new VFSGroupDataset<FImage>(TRAINING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
		trainingDataset = createTrainingAndValidationData(trainingData);
		testDataset = new VFSListDataset<FImage>(TESTING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
	}


	/**
	 * Wrapper for "runclassifier()" to include ClassiferException catching
	 *
	 * @param classifier
	 */
	private void runClassifierTask(final IClassifier classifier)
	{
		try
		{
			runClassifier(classifier);
		}
		catch (ClassifierException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * Runs the classifier, training it, testing it, then evaluating it
	 *
	 * @param instance The current classifier instance
	 * @throws ClassifierException
	 */
	private void runClassifier(IClassifier instance) throws ClassifierException
	{
		// Loading test dataset
		if (testDataset == null)
		{
			throw new ClassifierException("Error loading test dataset");
		}

		parallelAwarePrintln(instance, "Test set size = " + testDataset.size());
		final VFSListDataset<FImage> testData = testDataset;

		parallelAwarePrintln(instance, "Training dataset loaded. Staring training...");

		trainClassifer(instance, trainingDataset);

		parallelAwarePrintln(instance, "Training complete. Now predicting... (progress on stderr, output on stout)");

		testClassifier(instance, testData);

		evaluateClassifier(instance, trainingDataset);

	}


	/**
	 * Does some reorganising of the training dataset, and extracts a subset training data and validation data from it
	 *
	 * @param trainingDataset Dataset to reorganise
	 * @return
	 * @throws Exception
	 */
	private GroupedDataset<String, ListDataset<FImage>, FImage> createTrainingAndValidationData(GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset) {
		final GroupedUniformRandomisedSampler<String, FImage> groupSampler = new GroupedUniformRandomisedSampler<>(1.0d);

		//Converts the inner image list from the VFS version to the generic version
		GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = GroupSampler.sample(trainingDataset, trainingDataset.size(), false);

		final int trainingDataSize = trainingData.size();
		final GroupedRandomSplitter<String, FImage> trainingSplitter = splitTrainingData(trainingData, (double) trainingDataSize);

		GroupedDataset<String, ListDataset<FImage>, FImage> newTrainingDataset = trainingSplitter.getTrainingDataset();
		final GroupedDataset<String, ListDataset<FImage>, FImage> validationData = trainingSplitter.getValidationDataset();

		newTrainingDataset = addRotationImages(newTrainingDataset);

		return newTrainingDataset;
	}

	private GroupedDataset<String, ListDataset<FImage>, FImage> addRotationImages(GroupedDataset<String, ListDataset<FImage>, FImage> dataset) {
		GroupedDataset<String, ListDataset<FImage>, FImage> tmp = dataset;

		for (String key : dataset.keySet()) {
			ListDataset<FImage> newImages = new ListBackedDataset<>();

			for (FImage image : dataset.getInstances(key)) {
				newImages.add(image);
				newImages.add(AffineSimulation.transformImage(image, 0.01f, 1));
				newImages.add(AffineSimulation.transformImage(image, -0.01f, 1));
				newImages.add(AffineSimulation.transformImage(image, 0.02f, 1));
				newImages.add(AffineSimulation.transformImage(image, -0.02f, 1));
			}

			tmp.put(key, newImages);
		}

		return tmp;
	}


	/**
	 * Calls the train method of the classifier to train it using the training data
	 *
	 * @param instance     The current classifier instance
	 * @param trainingData
	 */
	private static void trainClassifer(final IClassifier instance, final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
	{
		instance.train(trainingData);
	}


	/**
	 * Applies the trained classifier to a set of testing images, attempts to predict their image classes
	 *
	 * @param instance The current classifier instance
	 * @param testData
	 */
	private void testClassifier(final IClassifier instance, final VFSListDataset<FImage> testData)
	{
		try
		{
			//Creates a fresh submission file for classifier output
			File submissionFile = new File(System.getProperty("user.dir") + "default.txt");
			submissionFile = setSubmissionFileLocation(instance, submissionFile);
			Files.deleteIfExists(submissionFile.toPath());
			Files.write(submissionFile.toPath(), "".getBytes(), StandardOpenOption.CREATE);

			ExecutorService classifyExecutor = Executors.newCachedThreadPool();

			//Iterates through each test image, runs the classifier on the image
			for (int j = 0; j < testData.size(); j++)
			{
				final File finalSubmissionFile = submissionFile;
				final int finalJ = j;
				//				ClassifierUtils.parallelAwarePrintln(instance, MessageFormat.format("Classifying Image {0}", j));
				classifyExecutor.execute(() -> classifyImage(instance, testData, finalSubmissionFile, finalJ));
			}

			//Awaits on classification of all images in the testing set (timeout 20 minutes)
			classifyExecutor.shutdown();
			classifyExecutor.awaitTermination(20, TimeUnit.MINUTES);

			parallelAwarePrintln(instance, "\n Done.");
		}
		catch (ClassifierException | IOException | InterruptedException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * Evaluates the performance and accuracy of the classifier with the validation data it sampled earlier
	 *
	 * @param instance     The current classifier instance
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

			parallelAwarePrintln(instance, result.getDetailReport());
		}
	}


	/**
	 * 80% of the training dataset is randomly selected as the trianing data, the other 20% is selected as validation data for use in the evaluator
	 *
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
	 * Select the correct submission filename based on the current classifier
	 *
	 * @param instance           The current classifier instance
	 * @param tempSubmissionFile
	 * @return
	 * @throws ClassifierException
	 */
	private static File setSubmissionFileLocation(final IClassifier instance, File tempSubmissionFile) throws ClassifierException
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
	 * Applies the classifiers "classify()" method to an image, prediciting its image class
	 * Then writes this to console and submission file, if these options are enabled
	 *
	 * @param instance              The current classifier instance
	 * @param testDatasetToClassify
	 * @param submissionFile
	 * @param j
	 */
	private void classifyImage(final IClassifier instance, final VFSListDataset<FImage> testDatasetToClassify, final File submissionFile, final int j)
	{
		try
		{
			final FImage img = testDatasetToClassify.get(j);
			final String filename = testDatasetToClassify.getID(j);

			ExecutorService printExecutor = Executors.newCachedThreadPool();

			final ClassificationResult<String> predicted = instance.classify(img);
			if (predicted != null)
			{
				if (consoleOutput)
				{
					printExecutor.execute(() -> printTestProgress(instance, testDatasetToClassify, j, filename, predicted));
				}
				if (writeSubmissionFile)
				{
					printExecutor.execute(() -> writeResult(submissionFile, filename, predicted));
				}
			}
			printExecutor.shutdown();
			printExecutor.awaitTermination(10, TimeUnit.SECONDS);
		}
		catch (InterruptedException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * Output assembler for console output
	 *
	 * @param instance              The current classifier instance
	 * @param classifiedTestDataset
	 * @param j
	 * @param file
	 * @param predicted
	 */
	private static void printTestProgress(final IClassifier instance, final VFSListDataset<FImage> classifiedTestDataset, final int j, final String file,
			final ClassificationResult<String> predicted)
	{
		if (predicted != null)
		{
			final String[] classes = getPredictedClassesArray(predicted);

			buildAndPrintProgressString(instance, classifiedTestDataset, j, file, classes);

		}
	}


	@NotNull
	private static String[] getPredictedClassesArray(final ClassificationResult<String> predicted)
	{
		final Set<String> predictedClasses = predicted.getPredictedClasses();
		return predictedClasses.toArray(new String[predictedClasses.size()]);
	}


	/**
	 * Writes a line of the submission file, using the name of the image, and its predicted class
	 *
	 * @param file
	 * @param imageName
	 * @param predictedImageClasses
	 */
	private static void writeResult(File file, String imageName, ClassificationResult<String> predictedImageClasses)
	{
		if (predictedImageClasses != null)
		{
			final String[] classes = getPredictedClassesArray(predictedImageClasses);

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


	/**
	 * String builder for console output, using image name and predicted class
	 *
	 * @param instance              The current classifier instance
	 * @param classifiedTestDataset
	 * @param j
	 * @param file
	 * @param classes
	 */
	private static void buildAndPrintProgressString(final IClassifier instance, final VFSListDataset<FImage> classifiedTestDataset, final int j, final String file,
			final String[] classes)
	{
		StringBuilder sb = new StringBuilder();
		sb.append(file).append(" ");

		for (final String cls : classes)
		{
			sb.append(cls);
			sb.append(" ");
		}
		sb.append(System.lineSeparator());
		try
		{
			sb.append("\r").append("[").append(instance.getClassifierID()).append("] -- ");
		}
		catch (ClassifierException e)
		{
			e.printStackTrace();
		}

		System.out.print(sb);
	}

}



