package StEEl;

import StEEl.run1.TinyImageClassifier;
import StEEl.run2.LinearClassifier;
import StEEl.run3.ComplexClassifier;
import org.apache.commons.vfs2.FileSystemException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.transform.AffineSimulation;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Map;
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

	private static final int                                                 TINYIMAGE_ID        = 1;
	private static final int                                                 LINEAR_ID           = 2;
	private static final int                                                 COMPLEX_ID          = 3;
	//	private static final File   TESTING_DATA_DIRECTORY    = new File("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\testing.zip");
	private @Nullable    GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset     = null;
	private @Nullable    VFSListDataset<FImage>                              testDataset         = null;
	private              boolean                                             consoleOutput       = false;
	private              boolean                                             writeSubmissionFile = false;


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
			ExecutorService topExecutor = Executors.newCachedThreadPool();

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
			//			topExecutor.execute(c1task);
			//			topExecutor.shutdown();
			//			topExecutor.awaitTermination(20, TimeUnit.MINUTES);
			//
//			topExecutor = Executors.newCachedThreadPool();
//
//			topExecutor.execute(c2task);
//			topExecutor.shutdown();
//			topExecutor.awaitTermination(20, TimeUnit.MINUTES);

						topExecutor = Executors.newCachedThreadPool();

						topExecutor.execute(c3task);
						topExecutor.shutdown();
						topExecutor.awaitTermination(20, TimeUnit.MINUTES);
		}
		catch (final @NotNull IOException | InterruptedException e)
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

		final GroupedDataset<String, VFSListDataset<FImage>, FImage> vfsTrainingData = new VFSGroupDataset<FImage>(TRAINING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
		trainingDataset = createTrainingAndValidationData(vfsTrainingData);
		testDataset = new VFSListDataset<FImage>(TESTING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
	}


	/**
	 * Wrapper for "runclassifier()" to include ClassiferException catching
	 *
	 * @param classifier
	 */
	private void runClassifierTask(final @NotNull IClassifier classifier)
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
	 * Does some reorganising of the training dataset, and extracts a subset training data and validation data from it
	 *
	 * @param vfsTrainingData
	 * @return
	 */
	private static GroupedDataset<String, ListDataset<FImage>, FImage> createTrainingAndValidationData(
			final @NotNull GroupedDataset<String, VFSListDataset<FImage>, FImage> vfsTrainingData)
	{
		//		final GroupedUniformRandomisedSampler<String, FImage> groupSampler = new GroupedUniformRandomisedSampler<>(1.0d);

		//Converts the inner image list from the VFS version to the generic version
		final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = GroupSampler.sample(vfsTrainingData, vfsTrainingData.size(), false);

		final int trainingDataSize = trainingData.size();
		final GroupedRandomSplitter<String, FImage> trainingSplitter = splitTrainingData(trainingData, (double) trainingDataSize);

		GroupedDataset<String, ListDataset<FImage>, FImage> newTrainingDataset = trainingSplitter.getTrainingDataset();
		final GroupedDataset<String, ListDataset<FImage>, FImage> validationData = trainingSplitter.getValidationDataset();

		newTrainingDataset = addRotationImages(newTrainingDataset);

		return newTrainingDataset;
	}


	/**
	 * Runs the classifier, training it, testing it, then evaluating it
	 *
	 * @param instance The current classifier instance
	 * @throws ClassifierException
	 */
	private void runClassifier(@NotNull IClassifier instance) throws ClassifierException
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

		if (trainingDataset != null)
		{
			evaluateClassifier(instance, trainingDataset);
		}
		else
		{
			throw new ClassifierException("Training dataset is null when attempting to evaluate");
		}

	}


	/**
	 * 80% of the training dataset is randomly selected as the trianing data, the other 20% is selected as validation data for use in the evaluator
	 *
	 * @param trainingData
	 * @param trainingDataSize
	 * @return
	 */
	private static @NotNull GroupedRandomSplitter<String, FImage> splitTrainingData(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData,
			final double trainingDataSize)
	{
		final int percent80 = (int) Math.round(trainingDataSize * 0.8);
		final int percent20 = (int) Math.round(trainingDataSize * 0.2);
		return new GroupedRandomSplitter<String, FImage>(trainingData, percent80, percent20, 0);
	}


	private static @NotNull GroupedDataset<String, ListDataset<FImage>, FImage> addRotationImages(@NotNull GroupedDataset<String, ListDataset<FImage>, FImage> dataset)
	{
		final ListDataset<FImage> newImages = new ListBackedDataset<>();

		for (final String key : dataset.keySet())
		{
			newImages.clear();
			for (final FImage image : dataset.getInstances(key))
			{
				newImages.add(image);
				newImages.add(AffineSimulation.transformImage(image, 0.01f, 1));
				newImages.add(AffineSimulation.transformImage(image, -0.01f, 1));
				newImages.add(AffineSimulation.transformImage(image, 0.02f, 1));
				newImages.add(AffineSimulation.transformImage(image, -0.02f, 1));
			}

			dataset.put(key, newImages);
		}

		return dataset;
	}


	/**
	 * Calls the train method of the classifier to train it using the training data
	 *
	 * @param instance     The current classifier instance
	 * @param trainingData
	 */
	private static void trainClassifer(final @NotNull IClassifier instance, final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
	{
		instance.train(trainingData);
	}


	/**
	 * Applies the trained classifier to a set of testing images, attempts to predict their image classes
	 *
	 * @param instance The current classifier instance
	 * @param testData
	 */
	private void testClassifier(final @NotNull IClassifier instance, final @NotNull VFSListDataset<FImage> testData)
	{
		try
		{
			//Creates a fresh submission file for classifier output
			File submissionFile = new File(System.getProperty("user.dir") + "default.txt");
			submissionFile = setSubmissionFileLocation(instance, submissionFile);
			Files.deleteIfExists(submissionFile.toPath());
			Files.write(submissionFile.toPath(), "".getBytes(), StandardOpenOption.CREATE);

			final ExecutorService classifyExecutor = Executors.newCachedThreadPool();

			//Iterates through each test image, runs the classifier on the image
			for (int j = 0; j < testData.size(); j++)
			{
				final File finalSubmissionFile = submissionFile;
				final int finalJ = j;
				classifyExecutor.execute(() -> classifyImage(instance, testData, finalSubmissionFile, finalJ));
			}

			//Awaits on classification of all images in the testing set (timeout 20 minutes)
			classifyExecutor.shutdown();
			classifyExecutor.awaitTermination(20, TimeUnit.MINUTES);

			parallelAwarePrintln(instance, "\n Done.");
		}
		catch (@NotNull ClassifierException | IOException | InterruptedException e)
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
	private static void evaluateClassifier(final @NotNull IClassifier instance, final @NotNull GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
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
	 * Select the correct submission filename based on the current classifier
	 *
	 * @param instance           The current classifier instance
	 * @param tempSubmissionFile
	 * @return
	 * @throws ClassifierException
	 */
	private static File setSubmissionFileLocation(final @NotNull IClassifier instance, File tempSubmissionFile) throws ClassifierException
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
	private void classifyImage(final @NotNull IClassifier instance, final @NotNull VFSListDataset<FImage> testDatasetToClassify, final @NotNull File submissionFile, final int j)
	{
		try
		{
			final FImage img = testDatasetToClassify.get(j);
			final String filename = testDatasetToClassify.getID(j);

			final ExecutorService printExecutor = Executors.newCachedThreadPool();

			final ClassificationResult<String> predicted = instance.classify(img);
			if (predicted != null)
			{
				if (consoleOutput)
				{
					printExecutor.execute(() -> ClassifierUtils.printTestProgress(instance, j, filename, predicted));
				}
				if (writeSubmissionFile)
				{
					printExecutor.execute(() -> ClassifierUtils.writeResult(submissionFile, filename, predicted));
				}
			}
			printExecutor.shutdown();
			printExecutor.awaitTermination(10, TimeUnit.SECONDS);
		}
		catch (final InterruptedException e)
		{
			e.printStackTrace();
		}
	}

}



