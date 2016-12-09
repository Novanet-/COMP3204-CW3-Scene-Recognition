package StEEl;

import StEEl.run1.TinyImageClassifier;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class ClassifierController
{

	private static final String CURRENT_WORKING_DIRECTORY = System.getProperty("user.dir");
	private static final File   RUN_1_FILE                = new File(CURRENT_WORKING_DIRECTORY + "/run1.txt");
	private static final File   RUN_2_FILE                = new File(CURRENT_WORKING_DIRECTORY + "/run2.txt");
	private static final File   RUN_3_FILE                = new File(CURRENT_WORKING_DIRECTORY + "/run3.txt");
	private static final File   TRAINING_DATA_DIRECTORY   = new File(CURRENT_WORKING_DIRECTORY + "/training/");
	//	private static final File   TRAINING_DATA_DIRECTORY   = new File("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\training.zip");
	private static final File   TESTING_DATA_DIRECTORY    = new File(CURRENT_WORKING_DIRECTORY + "/testing/");
	//	private static final File   TESTING_DATA_DIRECTORY    = new File("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\testing.zip");

	private GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset;
	private VFSListDataset<FImage>                                    testDataset;
	private VFSListDataset<FImage>                                    controlDataset;


	/**
	 *
	 */
	@SuppressWarnings("OverlyBroadCatchBlock")
	public final void runClassifiers()
	{
		try
		{
			initialiseData();
			run1Knn(new TinyImageClassifier());
			//			run2Linear();
			//			run3Complex();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * @throws IOException
	 * @throws FileSystemException
	 */
	private void initialiseData() throws IOException
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
		controlDataset = new VFSListDataset<FImage>(TRAINING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);

		//		trainingDataset = new VFSGroupDataset<FImage>("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\training.zip",
		//				ImageUtilities.FIMAGE_READER);
		//		testDataset = new VFSGroupDataset<FImage>("zip:D:\\Documents\\MEGA\\Uni\\COMP3204 Computer Vision\\CW3 Scene Recognition\\testing.zip", ImageUtilities.FIMAGE_READER);
	}


	/**
	 *
	 */
	private void run1Knn(IClassifier instance)
	{
		//		throw new UnsupportedOperationException();

		//create instance of classifier
		//train classifier
		//run classifier on test data
		//write results to file

		try
		{
			// Sample all data so we have a GroupedDataset<String, ListDataset<FImage>, FImage>, and not a group with a VFSListDataset.
			final GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = GroupSampler.sample(trainingDataset, trainingDataset.size(), false);
			//			final GroupedDataset<String, ListDataset<FImage>, FImage> testData = GroupSampler.sample(testDataset, testDataset.size(), false);
			ListDataset<FImage> testData = testDataset;
			ListDataset<FImage> controlData = controlDataset;

			// Loading test dataset
			if (testDataset == null)
			{
				System.out.println("Error loading trainig set.");
				return;
			}

//			//TODO: Testing, remove when done
//			testDataset = controlDataset;

//			final int size = testDataset.size();

			System.out.println("Training set size = " + trainingData.size());
			System.out.println("Test set size = " + testData.size());
			System.out.println("Control set size = " + controlData.size());

			System.out.println("Training dataset loaded. Staring training...");
			instance.train(trainingData);

			System.out.println("Training complete. Now predicting... (progress on stderr, output on stout)");




//			for (int j = 0; j < testDataset.size(); j++)
//			{
//				final FImage img = testDataset.get(j);
//				final String file = testDataset.getID(j);
//
//				final ClassificationResult<String> predicted = instance.classify(img);
//				final Set<String> predictedClasses = predicted.getPredictedClasses();
//				final String[] classes = predictedClasses.toArray(new String[predictedClasses.size()]);
//
//				System.out.print(file);
//				System.out.print(" ");
//				for (final String cls : classes)
//				{
//					System.out.print(cls);
//					System.out.print(" ");
//				}
//				System.out.println();
//				System.out.printf("\r %d %%  ", Math.round(((double) (j + 1) * 100.0) / (double) size));
//
//			}
//			System.out.println("\n Done.");

//			 OpenIMAJ evaluation method.
						ClassificationEvaluator<CMResult<String>, String, FImage> evaluator;
						evaluator = new ClassificationEvaluator<CMResult<String>, String, FImage>(instance, trainingData, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

						Map<FImage, ClassificationResult<String>> guesses = evaluator.evaluate();
						CMResult<String> result = evaluator.analyse(guesses);

						System.out.println(result.getDetailReport());
		}
		catch (final RuntimeException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 *
	 */
	private void run2Linear()
	{
		throw new UnsupportedOperationException("Method not implemented");

		//create instance of classifier
		//train classifier
		//run classifier on test data
		//write results to file
	}


	/**
	 *
	 */
	private void run3Complex()
	{
		throw new UnsupportedOperationException("Method not implemented");

		//create instance of classifier
		//train classifier
		//run classifier on test data
		//write results to file
	}


	private void recordResults(File file, Map<FImage, ClassificationResult<String>> predictions)
	{
		for (final Map.Entry<FImage, ClassificationResult<String>> entry : predictions.entrySet())
		{
			writeResult(file, entry.getKey().toString(), entry.getValue().toString());
		}

	}


	/**
	 * @param file
	 * @param imageName
	 * @param predictedImageClass
	 */
	private void writeResult(File file, String imageName, String predictedImageClass)
	{
		throw new UnsupportedOperationException();
	}

}



