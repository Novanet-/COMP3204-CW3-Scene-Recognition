package StEEl;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
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
	private static final File   TESTING_DATA_DIRECTORY    = new File(CURRENT_WORKING_DIRECTORY + "/testing/");

	private VFSGroupDataset<FImage> trainingData = null;
	private VFSGroupDataset<FImage> testingData  = null;


	/**
	 *
	 */
	@SuppressWarnings("OverlyBroadCatchBlock")
	public final void runClassifiers()
	{
		try
		{
			initialiseData();
			run1Knn();
			run2Linear();
			run3Complex();
		}
		catch (final IOException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * @throws IOException
	 * @throws org.apache.commons.vfs2.FileSystemException
	 */
	private void initialiseData() throws IOException, org.apache.commons.vfs2.FileSystemException
	{
		if (!TRAINING_DATA_DIRECTORY.exists())
		{
			throw new IOException("Training data missing");
		}
		if (!TESTING_DATA_DIRECTORY.exists())
		{
			throw new IOException("Testing data missing");
		}

		this.trainingData = new VFSGroupDataset<FImage>(TRAINING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
		this.testingData = new VFSGroupDataset<FImage>(TESTING_DATA_DIRECTORY.getPath(), ImageUtilities.FIMAGE_READER);
	}


	/**
	 *
	 */
	private void run1Knn()
	{
		throw new UnsupportedOperationException();

		//create instance of classifier
		//train classifier
		//run classifier on test data
		//write results to file
	}


	/**
	 *
	 */
	private void run2Linear()
	{
		throw new UnsupportedOperationException();

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
		throw new UnsupportedOperationException();

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



