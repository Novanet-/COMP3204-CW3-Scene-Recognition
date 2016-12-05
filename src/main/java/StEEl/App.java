package StEEl;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.io.IOException;

/**
 * OpenIMAJ Hello world!
 */
public class App
{


	public static void main(String[] args)
	{
		ClassifierController classifiers = new ClassifierController();
		classifiers.runClassifiers();
	}


	private void initialiseData() throws IOException
	{
		if (!trainingDataDirectory.exists())
		{
			throw new IOException("Training data missing");
		}
		if (!testingDataDirectory.exists())
		{
			throw new IOException("Testing data missing");
		}

		this.trainingData = new VFSGroupDataset<FImage>("zip:http://datasets.openimaj.org/att_faces.zip", ImageUtilities.FIMAGE_READER);
	}


	private void run1_knn()
	{
		throw new UnsupportedOperationException();
	}


	private  void run2_linear()
	{
		throw new UnsupportedOperationException();
	}


	private  void run3_complex()
	{
		throw new UnsupportedOperationException();
	}


	private void writeTestResults(File file, String imageName, String predictedImageClass)
	{
		throw new UnsupportedOperationException();
	}
}
