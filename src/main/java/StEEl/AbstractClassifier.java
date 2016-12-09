package StEEl;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;

/**
 * Created by Will on 09/12/2016.
 */
public abstract class AbstractClassifier implements IClassifier
{

	private static final int classifierID = 0;


	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public abstract void train(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet);


	/**
	 * Gets the  identifier for the classifier
	 *
	 * @return
	 */
	@Override
	public int getClassifierID() throws ClassifierException
	{
		if (classifierID == 0)
		{
			throw new ClassifierException("Classifier ID has not been set");
		}
		else
		{
			return classifierID;
		}
	}


	/**
	 * Classify an object.
	 *
	 * @param object the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public abstract ClassificationResult<String> classify(final FImage object);
}
