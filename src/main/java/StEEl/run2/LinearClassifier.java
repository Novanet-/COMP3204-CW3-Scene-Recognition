package StEEl.run2;

import StEEl.AbstractClassifier;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;

/**
 *
 */
public class LinearClassifier extends AbstractClassifier
{

	public LinearClassifier(final int classifierID)
	{
		super(classifierID);
	}


	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public void train(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{

	}


	/**
	 * Classify an object.
	 *
	 * @param object the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public ClassificationResult<String> classify(final FImage object)
	{
		return null;
	}
}
