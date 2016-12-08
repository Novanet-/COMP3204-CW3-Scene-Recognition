package StEEl.run3;

import StEEl.IClassifier;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;

/**
 * Created by Will on 08/12/2016.
 */
public class ComplexClassifier implements IClassifier
{

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
