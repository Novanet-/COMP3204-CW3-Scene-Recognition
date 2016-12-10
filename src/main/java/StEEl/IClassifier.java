package StEEl;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.image.FImage;

/**
 *
 */
interface IClassifier extends Classifier<String, FImage>
{

	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet);

	int getClassifierID() throws ClassifierException;

}

