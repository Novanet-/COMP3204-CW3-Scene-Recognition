package StEEl;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.image.FImage;

/**
 *
 */
public interface IClassifier extends Classifier<String, FImage>
{

	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet);

	public int getClassifierID() throws Exception;

}

