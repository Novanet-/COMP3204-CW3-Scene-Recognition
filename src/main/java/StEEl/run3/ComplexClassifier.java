package StEEl.run3;

import StEEl.AbstractClassifier;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

public class ComplexClassifier extends AbstractClassifier
{

	public ComplexClassifier(final int classifierID)
	{
		super(classifierID);
	}


	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public final void train(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{
		//Dense sift pyramid
		//Assigner assigning sift features to visual word, trainQuantiser

		//Feature extractor based on bag of visual words

		//Start training
	}


	/**
	 * Classify an object.
	 *
	 * @param object the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public final ClassificationResult<String> classify(final FImage object)
	{
		//naive bayes annotator classify
		return null;
	}

	//Generate visual words
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {

		//List of sift features from training set

		//For each image
			//Get sift features

		//Reduce set of sift features for time

		//Create a kmeans classifier with 600 categories (600 visual words)

		//Generate clusters (Visual words) from sift features.

		return null;
	}


	//Extract bag of visual words feature vector
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>
	{
		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;

		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
		{

		}

		public DoubleFV extractFeature(FImage image) {

			//Get sift features of input image


			//Bag of visual words histogram representation


			//Bag of visual words for blocks and combine


			//Return normalised feature vector
			return null;
		}
	}
}
