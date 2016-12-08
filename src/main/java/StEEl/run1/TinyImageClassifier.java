package StEEl.run1;

import StEEl.IClassifier;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import java.io.Serializable;
import java.util.*;

public class TinyImageClassifier implements IClassifier
{

	public static final double[][] featureArray = {};

	//Dimension of tiny image
	protected static final float SQUARE_SIZE = 16.0F;
	//Number of nearest neighbours to considers
	private static final   int   K_VALUE     = 15;

	private DoubleNearestNeighboursExact knn;

	//The classes of the training feature vectors (array indices correspond to featureVector indices)
	private List<String> classes;


	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public final void train(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{
		classes = new ArrayList<String>();
		final List<double[]> featureVectors = new ArrayList<double[]>();

		final FeatureExtractor<DoubleFV, FImage> ve = new TinyImageFeatureExtractor(SQUARE_SIZE);
		//For each image in each class

		for (final String group : trainingSet.getGroups())
		{
			for (final FImage image : trainingSet.get(group))
			{
				// Extract feature vector
				final DoubleFV featureVector = ve.extractFeature(image);
				featureVector.normaliseFV();
				final double[] fv = featureVector.values;

				featureVectors.add(fv);
				classes.add(group);
			}
		}

		//Array of all feature vectors
		final double[][] featureVectorArray = featureVectors.toArray(featureArray);

		//New knn with training feature vectors
		knn = new DoubleNearestNeighboursExact(featureVectorArray);
	}


	/**
	 * Classify an object.
	 *
	 * @param image the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public final ClassificationResult<String> classify(final FImage image)
	{
		//Extract feature vector for image
		final FeatureExtractor<DoubleFV, FImage> vectorExtractor = new TinyImageFeatureExtractor(SQUARE_SIZE);
		final DoubleFV featureVector = vectorExtractor.extractFeature(image);
		featureVector.normaliseFV();
		final double[] featureVectorArray = featureVector.values;

		//Find k nearest neighbours
		final List<IntDoublePair> neighbours = knn.searchKNN(featureVectorArray, K_VALUE);

		//Map classes to the number of neighbours that have that class
		final Map<String, Integer> classCount = countNeighbourClasses(neighbours);

		//List of class and their count
		final Set<Map.Entry<String, Integer>> classEntries = classCount.entrySet();
		final List<Map.Entry<String, Integer>> classGuessList = new ArrayList<Map.Entry<String, Integer>>(classEntries);

		//Sort list with greatest count first
		Collections.sort(classGuessList, new TinyImageClassifier.ClassEntryComparator());

		//Confidence in result

		final double resultConfidence = classGuessList.get(0).getValue().doubleValue() / (double) K_VALUE;

		//Guessed class is first in list
		final String guessedClass = classGuessList.get(0).getKey();

		final BasicClassificationResult<String> classificationResult = new BasicClassificationResult<String>();
		classificationResult.put(guessedClass, resultConfidence);

		return classificationResult;
	}


	private Map<String, Integer> countNeighbourClasses(final Iterable<IntDoublePair> neighbours)
	{
		final HashMap<String, Integer> classCount = new HashMap<String, Integer>();
		//For all neighbours
		for (final IntDoublePair result : neighbours)
		{
			//Get neighbour class
			final String resultClass = classes.get(result.first);

			int newCount = 1;
			//Retrieve existing count for the class
			if (classCount.containsKey(resultClass))
			{
				newCount += classCount.get(resultClass);
			}

			//Add 1 to class count
			classCount.put(resultClass, newCount);

		}
		return classCount;
	}


	private static class ClassEntryComparator implements Comparator<Map.Entry<String, Integer>>, Serializable
	{

		private static final long serialVersionUID = 4971388574385290739L;


		@Override
		public final int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2)
		{
			final Integer value = o1.getValue();
			return o2.getValue().compareTo(value);
		}
	}
}
