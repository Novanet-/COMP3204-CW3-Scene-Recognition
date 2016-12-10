package StEEl.run1;

import StEEl.AbstractClassifier;
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

public class TinyImageClassifier extends AbstractClassifier
{

	private static final double[][] featureArray = {};

	//Dimension of tiny image
	private static final float SQUARE_SIZE = 16.0F;
	//Number of nearest neighbours to considers
	private static final int   K_VALUE     = 15;

	private DoubleNearestNeighboursExact knn = null;

	//The classes of the training feature vectors (array indices correspond to featureVector indices)
	private List<String> classes = null;


	public TinyImageClassifier(final int classifierID)
	{
		super(classifierID);
	}


	/**
	 * Train the classifier with a training set
	 * <p>
	 * Extracts a "tiny image" from the image as a feature vector, and adds this to a KNN pool for use in classification
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
	 * <p>
	 * Uses KNN against the training set to guess a class for the image
	 *
	 * @param image the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public final ClassificationResult<String> classify(final FImage image)
	{
		//Create a tiny image feature extractor
		final FeatureExtractor<DoubleFV, FImage> vectorExtractor = new TinyImageFeatureExtractor(SQUARE_SIZE);

		//Extract the "tiny image" of the image as a 1D feature vector
		final DoubleFV featureVector = vectorExtractor.extractFeature(image);
		featureVector.normaliseFV();
		final double[] featureVectorArray = featureVector.values;

		//Find k nearest neighbours (match the images "tiny image" with all the "tiny images" in the training set
		final List<IntDoublePair> neighbours = knn.searchKNN(featureVectorArray, K_VALUE);

		//Count the amount of each image class that are neighbours to the current image
		final Map<String, Integer> classCount = countNeighbourClasses(neighbours);

		//Het the lsit of image classes, and their occurence amount in the knn search
		final Set<Map.Entry<String, Integer>> classEntries = classCount.entrySet();
		final List<Map.Entry<String, Integer>> classGuessList = new ArrayList<Map.Entry<String, Integer>>(classEntries);

		//Sort list of class appearances in descending order
		Collections.sort(classGuessList, new TinyImageClassifier.ClassEntryComparator());

		//The percentage of the K neighbours which were members of the class which gained a plurality, the higher the proportion of the neighbours which are the winning class,
		//the higher the confidence of that guess
		final double resultConfidence = classGuessList.get(0).getValue().doubleValue() / (double) K_VALUE;

		//Retrieve the name of the winning class
		final String guessedClass = classGuessList.get(0).getKey();

		//Store the result of the classification
		final BasicClassificationResult<String> classificationResult = new BasicClassificationResult<String>();
		classificationResult.put(guessedClass, resultConfidence);

		return classificationResult;
	}


	private Map<String, Integer> countNeighbourClasses(final Iterable<IntDoublePair> neighbours)
	{
		//Initialise the class:count map
		final HashMap<String, Integer> classCount = new HashMap<String, Integer>();

		//For every neighbour found in the search
		for (final IntDoublePair result : neighbours)
		{
			//Get the neighbour class
			final String resultClass = classes.get(result.first);

			int newCount = 1;

			//Retrieve existing count for the class
			if (classCount.containsKey(resultClass))
			{
				newCount += classCount.get(resultClass);
			}

			//Add 1 to the class count
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
