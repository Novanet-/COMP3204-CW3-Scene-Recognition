package StEEl.run3;

import StEEl.AbstractClassifier;
import StEEl.ClassifierUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

public class ComplexClassifier extends AbstractClassifier
{

	private static final   int                                 CLUSTERS           = 25;
	protected static final int                                 STEP               = 4;
	protected static final int                                 BINSIZE            = 8;
	private static final   int                                 DEFAULT_SIFT_LIMIT = 10000;
	protected static final float                               E_THRESHOLD        = 0.015f;
	private @Nullable      NaiveBayesAnnotator<FImage, String> annotator          = null;


	public ComplexClassifier(final int classifierID)
	{
		super(classifierID);
	}


	//Temporary function to run classifier without use of lambdas or threads to get around Exceptions being thrown
	public final void nonThreadedRun(@NotNull GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset, @NotNull VFSListDataset<FImage> testDataset)
	{
		final GroupedUniformRandomisedSampler<String, FImage> groupSampler = new GroupedUniformRandomisedSampler<>(1.0d);

		//Converts the inner image list from the VFS version to the genric version
		GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = GroupSampler.sample(trainingDataset, trainingDataset.size(), false);

		final int trainingDataSize = trainingData.size();

		final int percent80 = (int) Math.round((double) trainingDataSize * 0.08);
		final int percent20 = (int) Math.round((double) trainingDataSize * 0.08);
		final GroupedRandomSplitter<String, FImage> trainingSplitter = new GroupedRandomSplitter<String, FImage>(trainingData, percent80, percent20, 0);

		final GroupedDataset<String, ListDataset<FImage>, FImage> newTrainingDataset = trainingSplitter.getTrainingDataset();

		train(newTrainingDataset);
		for (final FImage image : testDataset)
		{
			System.out.println(classify(image).getPredictedClasses());
		}
	}


	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public final void train(final @NotNull GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{
		//Dense sift pyramid
		final DenseSIFT denseSIFT = new DenseSIFT(STEP, BINSIZE);

		//Assigner assigning sift features to visual word, trainQuantiser
		final HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(this, trainingSet, denseSIFT);

		//Feature extractor based on bag of visual words
		final FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(assigner);

		//Create Bayesian annotator
		annotator = new NaiveBayesAnnotator<>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);

		//Start training
		ClassifierUtils.parallelAwarePrintln(this, "Start training...");
		annotator.train(trainingSet);
		ClassifierUtils.parallelAwarePrintln(this, "Training finished.");
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
		return (annotator != null) ? annotator.classify(image) : null;
	}


	//Generate visual words
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(final ComplexClassifier instance, @NotNull Dataset<FImage> dataset, final @NotNull DenseSIFT pdsift)
	{
		return trainQuantiser(instance, dataset, pdsift, DEFAULT_SIFT_LIMIT);
	}


	private static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(final ComplexClassifier instance, @NotNull Dataset<FImage> dataset, final @NotNull DenseSIFT pdsift,
			final int siftLimit)
	{

		//List of sift features from training set
		final List<LocalFeatureList<ByteDSIFTKeypoint>> allKeys = new ArrayList<>();

		//For each image
		for (final FImage image : dataset)
		{
			//Get sift features
			pdsift.analyseImage(image);
			allKeys.add(pdsift.getByteKeypoints());
		}

		//Create a kmeans classifier with 600 categories (600 visual words)
		final ByteKMeans kMeans = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
		final DataSource<byte[]> dataSource = new LocalFeatureListDataSource<>(allKeys);

		//Generate clusters (Visual words) from sift features.
		ClassifierUtils.parallelAwarePrintln(instance, "Start clustering.");
		final ByteCentroidsResult result = kMeans.cluster(dataSource);
		ClassifierUtils.parallelAwarePrintln(instance, "Clustering finished.");

		return result.defaultHardAssigner();
	}

}
