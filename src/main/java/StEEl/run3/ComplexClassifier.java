package StEEl.run3;

import StEEl.AbstractClassifier;
import StEEl.ClassifierUtils;
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
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

public class ComplexClassifier extends AbstractClassifier {
	NaiveBayesAnnotator<FImage, String> annotator;
	private static final int CLUSTERS = 25;


	public ComplexClassifier(final int classifierID)
	{
		super(classifierID);
	}

	//Temporary function to run classifier without use of lambdas or threads to get around Exceptions being thrown
	public void NonThreadedRun(GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset, VFSListDataset<FImage> testDataset ) {
		final GroupedUniformRandomisedSampler<String, FImage> groupSampler = new GroupedUniformRandomisedSampler<>(1.0d);

		//Converts the inner image list from the VFS version to the genric version
		GroupedDataset<String, ListDataset<FImage>, FImage> trainingData = GroupSampler.sample(trainingDataset, trainingDataset.size(), false);

		final int trainingDataSize = trainingData.size();

		final int PERCENT80 = (int) Math.round(trainingDataSize * 0.08);
		final int PERCENT20 = (int) Math.round(trainingDataSize * 0.08);
		final GroupedRandomSplitter<String, FImage> trainingSplitter = new GroupedRandomSplitter<String, FImage>(trainingData, PERCENT80, PERCENT20, 0);

		GroupedDataset<String, ListDataset<FImage>, FImage> newTrainingDataset = trainingSplitter.getTrainingDataset();

		train(newTrainingDataset);
		for (FImage image : testDataset) {
			System.out.println(classify(image).getPredictedClasses());
		}
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
		final DenseSIFT denseSIFT = new DenseSIFT(4, 8);
		final PyramidDenseSIFT<FImage> pdSIFT = new PyramidDenseSIFT<FImage>(denseSIFT, 6f, 7);

		//Assigner assigning sift features to visual word, trainQuantiser
		final HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(this, trainingSet, pdSIFT);

		//Feature extractor based on bag of visual words
		final FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdSIFT, assigner);

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
	 * @param object the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public final ClassificationResult<String> classify(final FImage object) {
		return annotator.classify(object);
	}

	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(ComplexClassifier instance, Dataset<FImage> dataset, PyramidDenseSIFT<FImage> pdsift, int siftLimit) {

		//List of sift features from training set
		final List<LocalFeatureList<ByteDSIFTKeypoint>> allKeys = new ArrayList();

		//For each image
		for (FImage image : dataset) {
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

	//Generate visual words
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(ComplexClassifier instance, Dataset<FImage> dataset, PyramidDenseSIFT<FImage> pdsift) {
		return trainQuantiser(instance, dataset, pdsift, 10000);
	}


	//Extract bag of visual words feature vector
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>
	{
		final PyramidDenseSIFT<FImage> pdsift;
		final HardAssigner<byte[], float[], IntFloatPair> assigner;

		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.pdsift = pdsift;
			this.assigner = assigner;
		}

		final public DoubleFV extractFeature(FImage image) {
			//Get sift features of input image
			pdsift.analyseImage(image);

			//Bag of visual words histogram representation
			final BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

			//Bag of visual words for blocks and combine
			final BlockSpatialAggregator<byte[], SparseIntFV> spatialAggregator = new BlockSpatialAggregator<>(bovw, 2, 2);

			//Return normalised feature vector
			return spatialAggregator.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
	}
}
