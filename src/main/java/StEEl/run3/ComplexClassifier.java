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
import java.util.concurrent.atomic.AtomicReference;

public class ComplexClassifier extends AbstractClassifier {
	protected static final int                                 STEP               = 4;
	protected static final int                                 BINSIZE            = 8;
	protected static final float                               E_THRESHOLD        = 0.015f;
	private static final   int                                 CLUSTERS           = 25;
	private @Nullable NaiveBayesAnnotator<FImage, String> annotator          = null;

	public ComplexClassifier(final int classifierID) {
		super(classifierID);
	}

	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public final void train(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
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
	public final ClassificationResult<String> classify(final FImage image) {
		return annotator.classify(image);
	}

	/**
	 * Build a HardAssigner based on k-means ran on features extracted with dense SIFT
	 *
	 * @param instance
	 * @param dataset   The dataset to use for creating the HardAssigner.
	 * @param dsift The instance of dense SIFT to use for extracting features
	 */
	private static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(final @NotNull ComplexClassifier instance, @NotNull Dataset<FImage> dataset, final @NotNull DenseSIFT dsift) {

		//List of sift features from training set
		final AtomicReference<List<LocalFeatureList<ByteDSIFTKeypoint>>> allKeys = new AtomicReference<>(new ArrayList());

		//For each image
		for (FImage image : dataset) {
			//Get sift features
			dsift.analyseImage(image);
			allKeys.get().add(dsift.getByteKeypoints());
		}

		//Create a kmeans classifier with 600 categories (600 visual words)
		final ByteKMeans kMeans = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
		final DataSource<byte[]> dataSource = new LocalFeatureListDataSource<>(allKeys.get());

		//Generate clusters (Visual words) from sift features.
		ClassifierUtils.parallelAwarePrintln(instance, "Start clustering.");
		final ByteCentroidsResult result = kMeans.cluster(dataSource);
		ClassifierUtils.parallelAwarePrintln(instance, "Clustering finished.");

		return result.defaultHardAssigner();
	}
}
