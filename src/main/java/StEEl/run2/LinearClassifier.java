package StEEl.run2;

import StEEl.AbstractClassifier;
import StEEl.ClassifierUtils;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.split.TrainSplitProvider;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

public class LinearClassifier extends AbstractClassifier
{

	public static final  float[][] A                     = {};
	// Patch parameters
	public static final float STEP       = 8.0F;
	public static final float PATCH_SIZE = 12.0F;
	// Clustering parameters
	private static final int       CLUSTERS              = 500;
	private static final int       IMAGES_FOR_VOCABULARY = 10;
	private LiblinearAnnotator<FImage, String> annotator = null;


	/**
	 * @param classifierID
	 */
	public LinearClassifier(final int classifierID)
	{
		super(classifierID);
	}


	/**
	 * @param trainingSet
	 */
	@Override
	public final void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{
		// build vocabulary using images from all classes.
		final TrainSplitProvider<GroupedDataset<String, ListDataset<FImage>, FImage>> rndspl = new GroupedRandomSplitter<String, FImage>(trainingSet, IMAGES_FOR_VOCABULARY, 0, 0);
		final HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(this, rndspl.getTrainingDataset());

		// create FeatureExtractor.
		//		final BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
		final BagOfVisualWordsExtractor extractor = new BagOfVisualWordsExtractor(assigner);

		// Create and train a linear classifier.
		ClassifierUtils.parallelAwarePrintln(this, "Start training...");
		annotator = new LiblinearAnnotator<FImage, String>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(trainingSet);

		ClassifierUtils.parallelAwarePrintln(this, "Training finished.");
	}


	/**
	 * Build a HardAssigner based on k-means ran on randomly picked patches from images.
	 *
	 * @param instance
	 * @param sample   The dataset to use for creating the HardAssigner.
	 */
	private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(final LinearClassifier instance, Dataset<FImage> sample)
	{
		final List<float[]> allkeys = new ArrayList<float[]>();

		int count = 0;

		// extract patches
		for (final FImage image : sample)
		{
			ClassifierUtils.parallelAwarePrintln(instance, MessageFormat.format("Extracting RoI areas Image {0}", count));

			final List<LocalFeature<SpatialLocation, FloatFV>> allPatches = extract(image, STEP, PATCH_SIZE);
			final List<LocalFeature<SpatialLocation, FloatFV>> sampleList = ClassifierUtils.pickNRandomElements(allPatches, 50);

			ClassifierUtils.parallelAwarePrintln(instance, String.valueOf(sampleList.size()));

			for (final LocalFeature<SpatialLocation, FloatFV> lf : sampleList)
			{
				allkeys.add(lf.getFeatureVector().values);
			}
			count++;
		}

		// Instantiate CLUSTERS-Means.
		final FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
		final float[][] data = allkeys.toArray(A);

		// Clustering using K-means.
		ClassifierUtils.parallelAwarePrintln(instance, "Start clustering.");
		final FloatCentroidsResult result = km.cluster(data);
		ClassifierUtils.parallelAwarePrintln(instance, "Clustering finished.");

		return result.defaultHardAssigner();
	}


	/**
	 * Extract patches regarding to STEP and PATCH_SIZE.
	 *
	 * @param image      The image to extract features from.
	 * @param step       The step size.
	 * @param patch_size The size of the patches.
	 */
	static List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float patch_size)
	{
		final List<LocalFeature<SpatialLocation, FloatFV>> areaList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

		// Create patch positions
		final RectangleSampler rect = new RectangleSampler(image, step, step, patch_size, patch_size);

		// Extract feature from position r.
		for (final Rectangle r : rect)
		{

			final FImage area = image.extractROI(r);

			//2D array to 1D array
			final float[] vector = ArrayUtils.reshape(area.pixels);
			final FloatFV featureV = new FloatFV(vector);
			//Location of rectangle is location of feature
			final SpatialLocation sl = new SpatialLocation(r.x, r.y);

			//Generate as a local feature for compatibility with other modules
			final LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl, featureV);

			areaList.add(lf);
		}

		return areaList;
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
		return annotator.classify(image);
	}
}
