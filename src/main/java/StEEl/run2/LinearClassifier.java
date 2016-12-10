package StEEl.run2;

import StEEl.AbstractClassifier;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

/**
 *
 */
public class LinearClassifier extends AbstractClassifier
{

	// Clustering parameters
	public static int CLUSTERS              = 500;
	public static int IMAGES_FOR_VOCABULARY = 10;

	// Patch parameters
	public static float STEP       = 8;
	public static float PATCH_SIZE = 12;

	private LiblinearAnnotator<FImage, String> annotator;


	public LinearClassifier(final int classifierID)
	{
		super(classifierID);
	}


	/**
	 * Train the classifier with a training set
	 *
	 * @param trainingSet
	 */
	@Override
	public void train(final GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{
		// build vocabulary using images from all classes.
		GroupedRandomSplitter<String, FImage> rndspl = new GroupedRandomSplitter<String, FImage>(trainingSet, IMAGES_FOR_VOCABULARY, 0, 0);
		HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(rndspl.getTrainingDataset());

		// create FeatureExtractor.
		BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
		BagOfVisualWordsExtractor extractor = new BagOfVisualWordsExtractor(bovw, assigner);

		// Create and train a linear classifier.
		System.out.println("Start training...");
		annotator = new LiblinearAnnotator<FImage, String>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(trainingSet);

		System.out.println("Training finished.");
	}


	/**
	 * Classify an object.
	 *
	 * @param image the object to classify.
	 * @return classes and scores for the object.
	 */
	@Override
	public ClassificationResult<String> classify(final FImage image)
	{
		return annotator.classify(image);
	}


	/**
	 * Build a HardAssigner based on k-means ran on randomly picked patches from images.
	 *
	 * @param sample The dataset to use for creating the HardAssigner.
	 */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample)
	{
		List<float[]> allkeys = new ArrayList<float[]>();

		// extract patches
		for (FImage image : sample)
		{
			List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extract(image, STEP, PATCH_SIZE);
			System.out.println(sampleList.size());

			for (LocalFeature<SpatialLocation, FloatFV> lf : sampleList)
			{
				allkeys.add(lf.getFeatureVector().values);
			}
		}

		// Instantiate CLUSTERS-Means.
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
		float[][] data = allkeys.toArray(new float[][] {});

		// Clustering using K-means.
		System.out.println("Start clustering.");
		FloatCentroidsResult result = km.cluster(data);
		System.out.println("Clustering finished.");

		return result.defaultHardAssigner();
	}


	/**
	 * Extract patches regarding to STEP and PATCH_SIZE.
	 *
	 * @param image      The image to extract features from.
	 * @param step       The step size.
	 * @param patch_size The size of the patches.
	 */
	public static List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float patch_size)
	{
		List<LocalFeature<SpatialLocation, FloatFV>> areaList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

		// Create patch positions
		RectangleSampler rect = new RectangleSampler(image, step, step, patch_size, patch_size);

		// Extract feature from position r.
		for (Rectangle r : rect)
		{
			FImage area = image.extractROI(r);

			//2D array to 1D array
			float[] vector = ArrayUtils.reshape(area.pixels);
			FloatFV featureV = new FloatFV(vector);
			//Location of rectangle is location of feature
			SpatialLocation sl = new SpatialLocation(r.x, r.y);

			//Generate as a local feature for compatibility with other modules
			LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl, featureV);

			areaList.add(lf);
		}

		return areaList;
	}
}
