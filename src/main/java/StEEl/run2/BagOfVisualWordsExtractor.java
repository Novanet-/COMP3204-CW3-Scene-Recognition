package StEEl.run2;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.SpatialVectorAggregator;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

import static StEEl.run2.LinearClassifier.PATCH_SIZE;
import static StEEl.run2.LinearClassifier.STEP;


class BagOfVisualWordsExtractor implements FeatureExtractor<DoubleFV, FImage>
{

	private final HardAssigner<float[], float[], IntFloatPair> assigner;
	DoGSIFTEngine    engine = null;
	BagOfVisualWords bovw   = null;


	public BagOfVisualWordsExtractor(BagOfVisualWords bovw, HardAssigner<float[], float[], IntFloatPair> assigner)
	{
		//        engine = new DoGSIFTEngine();
		this.assigner = assigner;
		//        this.bovw = bovw;
	}


	@Override
	public final DoubleFV extractFeature(FImage image)
	{
		final BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<float[]>(assigner);
		final SpatialVectorAggregator spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bagOfVisualWords, 2, 2);
		return spatial.aggregate(extract(image, STEP, PATCH_SIZE), image.getBounds()).normaliseFV();
	}


	/**
	 * Extract patches regarding to STEP and PATCH_SIZE.
	 *
	 * @param image      The image to extract features from.
	 * @param step       The step size.
	 * @param patch_size The size of the patches.
	 */
	private static List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float patch_size)
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
}
