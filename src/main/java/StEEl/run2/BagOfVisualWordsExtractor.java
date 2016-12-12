package StEEl.run2;

import org.jetbrains.annotations.NotNull;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.util.List;

import static StEEl.run2.LinearClassifier.PATCH_SIZE;
import static StEEl.run2.LinearClassifier.STEP;

public class BagOfVisualWordsExtractor implements FeatureExtractor<DoubleFV, FImage>
{

	private final HardAssigner<float[], float[], IntFloatPair> assigner;


	public BagOfVisualWordsExtractor(HardAssigner<float[], float[], IntFloatPair> assigner)
	{
		super();
		this.assigner = assigner;
	}


	@Override
	public final @NotNull DoubleFV extractFeature(@NotNull FImage image)
	{
		final BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<float[]>(assigner);
		final BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bagOfVisualWords, 2, 2);
		final List<LocalFeature<SpatialLocation, FloatFV>> extractedFeature = LinearClassifier.extract(image, STEP, PATCH_SIZE);
		return spatial.aggregate(extractedFeature, image.getBounds()).normaliseFV();
	}
}
