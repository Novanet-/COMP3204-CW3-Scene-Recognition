package StEEl.run2;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.SpatialVectorAggregator;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.text.MessageFormat;
import java.util.concurrent.atomic.AtomicInteger;

import static StEEl.run2.LinearClassifier.PATCH_SIZE;
import static StEEl.run2.LinearClassifier.STEP;

public class BagOfVisualWordsExtractor implements FeatureExtractor<DoubleFV, FImage>
{

	private final HardAssigner<float[], float[], IntFloatPair> assigner;
	DoGSIFTEngine    engine = null;
	BagOfVisualWords bovw   = null;
	private final AtomicInteger count = new AtomicInteger(0);


	public BagOfVisualWordsExtractor(HardAssigner<float[], float[], IntFloatPair> assigner)
	{
		super();
		this.assigner = assigner;
	}


	@Override
	public final DoubleFV extractFeature(FImage image)
	{
		if (count.get() <= 1200)
		{
			System.out.println(MessageFormat.format("[2] -- Image {0} extracting feature", count.get()));
		}
		final BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<float[]>(assigner);
		final SpatialVectorAggregator spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bagOfVisualWords, 2, 2);
		count.getAndIncrement();

		return spatial.aggregate(LinearClassifier.extract(image, STEP, PATCH_SIZE), image.getBounds()).normaliseFV();

	}
}
