package StEEl.run3;

import org.jetbrains.annotations.NotNull;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.text.MessageFormat;
import java.util.concurrent.atomic.AtomicInteger;

//Extract bag of visual words feature vector
class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>
{

	private final HardAssigner<byte[], float[], IntFloatPair> assigner;
	private final AtomicInteger count = new AtomicInteger(0);


	PHOWExtractor(HardAssigner<byte[], float[], IntFloatPair> assigner)
	{
		super();
		this.assigner = assigner;
	}


	/**
	 * @param image
	 * @return
	 */
	@Override
	public final DoubleFV extractFeature(@NotNull FImage image)
	{
		if (count.get() <= 1200)
		{
			System.out.println(MessageFormat.format("[3] -- Image {0} extracting feature", count.get()));
		}
		final DenseSIFT denseSIFT = new DenseSIFT(ComplexClassifier.STEP, ComplexClassifier.BINSIZE);

		//Get sift features of input image
		denseSIFT.analyseImage(image);

		//Bag of visual words histogram representation
		final BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

		//Bag of visual words for blocks and combine
		final BlockSpatialAggregator<byte[], SparseIntFV> spatialAggregator = new BlockSpatialAggregator<>(bovw, 2, 2);

		count.getAndIncrement();

		//Return normalised feature vector
		final LocalFeatureList<ByteDSIFTKeypoint> byteKeypoints = denseSIFT.getByteKeypoints(ComplexClassifier.E_THRESHOLD);
		final Rectangle bounds = image.getBounds();
		final SparseIntFV aggregate = spatialAggregator.aggregate(byteKeypoints, bounds);
		return aggregate.normaliseFV();
	}


	@Override
	public final String toString()
	{
		return "PHOWExtractor{" + "assigner=" + assigner + ", count=" + count.get() + '}';
	}
}