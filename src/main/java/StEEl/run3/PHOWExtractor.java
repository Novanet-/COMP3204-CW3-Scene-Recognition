package StEEl.run3;

import org.jetbrains.annotations.NotNull;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

//Extract bag of visual words feature vector
class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>
{

	private final HardAssigner<byte[], float[], IntFloatPair> assigner;


	PHOWExtractor(HardAssigner<byte[], float[], IntFloatPair> assigner)
	{
		super();
		this.assigner = assigner;
	}


	@Override
	public final DoubleFV extractFeature(@NotNull FImage image)
	{
		final DenseSIFT denseSIFT = new DenseSIFT(ComplexClassifier.STEP, ComplexClassifier.BINSIZE);

		//Get sift features of input image
		denseSIFT.analyseImage(image);

		//Bag of visual words histogram representation
		final BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

		//Bag of visual words for blocks and combine
		final BlockSpatialAggregator<byte[], SparseIntFV> spatialAggregator = new BlockSpatialAggregator<>(bovw, 2, 2);

		//Return normalised feature vector
		return spatialAggregator.aggregate(denseSIFT.getByteKeypoints(ComplexClassifier.E_THRESHOLD), image.getBounds()).normaliseFV();
	}
}
