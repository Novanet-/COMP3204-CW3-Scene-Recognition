package StEEl.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.util.array.ArrayUtils;

/**
 * Extract TinyImage feature vector from image
 */
class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV,FImage>
{

	private float squareSize;


	public TinyImageFeatureExtractor(final float squareSize)
	{
		this.squareSize = squareSize;
	}


	@Override
	public final DoubleFV extractFeature(FImage object) {
		//Smallest dimension of image is the biggest the square can be
		final int size = Math.min(object.width, object.height);

		//Extract the square from centre
		final FImage center = object.extractCenter(size, size);

		//Resize image to tiny image
		final FImage small = center.process(new ResizeProcessor(squareSize, squareSize));

		//2D array to 1D vector
		return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(small.pixels)));
	}

}
