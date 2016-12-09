package StEEl.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.util.array.ArrayUtils;

/**
 * Extract TinyImage feature vector from image
 */
class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV, FImage>
{

	private float squareSize;


	public TinyImageFeatureExtractor(final float squareSize)
	{
		this.squareSize = squareSize;
	}


	@Override
	public final DoubleFV extractFeature(FImage object)
	{
		//The tiny image has to be a square, takes the smallest dimension
		final int size = Math.min(object.width, object.height);

		//Return the regtangular center of the image, extends width/2 and height/2 from the centre point
		final FImage center = object.extractCenter(size, size);

		//Scales the original image down to these smaller dimensions
		final FImage small = center.process(new ResizeProcessor(squareSize, squareSize));

		//Flatten the image vectors into a 1D array
		return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(small.pixels)));
	}

}
