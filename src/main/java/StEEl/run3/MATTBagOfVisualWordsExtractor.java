package StEEl.run3;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;


class MATTBagOfVisualWordsExtractor implements FeatureExtractor<SparseIntFV, FImage> {

    private final DoGSIFTEngine    engine;
    private final BagOfVisualWords bovw;

    public MATTBagOfVisualWordsExtractor(BagOfVisualWords bovw) {
        engine = new DoGSIFTEngine();
        this.bovw = bovw;
    }

    @Override
    public final SparseIntFV extractFeature(FImage object) {
        //Find the keypoints
        final LocalFeatureList<Keypoint> keypoints = engine.findFeatures(object);

        //Convert them into the right output format with relation to the bovw
        return bovw.aggregate(keypoints);
    }
}
