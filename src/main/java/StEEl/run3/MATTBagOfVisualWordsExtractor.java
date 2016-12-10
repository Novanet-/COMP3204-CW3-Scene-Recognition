package StEEl.run3;

import org.openimaj.data.DataSource;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseFloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;

import java.util.List;

/**
 * Created by matt on 09/12/16.
 */
public class MATTBagOfVisualWordsExtractor implements FeatureExtractor<SparseIntFV, FImage> {

    DoGSIFTEngine engine;
    BagOfVisualWords bovw;

    public MATTBagOfVisualWordsExtractor(BagOfVisualWords bovw) {
        engine = new DoGSIFTEngine();
        this.bovw = bovw;
    }

    @Override
    public SparseIntFV extractFeature(FImage object) {
        //Find the keypoints
        LocalFeatureList<Keypoint> keypoints = engine.findFeatures(object);

        //Convert them into the right output format with relation to the bovw
        return bovw.aggregate(keypoints);
    }
}