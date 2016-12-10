package StEEl.run2;

import StEEl.AbstractClassifier;
import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.VFS;
import org.apache.tools.ant.taskdefs.Local;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.BagOfWordsFeatureExtractor;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.SparseIntFVComparison;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.feature.local.keypoints.quantised.QuantisedKeypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openrdf.query.algebra.Str;

import java.net.URL;
import java.util.*;

/**
 * Created by Matt on 2016-12-07.
 */
public class LinearClassifier extends AbstractClassifier {
    LiblinearAnnotator annotator;
    FeatureExtractor extractor;

    public LinearClassifier(int id) {
        super(id);
    }
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage>  trainingData) {
        final GroupedDataset<String, ListDataset<LocalFeatureList<Keypoint>>, LocalFeatureList<Keypoint>> gds = new MapBackedDataset();
        LocalFeatureList<Keypoint> featureList = null;

        final DoGSIFTEngine engine = new DoGSIFTEngine();

        for (String key : trainingData.keySet()) {
            ListDataset<LocalFeatureList<Keypoint>> features = new ListBackedDataset<>();

            for (FImage image : trainingData.get(key)) {
                LocalFeatureList<Keypoint> f = engine.findFeatures(image);

                features.add(f);

                if (featureList == null)
                    featureList = f;
                else
                    featureList.addAll(f);
            }

            gds.put(key, features);
        }

        final ByteKMeans kmeans = ByteKMeans.createKDTreeEnsemble(10);
        final DataSource<byte[]> datasource = new LocalFeatureListDataSource(featureList);
        final ByteCentroidsResult result = kmeans.cluster(datasource);
        final HardAssigner<byte[], ?, ?> assigner = result.defaultHardAssigner();

        final BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

        FeatureExtractor<SparseIntFV, FImage> extractor = new BagOfVisualWordsExtractor(bovw);
        annotator = new LiblinearAnnotator(extractor, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC,1, 0.00001);
        annotator.train(trainingData);
    }

    @Override
    public ClassificationResult<String> classify(FImage object) {
        List<ScoredAnnotation> guess = annotator.annotate(object);

        ScoredAnnotation mostConfident = guess.get(0);
        for (ScoredAnnotation score : guess) {
            if (score.confidence < mostConfident.confidence) {
                mostConfident = score;
            }
        }

        final BasicClassificationResult<String> classificationResult = new BasicClassificationResult<String>();
        classificationResult.put(mostConfident.annotation.toString(), mostConfident.confidence);

        return classificationResult;
    }
}
