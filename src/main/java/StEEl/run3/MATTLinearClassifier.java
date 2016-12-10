package StEEl.run3;

import StEEl.AbstractClassifier;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;

import java.util.List;
import java.util.Map;


public class MATTLinearClassifier extends AbstractClassifier
{

	private LiblinearAnnotator<FImage, String> annotator = null;
	private FeatureExtractor                   extractor = null;


	public MATTLinearClassifier(int id)
	{
		super(id);
	}


	@Override
	public final void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet)
	{
		final GroupedDataset<String, ListDataset<LocalFeatureList<Keypoint>>, LocalFeatureList<Keypoint>> gds = new MapBackedDataset<>();
		LocalFeatureList<Keypoint> featureList = null;

		final Engine<Keypoint, FImage> engine = new DoGSIFTEngine();
		final ListDataset<LocalFeatureList<Keypoint>> features = new ListBackedDataset<>();

		for (final Map.Entry<String, ListDataset<FImage>> stringListDatasetEntry : trainingSet.entrySet())
		{

			for (final FImage image : stringListDatasetEntry.getValue())
			{
				final LocalFeatureList<Keypoint> f = engine.findFeatures(image);

				features.add(f);

				if (featureList == null)
				{
					featureList = f;
				}
				else
				{
					featureList.addAll(f);
				}
			}

			gds.put(stringListDatasetEntry.getKey(), features);
		}

		final ByteKMeans kmeans = ByteKMeans.createKDTreeEnsemble(10);
		final DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(featureList);
		final ByteCentroidsResult result = kmeans.cluster(datasource);
		final HardAssigner<byte[], ?, ?> assigner = result.defaultHardAssigner();

		final BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

		final FeatureExtractor<SparseIntFV, FImage> extractor = new MATTBagOfVisualWordsExtractor(bovw);
		annotator = new LiblinearAnnotator<>(extractor, LiblinearAnnotator.Mode.MULTILABEL, SolverType.L1R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(trainingSet);
	}


	@Override
	public final ClassificationResult<String> classify(FImage object)
	{
		final List<ScoredAnnotation<String>> guess = annotator.annotate(object);

		ScoredAnnotation<String> mostConfident = guess.get(0);
		for (final ScoredAnnotation<String> score : guess)
		{
			if (score.confidence < mostConfident.confidence)
			{
				mostConfident = score;
			}
		}

		final BasicClassificationResult<String> classificationResult = new BasicClassificationResult<String>();
		classificationResult.put(mostConfident.annotation, (double) mostConfident.confidence);

		return classificationResult;
	}
}
