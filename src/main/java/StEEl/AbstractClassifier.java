package StEEl;

public abstract class AbstractClassifier implements IClassifier
{

	private final int classifierID;


	protected AbstractClassifier(int classifierID)
	{
		this.classifierID = classifierID;
	}


	/**
	 * Gets the  identifier for the classifier
	 *
	 * @return
	 */
	@Override
	public final int getClassifierID() throws ClassifierException
	{
		if (classifierID == 0)
		{
			throw new ClassifierException("Classifier ID has not been set");
		}
		else
		{
			return classifierID;
		}
	}

}
