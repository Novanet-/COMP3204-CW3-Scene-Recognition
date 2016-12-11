package StEEl;

/**
 * OpenIMAJ Hello world!
 */
class App
{

	private static final boolean writeSubmissionFileArg = true;
	private static final boolean consoleOutputArg       = true;


	private App() {}


	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		final ClassifierController classifiers = new ClassifierController();
		//Parameters might be changed to console arguments later, for now they are both true, console and file output is enabled
		classifiers.runClassifiers(consoleOutputArg, writeSubmissionFileArg);
	}

}
