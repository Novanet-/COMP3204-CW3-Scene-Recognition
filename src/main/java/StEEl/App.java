package StEEl;

/**
 * OpenIMAJ Hello world!
 */
public class App
{

	static boolean writeSubmissionFileArg = true;
	static boolean consoleOutputArg       = true;


	private App() {}


	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		final ClassifierController classifiers = new ClassifierController();
		classifiers.runClassifiers(consoleOutputArg, writeSubmissionFileArg);
	}

}
