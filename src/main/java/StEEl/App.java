package StEEl;

/**
 * OpenIMAJ Hello world!
 */
public class App
{

	private App() {}


	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		final ClassifierController classifiers = new ClassifierController();
		classifiers.runClassifiers();
	}

}
