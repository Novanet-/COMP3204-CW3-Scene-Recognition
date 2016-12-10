package StEEl;

import java.text.MessageFormat;

public class ClassifierUtils
{

	private ClassifierUtils() {}


	public static void parallelAwarePrintln(IClassifier instance, String out)
	{
		try
		{
			final String newMessage = MessageFormat.format("[{0}] -- {1}", instance.getClassifierID(), out);
			System.out.println(newMessage);
		}
		catch (ClassifierException e)
		{
			e.printStackTrace();
		}
	}
		public static void parallelAwarePrint(IClassifier instance, String out)
		{
			try
			{
				final String newMessage = MessageFormat.format("[{0}] -- {1}", instance.getClassifierID(), out);
				System.out.print(newMessage);
			}
			catch (ClassifierException e)
			{
				e.printStackTrace();
			}
		}
}
