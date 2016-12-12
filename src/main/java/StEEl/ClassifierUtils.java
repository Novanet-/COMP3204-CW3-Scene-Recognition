package StEEl;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.text.MessageFormat;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ClassifierUtils
{

	private ClassifierUtils() {}


	public static void parallelAwarePrintln(@NotNull IClassifier instance, String out)
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


	public static void parallelAwarePrint(@NotNull IClassifier instance, String out)
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


	@Nullable
	public static <E> List<E> pickNRandomElements(@NotNull List<E> list, int n)
	{
		return pickNRandomElements(list, n, new Random());
	}


	private static <E> List<E> pickNRandomElements(@NotNull List<E> list, int n, @NotNull Random r)
	{
		int length = list.size();

		if (length < n)
			return null;

		for (int i = length - 1; i >= length - n; --i)
		{
			Collections.swap(list, i, r.nextInt(i + 1));
		}
		return list.subList(length - n, length);
	}
}
