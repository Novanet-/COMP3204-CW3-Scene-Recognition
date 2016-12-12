package StEEl;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.text.MessageFormat;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

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


	public static @Nullable <E> List<E> pickNRandomElements(@NotNull List<E> list, int n)
	{
		return pickNRandomElements(list, n, new Random());
	}


	private static <E> List<E> pickNRandomElements(@NotNull List<E> list, int n, @NotNull Random r)
	{
		final int length = list.size();

		if (length < n)
		{
			return null;
		}

		for (int i = length - 1; i >= (length - n); --i)
		{
			Collections.swap(list, i, r.nextInt(i + 1));
		}
		return list.subList(length - n, length);
	}


	/**
	 * Output assembler for console output
	 *
	 * @param instance  The current classifier instance
	 * @param j
	 * @param file
	 * @param predicted
	 */
	static void printTestProgress(final @NotNull IClassifier instance, final int j, final String file, final @Nullable ClassificationResult<String> predicted)
	{
		if (predicted != null)
		{
			final String[] classes = getPredictedClassesArray(predicted);

			buildAndPrintProgressString(instance, j, file, classes);

		}
	}


	private static @NotNull String[] getPredictedClassesArray(final @NotNull ClassificationResult<String> predicted)
	{
		final Set<String> predictedClasses = predicted.getPredictedClasses();
		return predictedClasses.toArray(new String[predictedClasses.size()]);
	}


	/**
	 * String builder for console output, using image name and predicted class
	 *
	 * @param instance The current classifier instance
	 * @param j
	 * @param file
	 * @param classes
	 */
	private static void buildAndPrintProgressString(final @NotNull IClassifier instance, final int j, final String file, final @NotNull String[] classes)
	{
		final StringBuilder sb = new StringBuilder();
		sb.append(file).append(' ');

		for (final String cls : classes)
		{
			sb.append(cls);
			sb.append(' ');
		}
		sb.append(System.lineSeparator());
		try
		{
			sb.append('\r').append('[').append(instance.getClassifierID()).append("] -- ");
		}
		catch (ClassifierException e)
		{
			e.printStackTrace();
		}

		System.out.print(sb);
	}


	/**
	 * Writes a line of the submission file, using the name of the image, and its predicted class
	 *
	 * @param file
	 * @param imageName
	 * @param predictedImageClasses
	 */
	static void writeResult(@NotNull File file, String imageName, @Nullable ClassificationResult<String> predictedImageClasses)
	{
		if (predictedImageClasses != null)
		{
			final String[] classes = getPredictedClassesArray(predictedImageClasses);

			try
			{
				final StringBuilder sb = new StringBuilder();
				sb.append(imageName).append(' ');
				for (final String cls : classes)
				{
					sb.append(cls);
					sb.append(' ');
				}
				sb.append(System.lineSeparator());
				Files.write(file.toPath(), sb.toString().getBytes(), StandardOpenOption.APPEND);
			}
			catch (final @NotNull IOException e)
			{
				e.printStackTrace();
			}
		}
		else
		{
		}
	}
}
