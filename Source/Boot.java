import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Scanner;

public class Boot
{
	public static void main(String[] p_args) throws Exception
	{
		
		try
		{
			double temp = 1.0;
			while(1 + 0.5*temp != 1)
				temp *= 0.5;
			OurUtil.g_machineEpsilion = temp;
			int i = 5;
			Scanner scanner = new Scanner(FileSystems.getDefault().getPath("J:/master_thesis_work/MachineLearning/Meta.test"));
			while(scanner.hasNextLine())	
			{
				TestEnvironment environment = new TestEnvironment();
				environment.Init("J:/master_thesis_work/MachineLearning/Tests/" + scanner.nextLine(),i++ );
				environment.Run();
				environment = null;
				System.gc();
			}
			scanner.close();
		}
		catch(Exception E)
		{
			String stack = "";
			for(int i = 0; i < E.getStackTrace().length; i++)
				stack += E.getStackTrace()[i];
			Debugger.DebugPrint("Exception caught in Boot: " + E.toString() + " Stacktrace: " + stack, Debugger.g_debug_NONE, Debugger.DebugType.CONSOLE);
		}
	}
}