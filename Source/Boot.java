public class Boot
{
	public static void main(String[] p_args) throws Exception
	{
		TestEnvironment environment = new TestEnvironment();
		try
		{
			double temp = 1.0;
			while(1 + 0.5*temp != 1)
				temp *= 0.5;
			OurUtil.g_machineEpsilion = temp;
		environment.Init("J:/master_thesis_work/MachineLearning/test.file");
		
		environment.Run();
		}
		catch(Exception E)
		{
			String stack = "";
			for(int i = 0; i < E.getStackTrace().length; i++)
				stack += E.getStackTrace()[i];
			Debugger.DebugPrint("Exception caught in Boot: " + E.toString() + " Stacktrace: " + stack, Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		}
	}
}