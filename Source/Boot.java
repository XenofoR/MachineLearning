public class Boot
{
	public static void main(String[] p_args) throws Exception
	{
		TestEnvironment environment = new TestEnvironment();
		try
		{
		environment.Init("J:/master_thesis_work/MachineLearning/test.file");
		
		environment.Run();
		}
		catch(Exception E)
		{
			Debugger.DebugPrint("Exception caught in Boot: " + E.toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		}
	}
}