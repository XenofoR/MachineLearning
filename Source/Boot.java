public class Boot
{
	public static void main(String[] p_args) throws Exception
	{
		TestEnvironment environment = new TestEnvironment();
		
		environment.Init("J:/master_thesis_work/MachineLearning/test.file");
		
		environment.Run();
	}
}