import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.net.URI;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TestEnvironment {
	
	private RegressionForest m_supervisedForest;
	private RegressionForest m_activeForest;
	int m_depth, m_trees, m_features, m_testType, m_testSize;
	String m_test;
	String m_inputPath, m_outputPath;
	public TestEnvironment()
	{
		
	}
	
	public void Init(String p_testFile) throws IOException
	{
		Path path = FileSystems.getDefault().getPath(p_testFile);

		ProcessFile(path);
		
	}
	
	public void Run() throws Exception
	{
		if(m_testType == 1)
		{
			m_activeForest = new ActiveForest(m_depth, m_trees, m_features);
			for(int i = 0; i < m_testSize; i++)
				m_activeForest.Train(m_inputPath + m_test);
		}
		else if(m_testType == 2)
		{
			m_supervisedForest = new SupervisedForest(m_depth, m_trees, m_features);
			for(int i = 0; i < m_testSize; i++)
				m_supervisedForest.Train(m_inputPath + m_test);
		}
		else if(m_testType ==3)
		{
			m_activeForest = new ActiveForest(m_depth, m_trees, m_features);
			m_supervisedForest = new SupervisedForest(m_depth, m_trees, m_features);
			for(int i = 0; i < m_testSize; i++)
			{
				m_supervisedForest.Train(m_inputPath + m_test);
				m_activeForest.Train(m_inputPath + m_test);
			}
		}
		else
			System.out.println("Invalid testType: " + m_testType);
				
	}
	
	
	private void ProcessFile(Path p_path) throws IOException
	{
		
		try (Scanner scanner = new Scanner(p_path))
		{
			while(scanner.hasNextLine())
				ProcessLine(scanner.nextLine());
			scanner.close();
		}
	}
	private void ProcessLine(String p_line)
	{

		//http://www.javapractices.com/topic/TopicAction.do?Id=42s
		Scanner scanner  = new Scanner(p_line);
		scanner.useDelimiter("=");
		if(scanner.hasNext())
		{
			String id = scanner.next();
			switch(id)
			{
			case("MaxDepth"):
				m_depth = scanner.nextInt();
				break;
			case("NumTrees"):
				m_trees = scanner.nextInt();
				break;
			case("NumFeatures"):
				m_features = scanner.nextInt();
				break;
			case("Sets"):
				m_testSize = scanner.nextInt();
				break;
			case("InputPath"):
				m_inputPath = scanner.next();
				break;
			case("OutputPath"):
				m_outputPath = scanner.next();
				break;
			case("Files"):

				m_test = scanner.next();
				scanner.useDelimiter("=");
				break;
			case("TestType"):
				m_testType = scanner.nextInt();
				break;
			default:
				System.out.println("Bad line found in test file: " + id);
				break;
			}
		}
		scanner.close();
		
	}

}
