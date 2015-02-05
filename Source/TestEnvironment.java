import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Calendar;
import java.util.Scanner;
import java.net.URI;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
public class TestEnvironment {
	
	private RegressionForest m_supervisedForest;
	private RegressionForest m_activeForest;
	int m_depth, m_trees, m_features, m_testType, m_testSize;
	String m_test;
	String m_inputPath, m_outputPath, m_currentTest;
	public TestEnvironment()
	{
		
	}
	
	public void Init(String p_testFile) throws IOException
	{
		Path path = FileSystems.getDefault().getPath(p_testFile);
		m_currentTest = path.getFileName().toString();
		ProcessFile(path);
		
	}
	
	public void Run() throws Exception
	{
		double[] activeResults  = new double[2];
		double[] supervisedResults = new double[2];
		if(m_testType == 1)
		{
			m_activeForest = new ActiveForest(m_depth, m_trees, m_features);
			for(int i = 0; i < m_testSize; i++)
				activeResults = m_activeForest.Train(m_inputPath + m_test);
		}
		else if(m_testType == 2)
		{
			m_supervisedForest = new SupervisedForest(m_depth, m_trees, m_features);
			for(int i = 0; i < m_testSize; i++)
				supervisedResults = m_supervisedForest.Train(m_inputPath + m_test);
		}
		else if(m_testType ==3)
		{
			m_activeForest = new ActiveForest(m_depth, m_trees, m_features);
			m_supervisedForest = new SupervisedForest(m_depth, m_trees, m_features);
			for(int i = 0; i < m_testSize; i++)
			{
				supervisedResults = m_supervisedForest.Train(m_inputPath + m_test);
				activeResults = m_activeForest.Train(m_inputPath + m_test);
			}
		}
		else
		{
			System.out.println("Aborting! Invalid testType: " + m_testType);
			return;
		}
		
		WriteResultFile(activeResults, supervisedResults);
		
				
	}
	
	private void WriteResultFile(double[] p_activeRes, double[] p_supervisedRes) throws Exception
	{
		SimpleDateFormat timeAndDate = new SimpleDateFormat("dd-MMM-yyyy HH-mm-ss");
		Calendar cal = Calendar.getInstance();
		String target = m_outputPath +timeAndDate.format(cal.getTime())+ " "+ m_currentTest;
		Writer w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(target), "utf-8"));
		
		w.write("Dataset: " + m_test + "\n");
		if(m_testType == 1 || m_testType == 3)
		{
			w.write("TestType: Active"  + "\n");
			w.write("Mean absolute error(MAE): " +p_activeRes[0] + "\n");
			w.write("Root mean squared error: " + p_activeRes[1] + "\n");
		}
		else if(m_testType == 2 || m_testType == 3)
		{
			w.write("TestType: Supervised" + "\n");
			w.write("Mean absolute error(MAE): " +p_supervisedRes[0] + "\n");
			w.write("Root mean squared error: " + p_supervisedRes[1] + "\n");
		}
		w.close();
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
