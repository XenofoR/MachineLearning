import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Calendar;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Vector;
import java.net.URI;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Random;

import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomTree;
public class TestEnvironment {
	private Loader m_loader;
	private Instances m_structure;
	private SupervisedForest m_supervisedForest;
	private ActiveForest m_activeForest;
	int m_depth, m_trees, m_features, m_testType, m_testSize;
	int[] m_labeledIndex;
	float m_alSplitPercentage;
	Evaluation m_evaluator;
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
		m_loader = new Loader();
		
	}
	
	public void Run() throws Exception
	{
		String[] activeResults = new String[2];
		String[] supervisedResults = new String[2];
		CreateDataStructure(m_inputPath + m_test);
		m_evaluator = new Evaluation(m_structure);
	
		if(m_testType == 1 || m_testType == 3)
		{
			m_activeForest = new ActiveForest();
			m_activeForest.setNumTrees(m_trees);
			m_activeForest.setMaxDepth(m_depth);
			m_activeForest.setDontCalculateOutOfBagError(true); //TODO Fix error once relevant
			//m_activeForest.SetData(m_structure);
			
			//TODO: REMOVE THIS ONCE WE ARE INTERESTED IN MORE COMPLEX DATA
			RemoveAttribute(3);
			RemoveAttribute(2);
			//END TODO
			
			Instances[] test = SplitDataStructure(m_structure);
			for(int i = 0; i < m_testSize; i++)
			{
				//m_evaluator.crossValidateModel(m_activeForest, m_structure, 10, new Random());
				//activeResults[0] = m_evaluator.toSummaryString();
				m_activeForest.buildClassifier(test[0], test[1]);
				activeResults[1] = m_activeForest.toString();
			}
		}
		else if(m_testType == 2 || m_testType == 3)
		{

			m_supervisedForest = new SupervisedForest();
			m_supervisedForest.setDebug(true);
			m_supervisedForest.setPrintTrees(true);
			m_supervisedForest.setNumTrees(m_trees);
			m_supervisedForest.setMaxDepth(m_depth);
			
			
			for(int i = 0; i < m_testSize; i++)
			{
				
			}
		}
		else
		{
			System.out.println("Aborting! Invalid testType: " + m_testType);
			return;
		}
		
		
		WriteResultFile(activeResults, supervisedResults);
		
		
	}
	
	private void WriteResultFile(String[] p_activeRes, String[] p_supervisedRes) throws Exception
	{
		SimpleDateFormat timeAndDate = new SimpleDateFormat("dd-MMM-yyyy HH-mm-ss");
		Calendar cal = Calendar.getInstance();
		String target = m_outputPath +timeAndDate.format(cal.getTime())+ " "+ m_currentTest;
		Writer w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(target), "utf-8"));
		
		w.write("Dataset: " + m_test + "\n");
		if(m_testType == 1 || m_testType == 3)
		{
			w.write("TestType: Active"  + "\n\n");
			w.write("====Crossvalidation results==== "  +p_activeRes[0] + "\n");
			w.write("====Training results====" + "\n"+ p_activeRes[1] + "\n");
			
			w.write("====Instances used as labeled====" +  "\n");
			for(int i = 0; i < m_labeledIndex.length; i ++)
			{
				for(int j = 0 ; j < 10; j++)
				{
					i++;
					if( i >= m_labeledIndex.length )
					{
						break;
					}
					w.write(" " + m_labeledIndex[i]);
				}
				w.write("\n");
			}
			
			w.write("====Purity of leafs====" + "\n");
			Vector<Vector<Double>> purity = m_activeForest.GetPurity();
			for(int i = 0; i < purity.size(); i++)
			{
				w.write("Tree" + i + ": ");
				for(int j = 0; j < purity.get(i).size(); j++)
				{
					w.write("" + purity.get(i).get(j) + "   ");
				}
				w.write("\n");
			}
			
			w.write("===Mean Correlation of Forest====" + "\n");
			w.write("" + m_activeForest.CalculateCorrelationPercentage() + "\n");
			
		}
		else if(m_testType == 2 || m_testType == 3)
		{
			w.write("TestType: Supervised" + "\n\n" );
			w.write("====Crossvalidation results==== " +p_supervisedRes[0] + "\n");
			w.write("====Training results====" + "\n"+ p_supervisedRes[1] + "\n");
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
			case("AlphaValue"):
				Utilities.g_alphaValue = Double.parseDouble(scanner.next());
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
			case("SplitLevel"):
				m_alSplitPercentage = Float.parseFloat(scanner.next());
				break;
			case("Plot"):
				Debugger.g_plot = scanner.nextBoolean();
				break;
			case("DebugLevel"):
				String temp = scanner.next();
				if(temp.equals("NONE") == true)
					Debugger.Init(Debugger.g_debug_NONE, null);
				else if(temp.equals("LOW") == true)
					Debugger.Init(Debugger.g_debug_LOW, null);
				else if(temp.equals("MEDIUM") == true)
					Debugger.Init(Debugger.g_debug_MEDIUM, null);
				else if(temp.equals("HIGH") == true)
					Debugger.Init(Debugger.g_debug_HIGH, null);
				break;
			default:
				System.out.println("Bad line found in test file: " + id);
				break;
			}
		}
		scanner.close();
		
	}
	
	private void CreateDataStructure(String p_file) throws Exception
	{
		File file = new File(p_file);
				
		m_loader.setFile(file);
				
		m_structure = m_loader.getStructure();
				
		m_structure.setClassIndex(m_structure.numAttributes() - 1);

	}
	
	/**Splits the datastructure into one labled part and one unlabled part */
	private Instances[] SplitDataStructure(Instances p_structure) 
	{
		Instances[] returnStructure = new Instances[2];
		
		Instances tempStructure = new Instances(p_structure); //Need a temporary structure so that we can remove instances that have been selected
		
		int numLabled = (int)(m_alSplitPercentage * p_structure.numInstances());
		m_labeledIndex = new int[numLabled];
		returnStructure[0] = new Instances(p_structure, numLabled);
		returnStructure[1] = new Instances(p_structure, p_structure.numInstances() - numLabled);
		
		Random ran = new Random();
		for(int i = 0; i < numLabled; i++)
		{
			int j = ran.nextInt(tempStructure.numInstances());
			Instance selected = tempStructure.get(j);
			m_labeledIndex[i] = j;
			returnStructure[0].add(selected);
			tempStructure.remove(j);
		}
		tempStructure.setClassIndex(-1); //This makes weka treat the data as unlabeled
		
		returnStructure[1] = tempStructure;
		
		return returnStructure;
	}
	
	private void RemoveAttribute(int p_index)
	{
		m_structure.deleteAttributeAt(p_index);
	}

}
