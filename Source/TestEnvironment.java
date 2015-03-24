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
	int m_depth, m_trees, m_features, m_testType, m_numTests;
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
		String[][] activeResults = new String[m_numTests][2];
		String[][] supervisedResults = new String[m_numTests][2];
		CreateDataStructure(m_inputPath + m_test);
		try
		{
			m_evaluator = new Evaluation(m_structure);
		}
		catch(Exception E)
		{
			Debugger.DebugPrint("Exception caught in ProcessFile: " + E.toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			return;
		}
		
		if(m_testType == 1 || m_testType == 3)
		{
			m_activeForest = new ActiveForest();
			m_activeForest.setNumTrees(m_trees);
			m_activeForest.setMaxDepth(m_depth);
			//m_activeForest.setDontCalculateOutOfBagError(true);
			

			Instances[] smallerSet = SplitDataStructure(m_structure, 0.1f);
			Instances[] test = SplitDataStructure(smallerSet[0], m_alSplitPercentage);
			for(int i = 0; i < m_numTests; i++)
			{
				//m_evaluator.crossValidateModel(m_activeForest, m_structure, 10, new Random());
				//activeResults[0] = m_evaluator.toSummaryString();

				try
				{
					Random ran = new Random();
					m_activeForest.setSeed(ran.nextInt());
					m_activeForest.buildClassifier(test[0], test[1]);
				}
				catch(Exception E)
				{
					StackTraceElement[] dawdadwadsada = E.getStackTrace();
					Debugger.DebugPrint("Exception caught in ProcessFile: " + E.toString() + "stacktrace: " + dawdadwadsada.toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
				}
				activeResults[i][1] = m_activeForest.toString();
				if(Utilities.g_clusterAnalysis)
					activeResults[i][1] += ClusterAnalysisToString();

			}
		}
		else if(m_testType == 2 || m_testType == 3)
		{

			m_supervisedForest = new SupervisedForest();
			m_supervisedForest.setDebug(true);
			m_supervisedForest.setPrintTrees(true);
			m_supervisedForest.setNumTrees(m_trees);
			m_supervisedForest.setMaxDepth(m_depth);
			
			
			for(int i = 0; i < m_numTests; i++)
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
	
	private String ClusterAnalysisToString()
	{
		String retString = "\n";
		
		retString += " \t" + ("====Purity and Variance difference of leafs====" + "\n");
		Vector<Vector<double[]>> purityVardiff = m_activeForest.GetPurityAndVardiff();
		double meanPurity = 0.0;
		double meanVarDiff = 0.0;
		for(int i = 0; i < purityVardiff.size(); i++)
		{
			retString += " \t" + ("Tree" + i + ": ");
			for(int j = 0; j < purityVardiff.get(i).size() - 1; j++)
			{
				retString += " \t" + ("Purity: " + purityVardiff.get(i).get(j)[0] + " VarianceDiff: " + purityVardiff.get(i).get(j)[1] + " || ");

			}
			retString += " \t" + ("\n");

		}
		
		retString += " \t" + ("===Mean Purity of Forest====" + "\n");
		retString += " \t" + ("" + purityVardiff.lastElement().lastElement()[0] + "\n");
		
		retString += " \t" + ("===Mean Variance Difference of Forest====" + "\n");
		retString += " \t" + ("" + purityVardiff.lastElement().lastElement()[1] + "\n");
		
		double[] meanCorrCov = m_activeForest.CalculateCorrelationPercentage();
		retString += " \t" + ("===Mean Correlation and Covariance of Forest====" + "\n");
		retString += " \t" + ("Correlation: " + meanCorrCov[0] + " Covariance: " + meanCorrCov[1] + "\n");
		
		double meanRandIndex = m_activeForest.CalculateRandIndex();
		retString += " \t" + ("===Mean Rand Index for Forest====" + "\n");
		retString += " \t" + ("" + meanRandIndex + "\n");
		
		return retString;
	}
	
	private String ForestInfoToString()
	{
		String retString = "\n";
		
		return retString;
	}
	
	private void WriteResultFile(String[][] p_activeRes, String[][] p_supervisedRes) throws Exception
	{
		SimpleDateFormat timeAndDate = new SimpleDateFormat("dd-MMM-yyyy HH-mm-ss");
		Calendar cal = Calendar.getInstance();
		String target = m_outputPath +timeAndDate.format(cal.getTime())+ " "+ m_currentTest;
		
			try
			{
			Writer w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(target), "utf-8"));
			w.write("Dataset: " + m_test + "\n");
			w.write("Splitlevel for active learning: " + m_alSplitPercentage + "\n");
			for(int test = 0; test < m_numTests; test++)
			{
				if(m_testType == 1 || m_testType == 3)
				{
					w.write("TestType: Active Test: " + test  + "\n\n");
					w.write("\t" +"====Crossvalidation results==== " + "\n\t" +p_activeRes[test][0] + "\n");
					w.write("\t" +"====Training results====" + "\n\t"+ p_activeRes[test][1] + "\n");
					
					
					
					w.write("\n");
					if(test == m_numTests-1)
					{
						w.write("\t" +"====Instances used as labeled====" +  "\n");
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
					}
				}
				else if(m_testType == 2 || m_testType == 3)
				{
					w.write("TestType: Supervised" + "\n\n" );
					w.write("\t" +"====Crossvalidation results==== " +p_supervisedRes[test][0] + "\n");
					w.write("\t" +"====Training results====" + "\n"+ p_supervisedRes[test][1] + "\n");
				}
			}
			w.close();
			}
			catch(Exception E)
			{
				Debugger.DebugPrint("Exception caught in ProcessFile: " + E.toString()+ "Stacktrace: " + E.getStackTrace().toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			}
		
		
	}
	private void ProcessFile(Path p_path) throws IOException
	{
		
		try (Scanner scanner = new Scanner(p_path))
		{
			while(scanner.hasNextLine())
				ProcessLine(scanner.nextLine());
			scanner.close();
		}
		catch(Exception E)
		{
			Debugger.DebugPrint("Exception caught: " + E.toString() + " Stacktrace: " + E.getStackTrace().toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
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
			case("NumTests"):
				m_numTests = scanner.nextInt();
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
			case("ClusterAnalysis"):
				Utilities.g_clusterAnalysis = scanner.nextBoolean();
				break;
			case("DebugLevel"):
				String temp = scanner.next();
				Utilities.g_debug = true;
				if(temp.equals("NONE") == true)
				{
					Debugger.Init(Debugger.g_debug_NONE, null);
					Utilities.g_debug = false;
				}
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
		try
		{
		File file = new File(p_file);
		m_loader.setFile(file);
		
		m_structure = m_loader.getStructure();
				
		m_structure.setClassIndex(m_structure.numAttributes() - 1);
		}
		catch(Exception E)
		{
			Debugger.DebugPrint("Exception caught: " + E.toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		}
		

	}
	
	/**Splits the datastructure into one labled part and one unlabled part */
	private Instances[] SplitDataStructure(Instances p_structure, float p_splitLevel) 
	{
		Instances[] returnStructure = new Instances[2];
		
		Instances tempStructure = new Instances(p_structure); //Need a temporary structure so that we can remove instances that have been selected
		
		int numLabled = (int)(p_splitLevel * p_structure.numInstances());
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
			tempStructure.delete(j);
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
