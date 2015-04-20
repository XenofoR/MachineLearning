import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Calendar;
import java.util.Scanner;
import java.util.Vector;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.Random;

import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
public class TestEnvironment {
	private Loader m_loader;
	private Instances m_structure;
	//private ActiveForest m_supervisedForest;
	private RandomForest m_supervisedForest;
	private ActiveForest m_activeForest;
	int m_depth, m_trees, m_features, m_testType, m_numTests;
	float m_alSplitPercentage, m_DataSeizeOffset;
	Validator m_evaluator;
	String m_test;
	String m_inputPath, m_outputPath, m_currentTest;
	Philadelphiaost m_oracle;
	InstanceComparator m_comparer;
	public TestEnvironment()
	{
		
	}
	
	public void Init(String p_testFile) throws IOException
	{
		Path path = FileSystems.getDefault().getPath(p_testFile);
		m_currentTest = path.getFileName().toString();
		ProcessFile(path);
		m_loader = new Loader();
		m_oracle = new Philadelphiaost();
		m_comparer = new InstanceComparator();
		
	}
	
	public void Run() throws Exception
	{
		String[][] activeResults = new String[m_numTests][2];
		String[][] supervisedResults = new String[m_numTests][2];
		CreateDataStructure(m_inputPath + m_test);
		try
		{
			m_evaluator = new Validator();
		}
		catch(Exception E)
		{
			Debugger.DebugPrint("Exception caught in ProcessFile: " + E.toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			return;
		}
		
		//m_structure.deleteAttributeAt(0);
		//m_structure.deleteAttributeAt(0);
		//m_structure.deleteAttributeAt(0);
		//m_structure.deleteAttributeAt(0);
		//m_structure.deleteAttributeAt(0);
		
		Instances[] smallerSet = SplitDataStructure(m_structure, m_DataSeizeOffset);
		Instances[] test = SplitDataStructure(smallerSet[0], m_alSplitPercentage);
		Random ran = new Random();
		for(int i = 0; i < m_numTests; i++)
		{
			int seed = ran.nextInt();
			double oob = 0.0;
			if(m_testType == 2 || m_testType == 3)
			{
				Instances emptySet = new Instances(smallerSet[0], 0);
				//m_supervisedForest = new ActiveForest();
				m_supervisedForest = new RandomForest();
				m_supervisedForest.setDebug(true);
				m_supervisedForest.setPrintTrees(true);
				m_supervisedForest.setNumTrees(m_trees);
				m_supervisedForest.setMaxDepth(m_depth);
				
				
				
					try
					{
						
						m_supervisedForest.setSeed(seed);
						//m_supervisedForest.buildClassifier(test[0], emptySet);
						m_supervisedForest.buildClassifier(test[0]);
					}
					catch(Exception E)
					{
						StackTraceElement[] dawdadwadsada = E.getStackTrace();
						Debugger.DebugPrint("Exception caught in ProcessFile: " + E.toString() + "stacktrace: " + dawdadwadsada.toString(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
					}
					supervisedResults[i][1] = m_supervisedForest.toString();
				
				m_evaluator.Init(smallerSet[1], m_supervisedForest);
				m_evaluator.ValidateModel();
				System.out.println("superMAE: " + m_evaluator.GetMAE() + "\n");
				System.out.println("superMAPE: " + m_evaluator.GetMAPE() + "\n");
				oob = 0.01;
			}
			
			if(m_testType == 1 || m_testType == 3)
			{
				m_evaluator = null;
				m_evaluator = new Validator();
				
				
				m_activeForest = new ActiveForest();
				m_activeForest.setNumTrees(m_trees);
				m_activeForest.setMaxDepth(m_depth);
				//m_activeForest.setPrintTrees(true);
				m_supervisedForest = null;
	

					try
					{
						m_oracle.Init(test[1]);
						m_activeForest.setSeed(seed);
						m_activeForest.buildClassifier(test[0], test[1]);
						m_evaluator.Init(smallerSet[1], m_activeForest);
						m_evaluator.ValidateModel();
						
						System.out.println("activeMAE: " + m_evaluator.GetMAE() + "\n");
						System.out.println("activeMAPE: " + m_evaluator.GetMAPE() + "\n");
						
						Instances inst = new Instances(test[1],0);
						m_comparer.setIncludeClass(false);
						while(oob < m_evaluator.GetMAE())
						{
							m_evaluator = null;
							m_evaluator = new Validator();
							inst.clear();
							inst = m_activeForest.GetOracleData();
							inst.setClassIndex(-1);
							  for(int j = 0; j < test[1].size(); j++)
								  for(int k = 0; k < inst.size(); k++)
									  if(m_comparer.compare(inst.instance(k), test[1].instance(j)) == 0)
									  {
										  test[1].remove(j);
										  break;
									  }
							  
							  
							test[0].addAll(m_oracle.ConsultOracle(m_activeForest.GetOracleData()));
							
							m_activeForest = null;
							m_activeForest = new ActiveForest();
							m_activeForest.setNumTrees(m_trees);
							m_activeForest.setMaxDepth(m_depth);
							
							m_activeForest.setSeed(seed);
							m_activeForest.buildClassifier(test[0], test[1]);
							
							m_evaluator.Init(smallerSet[1], m_activeForest);
							m_evaluator.ValidateModel();
							
							System.out.println("activeMAE: " + m_evaluator.GetMAE() + "\n");
							System.out.println("activeMAPE: " + m_evaluator.GetMAPE() + "\n");
						}	
					}
					catch(Exception E)
					{
						StackTraceElement[] dawdadwadsada = E.getStackTrace();
						String superman ="";
						for(int il = 0; il < dawdadwadsada.length; il++)
							superman += dawdadwadsada[il] + "\n";
						Debugger.DebugPrint("Exception caught in ProcessFile: " + E.toString() + "stacktrace: " + superman, Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
					}
					activeResults[i][1] = m_activeForest.toString();
					if(OurUtil.g_clusterAnalysis)
						activeResults[i][1] += ClusterAnalysisToString();
	
				
			}
		}
		WriteResultFile(activeResults, supervisedResults);
		
		
	}
	
	private String ClusterAnalysisToString()
	{
		String retString = "\n";
		
		retString += " \t" + ("====Purity and Variance difference of leafs====" + "\n");
		Vector<Vector<double[]>> purityVardiff = m_activeForest.GetPurityAndVardiff();
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
				}
				if(m_testType == 2 || m_testType == 3)
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
			case("DataSize"):
				m_DataSeizeOffset = Float.parseFloat(scanner.next());
				break;
			case("SplitLevel"):
				m_alSplitPercentage = Float.parseFloat(scanner.next());
				break;
			case("Plot"):
				Debugger.g_plot = scanner.nextBoolean();
				break;
			case("ClusterAnalysis"):
				OurUtil.g_clusterAnalysis = scanner.nextBoolean();
				break;
			case("DebugLevel"):
				String temp = scanner.next();
				OurUtil.g_debug = true;
				if(temp.equals("NONE") == true)
				{
					Debugger.Init(Debugger.g_debug_NONE, null);
					OurUtil.g_debug = false;
				}
				else if(temp.equals("LOW") == true)
					Debugger.Init(Debugger.g_debug_LOW, null);
				else if(temp.equals("MEDIUM") == true)
					Debugger.Init(Debugger.g_debug_MEDIUM, null);
				else if(temp.equals("HIGH") == true)
					Debugger.Init(Debugger.g_debug_HIGH, null);
				break;
			case("Threshold"):
				OurUtil.g_threshold = Float.parseFloat(scanner.next());
				break;
			case("ActiveTech"):
				temp = scanner.next();
				if(temp.equals("Random") == true)
					OurUtil.g_activeTech = OurUtil.ActiveTechnique.Random;
				else if(temp.equals("Worst") == true)
					OurUtil.g_activeTech = OurUtil.ActiveTechnique.Worst;
				else if(temp.equals("AllWorst") == true)
					OurUtil.g_activeTech = OurUtil.ActiveTechnique.AllWorst;
				else if(temp.equals("Ensemble") == true)
					OurUtil.g_activeTech = OurUtil.ActiveTechnique.Ensemble;
				else
					OurUtil.g_activeTech = OurUtil.ActiveTechnique.NONE;
				break;
			case("ActiveNumber"):
				OurUtil.g_activeNumber = scanner.nextInt();
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
			RemoveDuplicates remover = new RemoveDuplicates();
			
			
			File file = new File(p_file);
			m_loader.setFile(file);
			
			m_structure = m_loader.getStructure();
			remover.setInputFormat(m_structure);
			m_structure = remover.useFilter(m_structure, remover);
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
		returnStructure[0] = new Instances(p_structure, numLabled);
		returnStructure[1] = new Instances(p_structure, p_structure.numInstances() - numLabled);
			
		
		Random ran = new Random();
		for(int i = 0; i < numLabled; i++)
		{
			int j = ran.nextInt(tempStructure.numInstances());
			Instance selected = tempStructure.get(j);
			returnStructure[0].add(selected);
			tempStructure.delete(j);
		}
		tempStructure.setClassIndex(-1); //This makes weka treat the data as unlabeled
		
		returnStructure[1] = tempStructure;
		
		
		return returnStructure;
	}
	

}
