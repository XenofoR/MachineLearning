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
import weka.core.pmml.jaxbbindings.OUTLIERTREATMENTMETHOD;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
public class TestEnvironment {
	private Loader m_loader;
	private Instances m_structure;
	//private ActiveForest m_supervisedForest;
	private RandomForest m_supervisedForest;
	private ActiveForest m_activeForest;
	int m_depth, m_trees, m_features, m_testType, m_numTests;
	Validator m_validator;
	String m_test;
	String m_inputPath, m_outputPath, m_currentTest;
	Philadelphiaost m_oracle;
	InstanceComparator m_comparer;
	int m_validationFolds = 1;
	float m_activeLabeled = 1;
	float m_supervisedLabeled = 1;
	float m_activeUnlabeled = 1;
	float m_trainingSize = 1;
	boolean m_folderInPlace = false;
	int m_threshold;
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
		m_validator = new Validator(m_validationFolds);
		
	}
	
	public void Run() throws Exception
	{
		double[][] supervisedMAE;
		double[][] activeMAE;
		double[][] supervisedMAPE;
		double[][] activeMAPE;
		double[][] transductionError;
		Timer t = new Timer();
		CreateDataStructure(m_inputPath + m_test);

		Instances[] spliData = SplitDataStructure(m_structure, m_trainingSize);
		m_oracle.Init(spliData[0]);
		
		supervisedMAE = new double[m_numTests][];
		supervisedMAPE = new double[m_numTests][];
		activeMAE = new double[m_numTests][];
		activeMAPE = new double[m_numTests][];
		transductionError = new double[m_numTests][];
		Random ran = new Random();
		switch(m_testType)
		{
		case 1:
			for(int i = 0; i < m_numTests; i++)
			{
				int testTimeIndex = t.StartTimer();
				Long averageActive = 0L;
				Long averageFoldTime = 0L;
				Instances[] folds = SplitDataStructure(spliData[0], m_validationFolds);
				m_validator.Init(folds); 
				
				supervisedMAE[i] = new double[folds[0].numInstances()/OurUtil.g_activeNumber];
				supervisedMAPE[i] = new double[folds[0].numInstances()/OurUtil.g_activeNumber];
				activeMAE[i] = new double[folds[0].numInstances()/OurUtil.g_activeNumber];
				activeMAPE[i] = new double[folds[0].numInstances()/OurUtil.g_activeNumber];
				transductionError[i] = new double[folds[0].numInstances()/OurUtil.g_activeNumber];
				
				String clusterString = "";
				int seed = ran.nextInt();
				for(int j = 0; j < m_validationFolds; j++)
				{
					int foldTimeIndex = t.StartTimer();
					Instances currFold = m_validator.GetTrainingSet();
					Instances[] supervised = SplitDataStructure(currFold, m_supervisedLabeled);
					Instances[] active = SplitDataStructure(currFold, m_activeLabeled);
					
					int k = 0;
					while(active[1].numInstances() > OurUtil.g_activeNumber)
					{
						if(k >= m_threshold)
							break;
						int index = t.StartTimer();
						m_supervisedForest = new RandomForest();
						m_activeForest = new ActiveForest();
						m_supervisedForest.setNumTrees(m_trees);
						m_supervisedForest.setMaxDepth(m_depth);
						m_supervisedForest.setNumFeatures(m_features);
						m_supervisedForest.setSeed(seed);
						m_activeForest.setNumTrees(m_trees);
						m_activeForest.setMaxDepth(m_depth);
						m_activeForest.setNumFeatures(m_features);
						m_activeForest.setSeed(seed);
						m_activeForest.setNumExecutionSlots(8);
						m_supervisedForest.setNumExecutionSlots(8);

						m_supervisedForest.buildClassifier(supervised[0]);
						m_activeForest.buildClassifier(active[0], active[1]);
						Instances temp = m_oracle.ConsultOracle(m_activeForest.GetOracleData());
						
						RemovePredefined(temp, active[1]);
						active[0].addAll(temp);
						supervised[0].addAll(RemoveAtRandom(temp.numInstances(), supervised[1]));
						
						m_validator.ValidateModel(m_supervisedForest);
						supervisedMAE[i][k] += m_validator.GetMAE();
						supervisedMAPE[i][k] += m_validator.GetMAPE();
						m_validator.ValidateModel(m_activeForest);
						activeMAE[i][k] += m_validator.GetMAE();
						activeMAPE[i][k] += m_validator.GetMAPE();
						
						transductionError[i][k] = m_activeForest.GetAverageTransductionError();
						if(OurUtil.g_clusterAnalysis)
							clusterString += ClusterAnalysisToString();
						k++;
						m_supervisedForest = null;
						m_activeForest = null;

						System.out.println("======= Current Fold: " + j + " k-value: " + k  + " number of unlabeled left: " + active[1].numInstances() + " ========\n");
						
						System.gc();
						
						Long activeTime = t.GetRawTime(index);
						averageActive += (activeTime / (10 * m_validationFolds));
						System.out.println("Active loop time: " + activeTime);
						t.StopTimer(index);
					}
					Long foldTime = t.GetRawTime(foldTimeIndex);
					t.StopTimer(foldTimeIndex);
					averageFoldTime += (foldTime / (m_validationFolds));
					Debugger.DebugPrint("Fold loop Time: " + foldTime, Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
					supervised = null;
					active = null;
					
				}
				String testTime = t.GetFormatedTime(testTimeIndex);
				t.StopTimer(testTimeIndex);
				for(int j = 0; j < folds[0].numInstances()/OurUtil.g_activeNumber; j++)
				{
					supervisedMAE[i][j] /= m_validationFolds*OurUtil.g_activeNumber;
					supervisedMAPE[i][j] /= m_validationFolds*OurUtil.g_activeNumber;
					
					activeMAE[i][j] /= m_validationFolds*OurUtil.g_activeNumber;
					activeMAPE[i][j] /= m_validationFolds*OurUtil.g_activeNumber;
				}
				
				String supervisedResults[] = new String[2];
				String activeResults[] = new String[4];
				String metaData[] = new String[3];
				activeResults[0] = activeResults[1]= activeResults[2] = activeResults[3] = supervisedResults[0] = supervisedResults[1] = 
						metaData[0] = metaData[1]= metaData[2] = "";

				metaData[0] = "Average active loop time: " + t.ConvertRawToFormated(averageActive) + "\n";
				metaData[1] = "Average Fold loop time: " + t.ConvertRawToFormated(averageFoldTime) + "\n";
				metaData[2] = "Total test time: " + testTime + "\n";
				
				for(int j = 0; j < supervisedMAE[0].length; j++)
				{
					supervisedResults[0] += supervisedMAE[i][j] + " ";
					activeResults[0] += activeMAE[i][j] + " ";
				}

				for(int j = 0; j < supervisedMAE[0].length; j++)
				{
					supervisedResults[1] += supervisedMAPE[i][j] + " ";
					activeResults[1] += activeMAPE[i][j] + " ";
				}
				for(int j = 0; j < transductionError[0].length; j++)
					activeResults[2] += transductionError[i][j] + " ";
				if(OurUtil.g_clusterAnalysis)
					activeResults[3] = clusterString;
				if(!m_folderInPlace)
				{
					SimpleDateFormat timeAndDate = new SimpleDateFormat("dd-MMM-yyyy HH-mm-ss");
					Calendar cal = Calendar.getInstance();
					new File(m_outputPath + "/" + timeAndDate.format(cal.getTime())).mkdir();
					m_outputPath += "/" + timeAndDate.format(cal.getTime()) + "/";
					m_folderInPlace = true;
				}
				
				WriteResultFile(activeResults, supervisedResults, metaData, i);
				folds = null;
			}
			//Start at same labeled amount, ours actively choices ders chose by dice rooloing
			break;
		case 2:
			//Stop a test run after the transduction stage
			String clusterString = "";
			Long averageFoldTime = 0L;
			Long averageTestTime = 0L;
			int seed = ran.nextInt();
			for(int i = 0; i < m_numTests; i++)
			{
				int testTimeIndex = t.StartTimer();
				Instances[] folds = SplitDataStructure(spliData[0], m_validationFolds);
				m_validator.Init(folds); 
				
				transductionError[i] = new double[m_numTests];
				
				for(int j = 0; j < m_validationFolds; j++)
				{
					int foldTimeIndex = t.StartTimer();
					Instances currFold = m_validator.GetTrainingSet();
					Instances[] active = SplitDataStructure(currFold, m_activeLabeled);
					
					m_activeForest = new ActiveForest();
						
					m_activeForest.setNumTrees(m_trees);
					m_activeForest.setMaxDepth(m_depth);
					m_activeForest.setNumFeatures(m_features);
					m_activeForest.setSeed(seed);
					m_activeForest.setNumExecutionSlots(8);

					m_activeForest.buildClassifier(active[0], active[1]);
						
					transductionError[i][j] = m_activeForest.GetAverageTransductionError();
					if(OurUtil.g_clusterAnalysis)
						clusterString += ClusterAnalysisToString();
					m_activeForest = null;
						
					System.gc();
						
					Long foldTime = t.GetRawTime(foldTimeIndex);
					t.StopTimer(foldTimeIndex);
					averageFoldTime += (foldTime / (m_validationFolds));
					Debugger.DebugPrint("Fold loop Time: " + foldTime, Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
					active = null;
				}
				folds = null;
				Long testTime = t.GetRawTime(testTimeIndex);
				t.StopTimer(testTimeIndex);
				averageTestTime += (testTime / m_numTests);
				t.StopTimer(testTimeIndex);
			}

			
			String supervisedResults[] = new String[2];
			String activeResults[] = new String[4];
			String metaData[] = new String[3];
			activeResults[0] = activeResults[1]= activeResults[2] = activeResults[3] = 
					metaData[0] = metaData[1]= metaData[2] = "";

			metaData[1] = "Average Fold time: " + t.ConvertRawToFormated(averageFoldTime) + "\n";
			metaData[2] = "Average Test time: " + t.ConvertRawToFormated(averageTestTime) + "\n";
			
			for(int i = 0; i < m_numTests; i++)
				for(int j = 0; j < transductionError[0].length; j++)
					activeResults[2] += transductionError[i][j] + " ";
			
			if(OurUtil.g_clusterAnalysis)
				activeResults[3] = clusterString;
			
			WriteResultFile(activeResults, supervisedResults, metaData, 0);
			
			break;
		case 3:
			//We do nothing but return random values-.
			break;
		default:
			break;
		}
		
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
	
	
	private void WriteResultFile(String p_activeRes[], String p_supervisedRes[], String p_metaData[], int p_testNumber) throws Exception
	{

		String target = m_outputPath + p_testNumber + ".result";
		
			try
			{
			Writer w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(target), "utf-8"));
			w.write("Dataset: " + m_test + "\n");
			w.write("Tree parameters(max depth/num trees/feature split): " + m_depth + " " + m_trees + " " + m_features + "\n");
			w.write("Test parameters(test type/training size/num folds/supervised labeled/active learning labeled): " + m_testType + " " + m_trainingSize + " " + m_validationFolds + " " + m_supervisedLabeled + " " + m_activeLabeled + "\n");

					w.write("Supervised Results: \n");
					w.write("\t" +"MAE: " + p_supervisedRes[0] + "\n");
					w.write("\t" +"MAPE: " + p_supervisedRes[1] + "\n");
					
					w.write("Active choice parameters(t/at/nc): " + m_threshold + " " + ((OurUtil.g_activeTech == OurUtil.ActiveTechnique.Random) ? "Random" : 
																								(OurUtil.g_activeTech == OurUtil.ActiveTechnique.Worst) ? "Worst" : 
																								(OurUtil.g_activeTech == OurUtil.ActiveTechnique.AllWorst) ? "AllWorst" :
																								(OurUtil.g_activeTech == OurUtil.ActiveTechnique.Ensemble) ? "Ensemble" : "NONE") + " " + OurUtil.g_activeNumber + "\n");
					w.write("Active Results: \n");
					w.write("\t" +"MAE: " + p_activeRes[0] + "\n");
					w.write("\t" +"MAPE: " + p_activeRes[1] + "\n");
					w.write("\t" + "Trans: " + p_activeRes[2] +"\n");
					if(OurUtil.g_clusterAnalysis)
						w.write(p_activeRes[3]);
					w.write("Performance: \n");
					for(int i = 0; i < p_metaData.length; i++)
						w.write("\t" + p_metaData[i]);
					
			w.write("END");
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
				break;
			case("TestType"):
				m_testType = scanner.nextInt();
				break;
			case("DataSize"):
				m_trainingSize = Float.parseFloat(scanner.next());
				break;
			case("ValidationFolds"):
				m_validationFolds = scanner.nextInt();
				break;
			case("ActiveLabeled"):
				m_activeLabeled = Float.parseFloat(scanner.next());
				break;
			case("SupervisedLabeled"):
				m_supervisedLabeled = Float.parseFloat(scanner.next());
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
				m_threshold = scanner.nextInt();
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
			case("NumberOfChoices"):
				OurUtil.g_activeNumber = scanner.nextInt();
				break;
			case("UseMahalanobis"):
				OurUtil.g_useMahalanobis = scanner.nextBoolean();
				break;
			case("UseWeightedTransduction"):
				OurUtil.g_useWeightedTransduction = scanner.nextBoolean();
				break;
			case("ForceCompleteGraph"):
				OurUtil.g_forceCompleteGraph = scanner.nextBoolean();
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
	
	/**Splits the datastructure into a number of folds */
	private Instances[] SplitDataStructure(Instances p_structure, int p_splitLevel) 
	{
		Instances tempStructure = new Instances(p_structure);
		Instances[] returnStructure = new Instances[p_splitLevel];
		for(int i = 0; i < p_splitLevel; i++)
			returnStructure[i] = new Instances(p_structure, 0);
		int instancesPerFold = p_structure.numInstances()/p_splitLevel;
		Random ran = new Random();
		for(int i = 0; i < p_splitLevel; i++)
			for(int j = 0; j < instancesPerFold; j++)
			{
				int aRandomValueSelectedByUsingARandomMethodInJava = ran.nextInt(tempStructure.numInstances());
				Instance selected = tempStructure.get(aRandomValueSelectedByUsingARandomMethodInJava);
				returnStructure[i].add(selected);
				tempStructure.delete(aRandomValueSelectedByUsingARandomMethodInJava);
			}
		
		return returnStructure;
	}
	
	private void RemovePredefined(Instances p_duplicates, Instances p_source)
	{
		for(int i = 0 ; i < p_duplicates.size(); i++)
			  for(int j = 0 ; j < p_source.size(); j++)
				  if(m_comparer.compare(p_duplicates.instance(i), p_source.instance(j)) == 0)
				  {
					  p_source.remove(j);
					  continue;
				  }
	}
	
	private Instances RemoveAtRandom(int p_numToRemove, Instances p_source)
	{
		Random ran = new Random();
		Instances retInsts = new Instances(p_source, 0);
		for(int i = 0; i < p_numToRemove; i++)
		{
			if(p_source.numInstances() < p_numToRemove-i)
				break;
			
			int index = ran.nextInt((p_source.numInstances() - 1) + 1 ) + 1 - 1;
			retInsts.add(p_source.instance(index));
			p_source.remove(index);
		}
		return retInsts;	
	}

}
