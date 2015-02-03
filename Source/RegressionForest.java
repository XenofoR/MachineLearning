import java.io.File;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
public abstract class RegressionForest 
{
	protected int m_maxDepth, m_numTrees, m_features;
	protected ArffLoader m_loader;
	protected Instances m_structure;
	protected RandomForest m_forest;
	protected Evaluation m_evaluator;
	public RegressionForest(int p_maxDepth, int p_numTrees, int p_features)
	{
		m_maxDepth = p_maxDepth;
		m_numTrees = p_numTrees;
		m_features = p_features;
		
		m_forest = new RandomForest();
	}
	
	public String GetResult()
	{
		return "hai";
	}
	
	abstract void Train(String p_data) throws Exception; 
	
	abstract double[] Run(String p_data) throws Exception;
	
	protected void ReadFile(String p_file) throws Exception
	{
		m_loader.setFile(new File(p_file));
		
		m_structure = m_loader.getStructure();
		
		m_structure.setClassIndex(m_structure.numAttributes() - 1);
		
		m_evaluator = new Evaluation(m_structure);
	}
	
	
	
}
