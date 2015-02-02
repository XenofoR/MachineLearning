import java.io.File;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.trees.RandomForest;
public abstract class RegressionForest 
{
	protected int m_maxDepth, m_numTrees, m_features;
	protected ArffLoader m_loader;
	protected Instances m_structure;
	protected RandomForest m_forest;
	public RegressionForest(int p_maxDepth, int p_numTrees, int p_features)
	{
	
	}
	
	abstract void Train(String p_trainingData);
	
	abstract void Run(String p_data);
	
	protected void ReadFile(String p_file) throws Exception
	{
		m_loader.setFile(new File(p_file));
		
		m_structure = m_loader.getStructure();
		
		m_structure.setClassIndex(m_structure.numAttributes() - 1);
		
	}
	
	public String GetResult()
	{
		return "hai";
	}
	
}
