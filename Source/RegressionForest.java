import java.io.File;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

public abstract class RegressionForest extends weka.classifiers.trees.RandomForest
{
	protected int m_maxDepth, m_numTrees, m_features;
	protected Instances m_structure;
	protected RandomForest m_forest;
	protected Evaluation m_evaluator;
	public RegressionForest(int p_maxDepth, int p_numTrees, int p_features)
	{
		m_maxDepth = p_maxDepth;
		m_numTrees = p_numTrees;
		m_features = p_features;
		
		m_forest = new RandomForest();
		m_forest.setNumExecutionSlots(8);
	}
	
	abstract void SetData(Instances p_data) throws Exception;
	
	abstract String Train() throws Exception; 
	
	abstract String CrossValidate() throws Exception;
	
	abstract double[] Run() throws Exception;
	
	
	
	
}
