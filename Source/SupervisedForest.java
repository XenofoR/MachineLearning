import java.util.Iterator;
import java.util.Random;


import weka.classifiers.Evaluation;

public class SupervisedForest extends RegressionForest {

	public SupervisedForest(int p_maxDepth, int p_numTrees, int p_features) {
		super(p_maxDepth, p_numTrees, p_features);
	}

	void SetData(String p_data) throws Exception
	{
		ReadFile(p_data);
		
		m_evaluator = new Evaluation(m_structure);
		m_forest.setNumTrees(m_numTrees);
		m_forest.setMaxDepth(m_maxDepth);
	}
	
	String Train() throws Exception {
		
		m_forest.buildClassifier(m_structure);

		return m_forest.toString();
	}
	
	String CrossValidate() throws Exception
	{
		m_evaluator.crossValidateModel(m_forest, m_structure, 10, new Random(1));
		
		return m_evaluator.toSummaryString();
	}

	double[] Run() throws Exception {
		
		double[] returnValue = new double[m_structure.numInstances()];
		
		for(int i = 0; i < m_structure.numInstances(); i++)
		{
			returnValue[i] = m_forest.classifyInstance(m_structure.get(i));
		}
		
		return returnValue;
	}



}
