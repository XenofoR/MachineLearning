import java.util.Iterator;
import java.util.Random;

import weka.classifiers.Evaluation;

public class SupervisedForest extends RegressionForest {

	public SupervisedForest(int p_maxDepth, int p_numTrees, int p_features) {
		super(p_maxDepth, p_numTrees, p_features);
	}

	double[] Train(String p_data) throws Exception {
		
		ReadFile(p_data);
		
		m_evaluator = new Evaluation(m_structure);
		
		m_evaluator.crossValidateModel(m_forest, m_structure, 10, new Random(1));
		
		double[] values = new double[2];
		
		values[0] = m_evaluator.meanAbsoluteError();
		values[1] = m_evaluator.errorRate();
		return values;
	}

	double[] Run(String p_data) throws Exception {
		
		ReadFile(p_data);
		
		double[] returnValue = new double[m_structure.numInstances()];
		
		for(int i = 0; i < m_structure.numInstances(); i++)
		{
			returnValue[i] = m_forest.classifyInstance(m_structure.get(i));
		}
		
		return returnValue;
	}



}
