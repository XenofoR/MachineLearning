import java.io.File;
import java.util.Iterator;
import java.util.Random;

public class SupervisedForest extends RegressionForest {

	public SupervisedForest(int p_maxDepth, int p_numTrees, int p_features) {
		super(p_maxDepth, p_numTrees, p_features);
	}

	void Train(String p_data) throws Exception {
		
		ReadFile(p_data);
		
		String[] Options = new String[1];
		
		m_evaluator.crossValidateModel(m_forest, m_structure, 10, new Random(1), Options);

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
