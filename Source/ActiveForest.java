
public class ActiveForest extends RegressionForest {

	public ActiveForest(int p_maxDepth, int p_numTrees, int p_features) {
		super(p_maxDepth, p_numTrees, p_features);
		// TODO Auto-generated constructor stub
	}


	double[] Run(String p_data) throws Exception {
		double[] returnValue = new double[1]; //TODO: FIX SIZE
		return returnValue;
	}

	double[] Train(String p_trainingData) throws Exception {
		
		
		double[] values = new double[2];
		
		values[0] = m_evaluator.meanAbsoluteError();
		values[1] = m_evaluator.errorRate();
		return values;
	}

	

}
