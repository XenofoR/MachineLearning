import weka.core.Instances;


public class ActiveForest extends RegressionForest {

	public ActiveForest(int p_maxDepth, int p_numTrees, int p_features) {
		super(p_maxDepth, p_numTrees, p_features);
		// TODO Auto-generated constructor stub
	}
	
	void SetData(Instances p_data)
	{
		
	}
	void SetData(Instances p_labledData, Instances p_unlabledData)throws Exception
	{
		
	}

	double[] Run() throws Exception {
		double[] returnValue = new double[1]; //TODO: FIX SIZE
		return returnValue;
	}

	String Train() throws Exception {
		
		return "Not ready yet";
	}

	String CrossValidate() throws Exception
	{
		return "Not ready yet";
	}

}
