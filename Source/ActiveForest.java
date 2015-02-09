import weka.core.Instances;
import weka.core.Instance;


public class ActiveForest extends RegressionForest {

	Instances m_unlabledStructure;
	public ActiveForest(int p_maxDepth, int p_numTrees, int p_features) {
		super(p_maxDepth, p_numTrees, p_features);
		// TODO Auto-generated constructor stub
	}
	/** Should not be called on Active version */
	void SetData(Instances p_data)
	{
	}
	void SetData(Instances p_labledData, Instances p_unlabledData)throws Exception
	{
		m_structure = p_labledData;
		m_unlabledStructure = p_unlabledData;
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
	
	public Instance CalculateLabelRequest()
	{
		return null;
	}

}
