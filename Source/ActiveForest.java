import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;

//https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/trees/RandomForest.java
public class ActiveForest extends weka.classifiers.trees.RandomForest {
	Bilbo m_bagger;
	//Instances m_unlabledStructure;
	public ActiveForest() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public void buildClassifier(Instances p_labeledData, Instances p_unlabeledData) throws Exception
	{	
		// remove instances with missing class
		p_labeledData = new Instances(p_labeledData);
		p_labeledData.deleteWithMissingClass();

	    m_bagger = new Bilbo();

	    // RandomTree implements WeightedInstancesHandler, so we can
	    // represent copies using weights to achieve speed-up.
	    m_bagger.setRepresentCopiesUsingWeights(true);

	    NewTree rTree = new NewTree();

	    // set up the random tree options
	    m_KValue = m_numFeatures;
	    if (m_KValue < 1) {
	      m_KValue = (int) Utils.log2(p_labeledData.numAttributes() - 1) + 1;
	    }
	    rTree.setKValue(m_KValue);
	    rTree.setMaxDepth(getMaxDepth());
	    rTree.setDoNotCheckCapabilities(true);

	    // set up the bagger and build the forest
	    m_bagger.setClassifier(rTree);
	    m_bagger.setSeed(m_randomSeed);
	    m_bagger.setNumIterations(m_numTrees);
	    m_bagger.setCalcOutOfBag(!getDontCalculateOutOfBagError());
	    m_bagger.setNumExecutionSlots(m_numExecutionSlots);
	    m_bagger.buildClassifier(p_labeledData, p_unlabeledData);
	}
	
	public String GetInfo() {
		// TODO Auto-generated method stub
		return null;
	}
	/** Should not be called on Active version */
	/*
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
 */
	
	
}
