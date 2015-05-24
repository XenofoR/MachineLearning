import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import java.util.Vector;

//https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/trees/RandomForest.java
public class ActiveForest extends weka.classifiers.trees.RandomForest {
	Bilbo m_bagger;

	//Instances m_unlabledStructure;
	public ActiveForest() {
		super();
		m_bagger = new Bilbo();
	}
	
	public Vector<Vector<double[]>> GetPurityAndVardiff()
	{
		return m_bagger.GetPurityAndVardiff();
	}
	
	public double CalculateRandIndex()
	{
		return m_bagger.CalculateRandIndex();
	}
	
	public Instances GetOracleData()
	{
		return m_bagger.getOracleData();
	}
	public Instances[] GetTransductedData()
	{
		return m_bagger.getTransductedData();
	}
	public void SetTransToInd(Instances[] p_data)
	{
		m_bagger.setTransToInd(p_data);
	}
	public double GetAverageTransductionError()
	{
		return m_bagger.GetAverageTransductionError();
	}
	public Long[] GetAndAverageGraphTime()
	{
		return m_bagger.GetAndAverageGraphTime();
	}
	public double[] CalculateCorrelationPercentage()
	{
		return m_bagger.CalculateCorrelationPercentage();
	}
	
	 /**
	   * Gets the out of bag error that was calculated as the classifier was built.
	   * 
	   * @return the out of bag error
	   */
	  public double measureOutOfBagError() {

	    if (m_bagger != null && !m_dontCalculateOutOfBagError) {
	      return m_bagger.measureOutOfBagError();
	    } else {
	      return Double.NaN;
	    }
	  }
	  
	public void buildClassifier(Instances p_labeledData, Instances p_unlabeledData) throws Exception
	{	
		OurUtil.g_numTrees = 0;
		// remove instances with missing class
		p_labeledData = new Instances(p_labeledData);
		p_labeledData.deleteWithMissingClass();

	    

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
	
	public String toString() {

	    if (m_bagger == null) {
	      return "Random forest not built yet";
	    } else {
	      StringBuffer temp = new StringBuffer();
	      temp.append("Random forest of "
	        + m_numTrees
	        + " trees, each constructed while considering "
	        + m_KValue
	        + " random feature"
	        + (m_KValue == 1 ? "" : "s")
	        + ".\n"
	        + (!getDontCalculateOutOfBagError() ? "Out of bag error: "
	          + Utils.doubleToString(m_bagger.measureOutOfBagError(), 4) : "")
	        + "\n"
	        + (getMaxDepth() > 0 ? ("Max. depth of trees: " + getMaxDepth() + "\n")
	          : ("")) + "\n");
	      if (m_printTrees) {
	        temp.append(m_bagger.toString());
	      }
	      return temp.toString();
	    }
	  }
	
	public double[] distributionForInstance(Instance instance) throws Exception {

	    return m_bagger.distributionForInstance(instance);
	  }

}
