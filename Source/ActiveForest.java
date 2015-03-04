import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import java.util.Vector;

//https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/trees/RandomForest.java
public class ActiveForest extends weka.classifiers.trees.RandomForest {
	Bilbo m_bagger;
	double m_silhouetteIndex;
	//Instances m_unlabledStructure;
	public ActiveForest() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Vector<Vector<Double>> GetPurity()
	{
		return m_bagger.GetPurity();
	}
	
	public double CalculateSilhouetteIndex()
	{
		m_silhouetteIndex = m_bagger.CalculateSilhouetteIndex();
		return m_silhouetteIndex;
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
