import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

//https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/trees/RandomTree.java
public class NewTree extends weka.classifiers.trees.RandomTree
{
	InnerTree m_Tree;
	public void buildClassifier(Instances p_labeledData, Instances p_unlabeledData) throws Exception {

	    // Make sure K value is in range
	    if (m_KValue > p_labeledData.numAttributes() - 1) {
	      m_KValue = p_labeledData.numAttributes() - 1;
	    }
	    if (m_KValue < 1) {
	      m_KValue = (int) Utils.log2(p_labeledData.numAttributes() - 1) + 1;
	    }

	    // can classifier handle the data?
	    getCapabilities().testWithFail(p_labeledData);

	    // remove instances with missing class
	    p_labeledData = new Instances(p_labeledData);
	    p_labeledData.deleteWithMissingClass();

	    // only class? -> build ZeroR model
	    if (p_labeledData.numAttributes() == 1) {
	      System.err
	        .println("Cannot build model (only class attribute present in data!), "
	          + "using ZeroR model instead!");
	      m_zeroR = new weka.classifiers.rules.ZeroR();
	      m_zeroR.buildClassifier(p_labeledData);
	      return;
	    } else {
	      m_zeroR = null;
	    }

	    // Figure out appropriate datasets
	    Instances labeledTrain = null;
	    Instances labeledBackfit = null;
	    Instances unlabeledTrain = null;
	    Instances unlabeledBackfit = null;
	    Random rand = p_labeledData.getRandomNumberGenerator(m_randomSeed);
	    if (m_NumFolds <= 0) {
	      labeledTrain = p_labeledData;
	    } else {
	      p_labeledData.randomize(rand);
	      p_labeledData.stratify(m_NumFolds);
	      labeledTrain = p_labeledData.trainCV(m_NumFolds, 1, rand);
	      labeledBackfit = p_labeledData.testCV(m_NumFolds, 1);
	    }
	    
	    rand = p_labeledData.getRandomNumberGenerator(m_randomSeed);
	    if (m_NumFolds <= 0) {
	    	unlabeledTrain = p_unlabeledData;
	    } else {
	      p_unlabeledData.randomize(rand);
	      p_unlabeledData.stratify(m_NumFolds);
	      unlabeledTrain = p_unlabeledData.trainCV(m_NumFolds, 1, rand);
	      unlabeledBackfit = p_unlabeledData.testCV(m_NumFolds, 1);
	    }

	    // Create the attribute indices window
	    int[] attIndicesWindow = new int[p_labeledData.numAttributes() - 1];
	    int j = 0;
	    for (int i = 0; i < attIndicesWindow.length; i++) {
	      if (j == p_labeledData.classIndex()) {
	        j++; // do not include the class
	      }
	      attIndicesWindow[i] = j++;
	    }

	    double totalWeight = 0;
	    double totalSumSquared = 0;

	    // Compute initial class counts
	    double[] classProbs = new double[labeledTrain.numClasses()];
	    for (int i = 0; i < labeledTrain.numInstances(); i++) {
	      Instance inst = labeledTrain.instance(i);
	      if (p_labeledData.classAttribute().isNominal()) {
	        classProbs[(int) inst.classValue()] += inst.weight();
	        totalWeight += inst.weight();
	      } else {
	        classProbs[0] += inst.classValue() * inst.weight();
	        totalSumSquared += inst.classValue() * inst.classValue()
	          * inst.weight();
	        totalWeight += inst.weight();
	      }
	    }

	    double trainVariance = 0;
	    if (p_labeledData.classAttribute().isNumeric()) {
	      trainVariance = NewTree.singleVariance(classProbs[0], totalSumSquared,
	        totalWeight) / totalWeight;
	      classProbs[0] /= totalWeight;
	    }

	    // Build tree
	    m_Tree = new InnerTree();
	    m_Info = new Instances(p_labeledData, 0);
	    m_Tree.buildTree(labeledTrain, unlabeledTrain,  classProbs, attIndicesWindow, totalWeight, rand, 0,
	      m_MinVarianceProp * trainVariance);

	    // Backfit if required
	    if (labeledBackfit != null) {
	      m_Tree.backfitData(labeledBackfit); //TODO change to handle two instances
	    }
	  }
	
	protected class InnerTree extends Tree
	{
		protected InnerTree[] m_Successors;
		protected void buildTree(Instances p_labeledData, Instances p_unlabeledData, double[] p_classProbs,
		      int[] p_attIndicesWindow, double p_totalWeight, Random p_random, int p_depth,
		      double minVariance) throws Exception {
			
		      // Make leaf if there are no training instances
		      if (p_labeledData.numInstances() == 0) {
		        m_Attribute = -1;
		        m_ClassDistribution = null;
		        m_Prop = null;

		        if (p_labeledData.classAttribute().isNumeric()) {
		          m_Distribution = new double[2];
		        }
		        return;
		      }

		      double priorVar = 0;
		      if (p_labeledData.classAttribute().isNumeric()) {

		        // Compute prior variance
		        double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
		        for (int i = 0; i < p_labeledData.numInstances(); i++) {
		          Instance inst = p_labeledData.instance(i);
		          totalSum += inst.classValue() * inst.weight();
		          totalSumSquared += inst.classValue() * inst.classValue()
		            * inst.weight();
		          totalSumOfWeights += inst.weight();
		        }
		        priorVar = NewTree.singleVariance(totalSum, totalSumSquared,
		          totalSumOfWeights);
		      }

		      // Check if node doesn't contain enough instances or is pure
		      // or maximum depth reached
		      if (p_labeledData.classAttribute().isNominal()) {
		        p_totalWeight = Utils.sum(p_classProbs);
		      }
		      // System.err.println("Total weight " + totalWeight);
		      // double sum = Utils.sum(classProbs);
		      if (p_totalWeight < 2 * m_MinNum ||

		      // Nominal case
		        (p_labeledData.classAttribute().isNominal() && Utils.eq(
		          p_classProbs[Utils.maxIndex(p_classProbs)], Utils.sum(p_classProbs)))

		        ||

		        // Numeric case
		        (p_labeledData.classAttribute().isNumeric() && priorVar / p_totalWeight < minVariance)

		        ||

		        // check tree depth
		        ((getMaxDepth() > 0) && (p_depth >= getMaxDepth()))) {

		        // Make leaf
		        m_Attribute = -1;
		        m_ClassDistribution = p_classProbs.clone();
		        if (p_labeledData.classAttribute().isNumeric()) {
		          m_Distribution = new double[2];
		          m_Distribution[0] = priorVar;
		          m_Distribution[1] = p_totalWeight;
		        }

		        m_Prop = null;
		        return;
		      }

		      // Compute class distributions and value of splitting
		      // criterion for each attribute
		      double val = -Double.MAX_VALUE;
		      double split = -Double.MAX_VALUE;
		      double[][] bestDists = null;
		      double[] bestProps = null;
		      int bestIndex = 0;

		      // Handles to get arrays out of distribution method
		      double[][] props = new double[1][0];
		      double[][][] dists = new double[1][0][0];
		      double[][] totalSubsetWeights = new double[p_labeledData.numAttributes()][0];

		      // Investigate K random attributes
		      int attIndex = 0;
		      int windowSize = p_attIndicesWindow.length;
		      int k = m_KValue;
		      boolean gainFound = false;
		      double[] tempNumericVals = new double[p_labeledData.numAttributes()];
		      while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

		        int chosenIndex = p_random.nextInt(windowSize);
		        attIndex = p_attIndicesWindow[chosenIndex];

		        // shift chosen attIndex out of window
		        p_attIndicesWindow[chosenIndex] = p_attIndicesWindow[windowSize - 1];
		        p_attIndicesWindow[windowSize - 1] = attIndex;
		        windowSize--;

		        double currSplit = p_labeledData.classAttribute().isNominal() ? distribution(
		          props, dists, attIndex, p_labeledData) : numericDistribution(props, dists,
		          attIndex, totalSubsetWeights, p_labeledData, tempNumericVals);
		         //Calculate information gain
		        double currVal = p_labeledData.classAttribute().isNominal() ? gain(dists[0],
		          priorVal(dists[0])) : tempNumericVals[attIndex];

		        if (Utils.gr(currVal, 0)) {
		          gainFound = true;
		        }

		        if ((currVal > val) || ((currVal == val) && (attIndex < bestIndex))) {
		          val = currVal;
		          bestIndex = attIndex;
		          split = currSplit;
		          bestProps = props[0];
		          bestDists = dists[0];
		        }
		      }

		      // Find best attribute
		      m_Attribute = bestIndex;

		      // Any useful split found?
		      if (Utils.gr(val, 0)) {

		        // Build subtrees
		        m_SplitPoint = split;
		        m_Prop = bestProps;
		        Instances[] subsets = splitData(p_labeledData);
		        m_Successors = new InnerTree[bestDists.length];
		        double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];

		        for (int i = 0; i < bestDists.length; i++) {
		          m_Successors[i] = new InnerTree();
		          m_Successors[i].buildTree(subsets[i], bestDists[i], p_attIndicesWindow,
		            p_labeledData.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i],
		            p_random, p_depth + 1, minVariance);
		        }

		        // If all successors are non-empty, we don't need to store the class
		        // distribution
		        boolean emptySuccessor = false;
		        for (int i = 0; i < subsets.length; i++) {
		          if (m_Successors[i].m_ClassDistribution == null) {
		            emptySuccessor = true;
		            break;
		          }
		        }
		        if (emptySuccessor) {
		          m_ClassDistribution = p_classProbs.clone();
		        }
		      } else {

		        // Make leaf
		        m_Attribute = -1;
		        m_ClassDistribution = p_classProbs.clone();
		        if (p_labeledData.classAttribute().isNumeric()) {
		          m_Distribution = new double[2];
		          m_Distribution[0] = priorVar;
		          m_Distribution[1] = p_totalWeight;
		        }
		      }
		    }
	}
}
	//Override standard stuff here
