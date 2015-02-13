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
	protected class InnerTree extends Tree
	{
		protected InnerTree[] m_Successors;
		protected void buildTree(Instances data, double[] classProbs,
		      int[] attIndicesWindow, double totalWeight, Random random, int depth,
		      double minVariance) throws Exception {
			
		      // Make leaf if there are no training instances
		      if (data.numInstances() == 0) {
		        m_Attribute = -1;
		        m_ClassDistribution = null;
		        m_Prop = null;

		        if (data.classAttribute().isNumeric()) {
		          m_Distribution = new double[2];
		        }
		        return;
		      }

		      double priorVar = 0;
		      if (data.classAttribute().isNumeric()) {

		        // Compute prior variance
		        double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
		        for (int i = 0; i < data.numInstances(); i++) {
		          Instance inst = data.instance(i);
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
		      if (data.classAttribute().isNominal()) {
		        totalWeight = Utils.sum(classProbs);
		      }
		      // System.err.println("Total weight " + totalWeight);
		      // double sum = Utils.sum(classProbs);
		      if (totalWeight < 2 * m_MinNum ||

		      // Nominal case
		        (data.classAttribute().isNominal() && Utils.eq(
		          classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs)))

		        ||

		        // Numeric case
		        (data.classAttribute().isNumeric() && priorVar / totalWeight < minVariance)

		        ||

		        // check tree depth
		        ((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {

		        // Make leaf
		        m_Attribute = -1;
		        m_ClassDistribution = classProbs.clone();
		        if (data.classAttribute().isNumeric()) {
		          m_Distribution = new double[2];
		          m_Distribution[0] = priorVar;
		          m_Distribution[1] = totalWeight;
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
		      double[][] totalSubsetWeights = new double[data.numAttributes()][0];

		      // Investigate K random attributes
		      int attIndex = 0;
		      int windowSize = attIndicesWindow.length;
		      int k = m_KValue;
		      boolean gainFound = false;
		      double[] tempNumericVals = new double[data.numAttributes()];
		      while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

		        int chosenIndex = random.nextInt(windowSize);
		        attIndex = attIndicesWindow[chosenIndex];

		        // shift chosen attIndex out of window
		        attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
		        attIndicesWindow[windowSize - 1] = attIndex;
		        windowSize--;

		        double currSplit = data.classAttribute().isNominal() ? distribution(
		          props, dists, attIndex, data) : numericDistribution(props, dists,
		          attIndex, totalSubsetWeights, data, tempNumericVals);

		        double currVal = data.classAttribute().isNominal() ? gain(dists[0],
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
		        Instances[] subsets = splitData(data);
		        m_Successors = new InnerTree[bestDists.length];
		        double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];

		        for (int i = 0; i < bestDists.length; i++) {
		          m_Successors[i] = new InnerTree();
		          m_Successors[i].buildTree(subsets[i], bestDists[i], attIndicesWindow,
		            data.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i],
		            random, depth + 1, minVariance);
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
		          m_ClassDistribution = classProbs.clone();
		        }
		      } else {

		        // Make leaf
		        m_Attribute = -1;
		        m_ClassDistribution = classProbs.clone();
		        if (data.classAttribute().isNumeric()) {
		          m_Distribution = new double[2];
		          m_Distribution[0] = priorVar;
		          m_Distribution[1] = totalWeight;
		        }
		      }
		    }
	}
}
	//Override standard stuff here
