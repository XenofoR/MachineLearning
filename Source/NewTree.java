import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Vector;
import java.lang.Math;

import javax.swing.DebugGraphics;





import weka.attributeSelection.PrincipalComponents;
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
/*
 Changes made:
 - buildClassifier() takes two instances instead of one.
 - added our own implementation of splitData() which can split an instance on a specific attribute index and splitPoint.
 - numericDistribution() also takes a second instance set now and we added our covariance calculation to the best split calculation
 - numericDistrubution() added cluster gain calculation to the final gain calculation
 */
public class NewTree extends weka.classifiers.trees.RandomTree
{
	
	
	
	InnerTree m_Tree;
	
	public String PrintCovarianceMatrices()
	{
		String output = "";
		
		output = m_Tree.PrintCovarianceMatrices();
		
		return output;
	}

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
	
	
	public String toString() {

	    // only ZeroR model?
	    if (m_zeroR != null) {
	      StringBuffer buf = new StringBuffer();
	      buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
	      buf.append(this.getClass().getName().replaceAll(".*\\.", "")
	        .replaceAll(".", "=")
	        + "\n\n");
	      buf
	        .append("Warning: No model could be built, hence ZeroR model is used:\n\n");
	      buf.append(m_zeroR.toString());
	      return buf.toString();
	    }

	    if (m_Tree == null) {
	      return "RandomTree: no model has been built yet.";
	    } else {
	      return "\nRandomTree\n==========\n"
	        + m_Tree.toString(0)
	        + "\n"
	        + "\nSize of the tree : "
	        + m_Tree.numNodes()
	        + (getMaxDepth() > 0 ? ("\nMax depth of tree: " + getMaxDepth()) : (""));
	    }
	  }
	
	protected class InnerTree extends Tree
	{
		double[][] m_covarianceMatrix = null;
		 public int numNodes() {

		      if (m_Attribute == -1) {
		        return 1;
		      } else {
		        int size = 1;
		        for (Tree m_Successor : m_Successors) {
		          size += m_Successor.numNodes();
		        }
		        return size;
		      }
		    }
		 
		 public String PrintCovarianceMatrices()
			{
				String output = "\n";
				
				if(m_covarianceMatrix != null)
				{
					output += Arrays.deepToString(m_covarianceMatrix);
				}
				else
				{
					output += "SPLIT";
					for(int i = 0; i < 2; i++)
						output += m_Successors[i].PrintCovarianceMatrices();
				}
				return output;
			}
		 
		protected String toString(int level) {

		      try {
		        StringBuffer text = new StringBuffer();

		        if (m_Attribute == -1) {

		          // Output leaf info
		          return leafString();
		        } else if (m_Info.attribute(m_Attribute).isNominal()) {

		          // For nominal attributes
		          for (int i = 0; i < m_Successors.length; i++) {
		            text.append("\n");
		            for (int j = 0; j < level; j++) {
		              text.append("|   ");
		            }
		            text.append(m_Info.attribute(m_Attribute).name() + " = "
		              + m_Info.attribute(m_Attribute).value(i));
		            text.append(m_Successors[i].toString(level + 1));
		          }
		        } else {

		          // For numeric attributes
		          text.append("\n");
		          for (int j = 0; j < level; j++) {
		            text.append("|   ");
		          }
		          text.append(m_Info.attribute(m_Attribute).name() + " < "
		            + Utils.doubleToString(m_SplitPoint, 2));
		          text.append(m_Successors[0].toString(level + 1));
		          text.append("\n");
		          for (int j = 0; j < level; j++) {
		            text.append("|   ");
		          }
		          text.append(m_Info.attribute(m_Attribute).name() + " >= "
		            + Utils.doubleToString(m_SplitPoint, 2));
		          text.append(m_Successors[1].toString(level + 1));
		        }

		        return text.toString();
		      } catch (Exception e) {
		        e.printStackTrace();
		        return "RandomTree: tree can't be printed";
		      }
		    }
		protected InnerTree[] m_Successors;
		public String toString() {

		    // only ZeroR model?
		    if (m_zeroR != null) {
		      StringBuffer buf = new StringBuffer();
		      buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
		      buf.append(this.getClass().getName().replaceAll(".*\\.", "")
		        .replaceAll(".", "=")
		        + "\n\n");
		      buf
		        .append("Warning: No model could be built, hence ZeroR model is used:\n\n");
		      buf.append(m_zeroR.toString());
		      return buf.toString();
		    }

		    if (m_Tree == null) {
		      return "RandomTree: no model has been built yet.";
		    } else {
		      return "\nRandomTree\n==========\n"
		        + m_Tree.toString(0)
		        + "\n"
		        + "\nSize of the tree : "
		        + m_Tree.numNodes()
		        + (getMaxDepth() > 0 ? ("\nMax depth of tree: " + getMaxDepth()) : (""));
		    }
		  }
		protected void buildTree(Instances p_labeledData, Instances p_unlabeledData, double[] p_classProbs,
		      int[] p_attIndicesWindow, double p_totalWeight, Random p_random, int p_depth,
		      double minVariance) throws Exception {
			
		      // Make leaf if there are no training instances
		      if (p_labeledData.numInstances() == 0 && p_unlabeledData.numInstances() == 0) {
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
		        Instances instance = new Instances(p_labeledData);
		        instance.addAll(p_unlabeledData);
		        priorVar = NewTree.singleVariance(totalSum, totalSumSquared,
		          totalSumOfWeights) + SingleCovariance(instance);
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
		        Instances instance = new Instances(p_labeledData);
		        instance.addAll(p_unlabeledData);
		        m_covarianceMatrix = new double[instance.numAttributes()-1][instance.numAttributes()-1];
		        Utilities.CalculateCovarianceMatrix(instance, m_covarianceMatrix);
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
		      while ((windowSize > 0) && (k-- > 0 || !gainFound)) 
		      {

		        int chosenIndex = p_random.nextInt(windowSize);
		        attIndex = p_attIndicesWindow[chosenIndex];

		        // shift chosen attIndex out of window
		        p_attIndicesWindow[chosenIndex] = p_attIndicesWindow[windowSize - 1];
		        p_attIndicesWindow[windowSize - 1] = attIndex;
		        windowSize--;

		        double currSplit = p_labeledData.classAttribute().isNominal() ? distribution(
		          props, dists, attIndex, p_labeledData) : numericDistribution(props, dists,
		          attIndex, totalSubsetWeights, p_labeledData, p_unlabeledData, tempNumericVals);
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
		        Instances[] unlabeledSubset = splitData(p_unlabeledData);
		        m_Successors = new InnerTree[bestDists.length];
		        double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];

		        for (int i = 0; i < bestDists.length; i++) {
		          m_Successors[i] = new InnerTree();
		          m_Successors[i].buildTree(subsets[i], unlabeledSubset[i], bestDists[i], p_attIndicesWindow,
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
		      //We are a leaf, so we save the covariance matrix
		      if(m_Attribute == -1)
		      {
		    	  Instances instances = new Instances(p_labeledData);
			      instances.addAll(p_unlabeledData);
			      m_covarianceMatrix = new double[instances.numAttributes()-1][instances.numAttributes()-1];
			      Utilities.CalculateCovarianceMatrix( instances, m_covarianceMatrix);
		      }
		    }
		
		protected double numericDistribution(double[][] props, double[][][] dists,
			      int att, double[][] subsetWeights, Instances p_labeledData, Instances p_unlabeledData, double[] vals)
			      throws Exception {
				
			      double splitPoint = Double.NaN;
			      Attribute attribute = p_labeledData.attribute(att);
			      double[][] dist = null;
			      double[] sums = null;
			      double[] sumSquared = null;
			      double[] sumOfWeights = null;
			      double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
			      int indexOfFirstMissingValue = p_labeledData.numInstances();

			      if (attribute.isNominal()) {
			        sums = new double[attribute.numValues()];
			        sumSquared = new double[attribute.numValues()];
			        sumOfWeights = new double[attribute.numValues()];
			        int attVal;

			        for (int i = 0; i < p_labeledData.numInstances(); i++) {
			          Instance inst = p_labeledData.instance(i);
			          if (inst.isMissing(att)) {

			            // Skip missing values at this stage
			            if (indexOfFirstMissingValue == p_labeledData.numInstances()) {
			              indexOfFirstMissingValue = i;
			            }
			            continue;
			          }

			          attVal = (int) inst.value(att);
			          sums[attVal] += inst.classValue() * inst.weight();
			          sumSquared[attVal] += inst.classValue() * inst.classValue()
			            * inst.weight();
			          sumOfWeights[attVal] += inst.weight();
			        }

			        totalSum = Utils.sum(sums);
			        totalSumSquared = Utils.sum(sumSquared);
			        totalSumOfWeights = Utils.sum(sumOfWeights);
			      } else {
			        // For numeric attributes
			        sums = new double[2];
			        sumSquared = new double[2];
			        sumOfWeights = new double[2];
			        double[] currSums = new double[2];
			        double[] currSumSquared = new double[2];
			        double[] currSumOfWeights = new double[2];

			        // Sort data
			        p_labeledData.sort(att);

			        // Move all instances into second subset
			        for (int j = 0; j < p_labeledData.numInstances(); j++) {
			          Instance inst = p_labeledData.instance(j);
			          if (inst.isMissing(att)) {

			            // Can stop as soon as we hit a missing value
			            indexOfFirstMissingValue = j;
			            break;
			          }

			          currSums[1] += inst.classValue() * inst.weight();
			          currSumSquared[1] += inst.classValue() * inst.classValue()
			            * inst.weight();
			          currSumOfWeights[1] += inst.weight();
			        }

			        totalSum = currSums[1];
			        totalSumSquared = currSumSquared[1];
			        totalSumOfWeights = currSumOfWeights[1];

			        sums[1] = currSums[1];
			        sumSquared[1] = currSumSquared[1];
			        sumOfWeights[1] = currSumOfWeights[1];

			        // Try all possible split points
			        double currSplit = p_labeledData.instance(0).value(att);
			        double currVal, bestVal = Double.MAX_VALUE;

			        //For clustering we need to consider both labeled and unlabeled data, so we move them to one set
			        Instances clusterData = new Instances(p_labeledData);
			        clusterData.setClassIndex(-1);
			        clusterData.addAll(p_unlabeledData);
			        int endfor = indexOfFirstMissingValue + p_unlabeledData.numInstances();
			        for (int i = 0; i < endfor; i++) {

			        	Instance inst;
			        	if(i < indexOfFirstMissingValue)
			        		inst = p_labeledData.instance(i);
			        	else
			        		inst = p_unlabeledData.instance(i - indexOfFirstMissingValue);
			        	
			          //TODO: THIS IS PLACE TO ENTER CLUSTER ALGORITHM
			          if (inst.value(att) > currSplit) {
			        	double k = variance(currSums, currSumSquared,
					              currSumOfWeights);
			        	double c = (1.5 * Covariance(clusterData.numInstances(), splitData(clusterData, currSplit, att)));
			            currVal = k+ c;
			            k -= c;
			            Debugger.DebugPrint("Diff between variance and covariane? = " + k, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			            if (currVal < bestVal) {
			              bestVal = currVal;
			              splitPoint = (inst.value(att) + currSplit) / 2.0;

			              // Check for numeric precision problems
			              if (splitPoint <= currSplit) {
			                splitPoint = inst.value(att);
			              }

			              for (int j = 0; j < 2; j++) {
			                sums[j] = currSums[j];
			                sumSquared[j] = currSumSquared[j];
			                sumOfWeights[j] = currSumOfWeights[j];
			              }
			            }
			          }

			          currSplit = inst.value(att);
			          
			          if(inst.classIndex() != -1)
			          {
				          double classVal = inst.classValue() * inst.weight();
				          double classValSquared = inst.classValue() * classVal;
	
				          currSums[0] += classVal;
				          currSumSquared[0] += classValSquared;
				          currSumOfWeights[0] += inst.weight();
	
				          currSums[1] -= classVal;
				          currSumSquared[1] -= classValSquared;
				          currSumOfWeights[1] -= inst.weight();
			          }
			        }
			      }

			      // Compute weights
			      props[0] = new double[sums.length];
			      for (int k = 0; k < props[0].length; k++) {
			        props[0][k] = sumOfWeights[k];
			      }
			      if (!(Utils.sum(props[0]) > 0)) {
			        for (int k = 0; k < props[0].length; k++) {
			          props[0][k] = 1.0 / props[0].length;
			        }
			      } else {
			        Utils.normalize(props[0]);
			      }

			      // Distribute weights for instances with missing values
			      for (int i = indexOfFirstMissingValue; i < p_labeledData.numInstances(); i++) {
			        Instance inst = p_labeledData.instance(i);

			        for (int j = 0; j < sums.length; j++) {
			          sums[j] += props[0][j] * inst.classValue() * inst.weight();
			          sumSquared[j] += props[0][j] * inst.classValue() * inst.classValue()
			            * inst.weight();
			          //J == 0 || 1
			          sumOfWeights[j] += props[0][j] * inst.weight();
			        }
			        totalSum += inst.classValue() * inst.weight();
			        totalSumSquared += inst.classValue() * inst.classValue()
			          * inst.weight();
			        totalSumOfWeights += inst.weight();
			      }

			      // Compute final distribution
			      dist = new double[sums.length][p_labeledData.numClasses()];
			      for (int j = 0; j < sums.length; j++) {
			        if (sumOfWeights[j] > 0) {
			          dist[j][0] = sums[j] / sumOfWeights[j];
			        } else {
			          dist[j][0] = totalSum / totalSumOfWeights;
			        }
			      }
			      // Compute variance gain
			      double priorVar = singleVariance(totalSum, totalSumSquared,
			        totalSumOfWeights);
			      double var = variance(sums, sumSquared, sumOfWeights);
			      
			      //Add cluster gain over the parent to the final gain calculations.
			      Instances clusterInstances = new Instances(p_labeledData);
			      clusterInstances.addAll(p_unlabeledData);
			      clusterInstances.setClassIndex(-1);
			      double clusterPrior = SingleCovariance(clusterInstances);
			      double clusterVar = 1.5 * Covariance(clusterInstances.numInstances(), splitData(clusterInstances, splitPoint, att));
			      double gain = (priorVar - var) + (clusterPrior - clusterVar);

			      // Return distribution and split point
			      subsetWeights[att] = sumOfWeights;
			      dists[0] = dist;
			      vals[att] = gain;

			      return splitPoint;
			    }
		
		private double Covariance(int p_sumParentInstances, Instances[] p_instances) throws Exception
		{
			double hejhoppiklingonskogen = 0.0, parentByChild, singleResult;
			Debugger.DebugPrint("Entering Covariance", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			Debugger.DebugPrint("SumParents = " + p_sumParentInstances + "\n" + "Sum child1 = " + p_instances[0].numInstances() + "\n" + "Sum child2 = " + p_instances[1].numInstances(),
								Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			for(int i = 0; i < 2; i++)
			{
				parentByChild = (double)p_instances[i].numInstances() / (double)p_sumParentInstances;
				singleResult =  SingleCovariance(p_instances[i]);			
				hejhoppiklingonskogen += (parentByChild * singleResult) ;
				Debugger.DebugPrint("Covariance value= " + hejhoppiklingonskogen, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			}
			Debugger.DebugPrint("Leaving Covariance", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			return hejhoppiklingonskogen;
		}
		
		private double SingleCovariance(Instances p_instances) throws Exception
		{
			if(p_instances.numInstances() < 2)
				return 0;

			double[][] covarianceMatrix = new double[p_instances.numAttributes() -1][p_instances.numAttributes() - 1];
			Utilities.CalculateCovarianceMatrix(p_instances, covarianceMatrix);
			
			double det = Utilities.CalculateDeterminant(covarianceMatrix);
			Debugger.DebugPrint("Determinant: "+ det, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			det = Math.abs(det);
			
			if(det <= 0)
				return 0.0;
			return Math.log(det);
		}
		
		protected Instances[] splitData(Instances p_data, double p_splitPoint, int p_attr) throws Exception {

		      // Allocate array of Instances objects
		      Instances[] subsets = new Instances[2];
		      for (int i = 0; i < 2; i++) {
		        subsets[i] = new Instances(p_data, p_data.numInstances());
		      }

		      // Go through the data
		      for (int i = 0; i < p_data.numInstances(); i++) {

		        // Get instance
		        Instance inst = p_data.instance(i);

		        // We will disregard missing attributes entirely, thus these instances will not effect the density
		        if (inst.isMissing(p_attr)) {

		          continue;
		        }

		        // Do we have a nominal attribute?
		        if (p_data.attribute(p_attr).isNominal()) {
		          subsets[(int) inst.value(p_attr)].add(inst);

		          // Proceed to next instance
		          continue;
		        }

		        // Do we have a numeric attribute?
		        if (p_data.attribute(p_attr).isNumeric()) {
		          subsets[(inst.value(p_attr) < p_splitPoint) ? 0 : 1].add(inst);

		          // Proceed to next instance
		          continue;
		        }

		        // Else throw an exception
		        throw new IllegalArgumentException("Unknown attribute type");
		      }

		      // Save memory
		      for (int i = 0; i < 2; i++) {
		        subsets[i].compactify();
		      }

		      // Return the subsets
		      return subsets;
		    }
	}
	
	
	
	 
}
	//Override standard stuff here
