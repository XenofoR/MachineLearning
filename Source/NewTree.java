import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Vector;
import java.lang.Math;

import javax.rmi.CORBA.Util;
import javax.swing.DebugGraphics;
import javax.xml.crypto.KeySelector.Purpose;





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
 - InnerTree contains two new functions singleCovariance() and Covariance() to mirror how weka does regression
 - BuildTree in InnerTree now constructs a member covarianceMatrix if the tree represents a leaf.
 - numericDistrubution() added cluster gain calculation to the final gain calculation
 */
public class NewTree extends weka.classifiers.trees.RandomTree
{
	
	private Plotter m_plotter;
	
	InnerTree m_Tree;
	
	public String PrintCovarianceMatrices()
	{
		String output = "";
		
		output = m_Tree.PrintCovarianceMatrices();
		
		return output;
	}

	public NewTree()
	{
		super();
		m_plotter = new Plotter();
		m_plotter.Init("IAM A TREE");
		m_plotter.SetPlot(Debugger.g_plot);
	}
	
	public Vector<Double> GetPurity()
	{
		Vector<Double> returnVector = new Vector<Double>();
		
		m_Tree.GetPurity(returnVector);
		
		return returnVector;
	}
	
	public double CalculateSilhouetteIndex()
	{
		double index = 0.0;
		Vector<double[][]> covarianceMatrix = new Vector<double[][]>();
		
		m_Tree.FindCovarianceMatrices(covarianceMatrix);
		
		
		for(Iterator<double[][]> i = covarianceMatrix.iterator(); i.hasNext();)
		{
			double internalDisimilarity;
			double externalDisimilarity;
			
			
			for(int j = 0; j < i.next().length; j++)
			{
				
			}
		}
		
		return index;
	}
	
	public double[] distributionForInstance(Instance instance) throws Exception {

	    if (m_zeroR != null) {
	      return m_zeroR.distributionForInstance(instance);
	    } else {
	      return m_Tree.distributionForInstance(instance);
	    }
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

	      classProbs[0] += inst.classValue() * inst.weight();
	      totalSumSquared += inst.classValue() * inst.classValue()
	        * inst.weight();
	      totalWeight += inst.weight();
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
	    m_plotter.Display2dPlot();
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
		protected InnerTree[] m_Successors;
		double m_purity;
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
		 
		 public void FindCovarianceMatrices(Vector<double[][]> p_matricies)
		 {
			 if(m_covarianceMatrix != null)
				 p_matricies.add(m_covarianceMatrix);
			 else
			 {
				 m_Successors[0].FindCovarianceMatrices(p_matricies);
				 m_Successors[1].FindCovarianceMatrices(p_matricies);
			 }
		 }
		 
		 public void GetPurity(Vector<Double> p_returnVector)
		 {
			 if(m_Attribute == -1)
				 p_returnVector.add(m_purity);
			 else
			 {
				 m_Successors[0].GetPurity(p_returnVector);
				 m_Successors[1].GetPurity(p_returnVector);
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
		        }  else {

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
		
		public double[] distributionForInstance(Instance instance) throws Exception {

		      double[] returnedDist = null;

		      if (m_Attribute > -1) {

		        // Node is not a leaf
		        if (instance.isMissing(m_Attribute)) {

		          // Value is missing
		          returnedDist = new double[m_Info.numClasses()];

		          // Split instance up
		          for (int i = 0; i < m_Successors.length; i++) {
		            double[] help = m_Successors[i].distributionForInstance(instance);
		            if (help != null) {
		              for (int j = 0; j < help.length; j++) {
		                returnedDist[j] += m_Prop[i] * help[j];
		              }
		            }
		          }
		        } else if (m_Info.attribute(m_Attribute).isNominal()) {

		          // For nominal attributes
		          returnedDist = m_Successors[(int) instance.value(m_Attribute)]
		            .distributionForInstance(instance);
		        } else {

		          // For numeric attributes
		          if (instance.value(m_Attribute) < m_SplitPoint) {
		            returnedDist = m_Successors[0].distributionForInstance(instance);
		          } else {
		            returnedDist = m_Successors[1].distributionForInstance(instance);
		          }
		        }
		      }

		      // Node is a leaf or successor is empty?
		      if ((m_Attribute == -1) || (returnedDist == null)) {

		        // Is node empty?
		        if (m_ClassDistribution == null) {
		          if (getAllowUnclassifiedInstances()) {
		            double[] result = new double[m_Info.numClasses()];
		            if (m_Info.classAttribute().isNumeric()) {
		              result[0] = Utils.missingValue();
		            }
		            return result;
		          } else {
		            return null;
		          }
		        }

		        // Else return normalized distribution
		        double[] normalizedDistribution = m_ClassDistribution.clone();
		        if (m_Info.classAttribute().isNominal()) {
		          Utils.normalize(normalizedDistribution);
		        }
		        return normalizedDistribution;
		      } else {
		        return returnedDist;
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
		      double priorCovar = 0;
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
		          totalSumOfWeights);
		        priorCovar = SingleCovariance(instance);
		      }

		      // System.err.println("Total weight " + totalWeight);
		      // double sum = Utils.sum(classProbs);
		      if (p_totalWeight < 2 * m_MinNum ||

		        // Numeric case
		        (p_labeledData.classAttribute().isNumeric() && (priorVar + priorCovar) / p_totalWeight < minVariance)

		        
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
		        
		        //Construct covarianceMatrix for the cluster contained in this leaf
		        Instances instance = new Instances(p_labeledData);
		        instance.addAll(p_unlabeledData);
		        m_covarianceMatrix = new double[instance.numAttributes()-1][instance.numAttributes()-1];
		        Utilities.CalculateCovarianceMatrix(instance, m_covarianceMatrix);

		        
		        //calculate Purity
		        double labeledMean = 0.0;
		        for(int i = 0; i < p_labeledData.numInstances(); i++)
		        {
		        	labeledMean += p_labeledData.instance(i).classValue();
		        }
		        labeledMean /= p_labeledData.numInstances();
		        
		        double variance = 0.0;
		        for(int i = 0; i <  p_labeledData.numInstances(); i++)
		        {
		        	variance = Math.pow(p_labeledData.instance(i).classValue() - labeledMean, 2);
		        }
		        variance /= (p_labeledData.numInstances() - 1);
		        
		        p_unlabeledData.setClassIndex(p_unlabeledData.numAttributes() - 1);
		        for(int i = 0; i < p_unlabeledData.numInstances(); i++)
		        {
		        	if((m_ClassDistribution[0] - variance) < p_unlabeledData.instance(i).classValue() && p_unlabeledData.instance(i).classValue() < (m_ClassDistribution[0] + variance))
		        		m_purity ++;
		        }
		        m_purity /= p_unlabeledData.numInstances();
		        p_unlabeledData.setClassIndex(-1);
		        m_plotter.Set2dPlotValues(p_unlabeledData, p_labeledData);

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
		       
		        m_Distribution = new double[2];
		        m_Distribution[0] = priorVar;
		        m_Distribution[1] = p_totalWeight;
		      }
		      //We are a leaf, so we save the covariance matrix
		      if(m_Attribute == -1)
		      {
		    	  Instances instances = new Instances(p_labeledData);
			      instances.addAll(p_unlabeledData);
			      m_covarianceMatrix = new double[instances.numAttributes()-1][instances.numAttributes()-1];
			      Utilities.CalculateCovarianceMatrix( instances, m_covarianceMatrix);
			      m_plotter.Set2dPlotValues(p_unlabeledData, p_labeledData);
			      
			      //calculate purity
			      double labeledMean = 0.0;
			        for(int i = 0; i < p_labeledData.numInstances(); i++)
			        {
			        	labeledMean += p_labeledData.instance(i).classValue();
			        }
			        labeledMean /= p_labeledData.numInstances();
			        
			        double variance = 0.0;
			        for(int i = 0; i <  p_labeledData.numInstances(); i++)
			        {
			        	variance = Math.pow(p_labeledData.instance(i).classValue() - labeledMean, 2);
			        }
			        variance /= (p_labeledData.numInstances() - 1);
			        
			        p_unlabeledData.setClassIndex(p_unlabeledData.numAttributes() - 1);
			        for(int i = 0; i < p_unlabeledData.numInstances(); i++)
			        {
			        	if((m_ClassDistribution[0] - variance) < p_unlabeledData.instance(i).classValue() && p_unlabeledData.instance(i).classValue() < (m_ClassDistribution[0] + variance))
			        		m_purity ++;
			        }
			        m_purity /= p_unlabeledData.numInstances();
			        p_unlabeledData.setClassIndex(-1);
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

			        // For numeric attributes
			        sums = new double[2];
			        sumSquared = new double[2];
			        sumOfWeights = new double[2];
			        double[] currSums = new double[2];
			        double[] currSumSquared = new double[2];
			        double[] currSumOfWeights = new double[2];

			        // Sort data
			        p_labeledData.sort(att);
			        p_unlabeledData.sort(att);

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
			        double currSplit;
			        if(p_labeledData.numInstances() != 0)
			        	currSplit = p_labeledData.instance(0).value(att);
			        else
			        	currSplit = p_unlabeledData.instance(0).value(att);
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
			        	
			          //TODO: CHANGE THE FISK to a non static value and update variance calculation to seperate stuff
			          if (inst.value(att) > currSplit) {
			        	double k = variance(currSums, currSumSquared,
					              currSumOfWeights);
			        	double c = (Utilities.g_alphaValue * Covariance(clusterData.numInstances(), splitData(clusterData, inst.value(att), att)));
			            currVal = k + c;
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
			      //TODO: change the FISK to a non static value
			      Instances clusterInstances = new Instances(p_labeledData);
			      clusterInstances.addAll(p_unlabeledData);
			      clusterInstances.setClassIndex(-1);
			      double clusterPrior = SingleCovariance(clusterInstances);
			      double clusterVar = Covariance(clusterInstances.numInstances(), splitData(clusterInstances, splitPoint, att));
			      //System.out.println("covariance prior: " + clusterPrior + " covariance current: " + clusterVar);
			      double gain = (priorVar - var) + Utilities.g_alphaValue *(clusterPrior - clusterVar);

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
			//if(p_instances.numInstances() < 2)
				//return 0;

			double[][] covarianceMatrix = new double[p_instances.numAttributes() -1][p_instances.numAttributes() - 1];
			Utilities.CalculateCovarianceMatrix(p_instances, covarianceMatrix);
			
			double det = Utilities.CalculateDeterminant(covarianceMatrix);
			Debugger.DebugPrint("Determinant: "+ det, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			det = Math.abs(det);
			
			if(det <= 0)
				return 0.0;
			return (Math.log(det)/Math.log(2));
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
