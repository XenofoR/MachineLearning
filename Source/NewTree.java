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
import java.math.BigInteger;
import java.lang.reflect.Array;

import javax.rmi.CORBA.Util;
import javax.swing.DebugGraphics;
import javax.xml.crypto.KeySelector.Purpose;







import weka.core.InstanceComparator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.neighboursearch.PerformanceStats;
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
	private Graph m_graph;
	private double[][] m_leafDistanceMatrix;
	private InstanceComparator m_instanceComp;
	int m_counter = 0;
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
		m_instanceComp = new InstanceComparator();
		m_graph = new Graph();
	}
	
	
	public Vector<double[]> GetPurityAndVardiff()
	{
		Vector<double[]> returnVector = new Vector<double[]>();
		
		m_Tree.GetPurityAndVardiff(returnVector);
		
		double[] mean = new double[2];
		int leafWithPurity = 0;
		for(int i = 0; i < returnVector.size(); i++)
		{
			//Only use leafs with labeled and unlabeled data
			if(returnVector.get(i)[0] != -1)
			{
				mean[0] += returnVector.get(i)[0];
				leafWithPurity ++;
			}
			mean[1] += returnVector.get(i)[1];
		}
		mean[0] /= leafWithPurity;
		mean[1] /= returnVector.size();
		returnVector.addElement(mean);
		
		return returnVector;
	}
	
	public double CalculateRandIndex()
	{
		Vector<Instances> FPInstances = new Vector<Instances>();
		Vector<int[]> TPFPvalues = new Vector<int[]>();
		Vector<double[]> leafClassAndVariance = new Vector<double[]>();
		double randIndex = 0.0;
		double precision = 0.0;
		double recall = 0.0;
		double betaSquare = 1;
		
		m_Tree.GetLeafInstances(FPInstances, TPFPvalues);
		m_Tree.GetLeafClassAndVariance(leafClassAndVariance);
		int FN, TP, FP;
		int impurity;
		
		impurity = FN = TP = FP = 0;
		for(int i = 0; i < FPInstances.size(); i++)
		{
			precision = recall = 0;
			for(int j = 0; j < FPInstances.size(); j++)
			{
				impurity = 0;
				if(i != j)	
					for(int k = 0; k < FPInstances.get(j).numInstances(); k++)
						
						if(leafClassAndVariance.get(i)[0] - leafClassAndVariance.get(i)[1] < FPInstances.get(j).get(k).classValue() && 
								FPInstances.get(j).get(k).classValue() < leafClassAndVariance.get(i)[0] + leafClassAndVariance.get(i)[1])
						{
							impurity ++;
						}
				FN += Utilities.CalculateCombination(impurity, 2);
			}
			
			TP += TPFPvalues.get(i)[0];
			FP += TPFPvalues.get(i)[1];
			
		}

		precision = (double)TP / (TP + FP);
		recall = (double)TP / (TP + FN);
		randIndex = (1+betaSquare)*(precision*recall/((betaSquare*precision)+recall));

		
		return randIndex; // FPInstances.size();
	}
	
	public double[][] CalculateDistanceMatrix()
	{
		Vector<double[]> clusterCenter = new Vector<double[]>();
		
		m_Tree.FindLeafCenters(clusterCenter);
		
		
		m_leafDistanceMatrix = new double[clusterCenter.size()][clusterCenter.size()];

		for(int i = 0; i < m_leafDistanceMatrix.length; i++)
		{
		
			for(int j = 0; j < m_leafDistanceMatrix.length; j++)
			{
				double length = 0.0;
				for(int k = 0; k < clusterCenter.get(i).length; k++)
					length += Math.pow(clusterCenter.get(i)[k] - clusterCenter.get(j)[k], 2);
					
				Math.sqrt(length);
				m_leafDistanceMatrix[i][j] = length;
			}
		}
		return m_leafDistanceMatrix;
	}
	
	public double[] CalculateCorrelationPercentage()
	{
		//CalculateDistanceMatrixAndRandMatrix();
		Vector<double[][]> covarianceMatrix = new Vector<double[][]>();
		double[][] correlation;
		m_Tree.FindCovarianceMatrices(covarianceMatrix);
		
		double totalCorrelation[] = {0.0, 0.0};
		for(int i = 0; i < covarianceMatrix.size(); i++)
		{
			correlation = Utilities.NormalizeMatrix(covarianceMatrix.get(i)).clone();
			double det = Utilities.CalculateDeterminant(correlation);
			double det2 = Utilities.CalculateDeterminant(covarianceMatrix.get(i));
			if(det < -1 || det > 1)
			{
				System.out.println("" + Arrays.toString(correlation[0]) + "\n");
				System.out.println("" + Arrays.toString(correlation[1]) + "\n");
				System.out.println("" + Arrays.toString(correlation[2]) + "\n");
				System.out.println("" + Arrays.toString(correlation[3]) + "\n");
				System.out.println("\n");
			}
			totalCorrelation[0] += det;
			totalCorrelation[1] += det2;
		}
		totalCorrelation[0] /= covarianceMatrix.size();
		totalCorrelation[1] /= covarianceMatrix.size();
		
		return totalCorrelation;
	}
	
	public double[] distributionForInstance(Instance instance) throws Exception {

	    if (m_zeroR != null) {
	      return m_zeroR.distributionForInstance(instance);
	    } else {
	      return m_Tree.distributionForInstance(instance);
	    }
	  }
	
	public void buildClassifier(Instances p_labeledData, Instances p_unlabeledData) throws Exception {

		m_graph.Init();
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
	      m_MinVarianceProp * trainVariance, -1, 0);

	    // Backfit if required
	    if (labeledBackfit != null) {
	      m_Tree.backfitData(labeledBackfit); //TODO change to handle two instances
	    }
	    m_plotter.Display2dPlot();
	    
	    Utilities.g_numTrees++;
	    System.out.println("Tree: " + Utilities.g_numTrees  + " Finished!\n");

	    //Will become the worst instance, aka the instance that should be sent to active learning
	    //Instance ins = null;
	    double[] dist = {0};
	    Instance ins = m_graph.CalculateHighestUncertaintyAndPropagateLabels(dist);
	    System.out.println("GRAPH HAS BEEN GRAPHIFIED");
	    System.out.println("Average error rate of transduction: " + m_graph.GetAverageErrorRate());
	    
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
	
	  /**
	   * Computes variance for subsets.
	   * 
	   * @param s
	   * @param sS
	   * @param sumOfWeights
	   * @return the variance
	   */
	  protected static double variance(double[] s, double[] sS,
	    double[] sumOfWeights) {

	    double var = 0;

	    for (int i = 0; i < s.length; i++) {
	      if (sumOfWeights[i] > 0) {
	        var += singleVariance(s[i], sS[i], sumOfWeights[i]);
	      }
	    }

	    return var;
	  }

	  /**
	   * Computes the variance for a single set
	   * 
	   * @param s
	   * @param sS
	   * @param weight the weight
	   * @return the variance
	   */
	  protected static double singleVariance(double s, double sS, double weight) {

		  double ret = sS - ((s * s) / weight);
	    return ret;
	  }
	
	protected class InnerTree extends Tree
	{
		//Variables used for cluster analysis and semi-supervision
		double[][] m_covarianceMatrix = null;
		double[][] m_correlationMatrix = null;
		Instances m_FPInstances = null; 
		double[] m_center = null;
		double m_purity;
		double m_varianceDiff;
		double m_classVariance;
		int m_TP, m_FP;
		int m_id = 0;
		double m_alpha = 0.0;

		
		//Weka implemented variables
		protected InnerTree[] m_Successors;
		
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
		 
		 public void GetLeafInstances(Vector<Instances> p_instanceVector, Vector<int[]> p_TPFP)
		 {
			 if(m_Attribute == -1)
			 {
				 p_instanceVector.add(m_FPInstances);
				 int[] temp = new int[]{m_TP, m_FP};
				 p_TPFP.add(temp);
			 }
			 else
			 {
				 m_Successors[0].GetLeafInstances(p_instanceVector, p_TPFP);
				 m_Successors[1].GetLeafInstances(p_instanceVector, p_TPFP);
			 }
		 }
		 
		 public void GetLeafClassAndVariance(Vector<double[]> p_returnVector)
		 {
			 if(m_Attribute == -1)
			 {
				 double[] values = {m_ClassDistribution[0], m_classVariance};
				 p_returnVector.add(values);
			 }
			 else
			 {
				 m_Successors[0].GetPurityAndVardiff(p_returnVector);
				 m_Successors[1].GetPurityAndVardiff(p_returnVector);
			 }
		 }
		 
		 public void FindLeafCenters(Vector<double[]> p_center)
		 {
			 if(m_Attribute == -1)
				 p_center.add(m_center);
			 else
			 {
				 m_Successors[0].FindLeafCenters(p_center);
				 m_Successors[1].FindLeafCenters(p_center);
			 }
		 }
		 
		 public void FindCovarianceMatrices(Vector<double[][]> p_matricies)
		 {
			 if(m_Attribute == -1)
				 p_matricies.add(m_covarianceMatrix);
			 else
			 {
				 m_Successors[0].FindCovarianceMatrices(p_matricies);
				 m_Successors[1].FindCovarianceMatrices(p_matricies);
			 }
		 }
		 
		 public void GetPurityAndVardiff(Vector<double[]> p_returnVector)
		 {
			 if(m_Attribute == -1)
			 {
				 double[] values = {m_purity, m_varianceDiff};
				 p_returnVector.add(values);
			 }
			 else
			 {
				 m_Successors[0].GetPurityAndVardiff(p_returnVector);
				 m_Successors[1].GetPurityAndVardiff(p_returnVector);
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
		      double minVariance, int p_parentId, int p_myId) throws Exception {
			m_id = p_myId;

			
			m_alpha = (double)p_unlabeledData.numInstances() / (p_labeledData.numInstances() + p_unlabeledData.numInstances());

			m_center = new double[p_unlabeledData.numAttributes()];
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
		          totalSumOfWeights);
		      }

		      // System.err.println("Total weight " + totalWeight);
		      // double sum = Utils.sum(classProbs);
		      if ((p_totalWeight < 2 * m_MinNum && p_unlabeledData.numInstances() < 2) ||

		        // Numeric case
		        (p_labeledData.numInstances() > 1 && (priorVar) / p_totalWeight < minVariance)

		        
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
		        Instances instances = new Instances(p_labeledData);
		        instances.addAll(p_unlabeledData);
		        m_covarianceMatrix = new double[instances.numAttributes()-1][instances.numAttributes()-1];
		        Utilities.CalculateCovarianceMatrix(instances, m_covarianceMatrix, m_center);


		        m_graph.AddLeaf(p_labeledData, p_unlabeledData, m_covarianceMatrix, p_parentId, m_id);

		        
		        if(Utilities.g_clusterAnalysis)
		        	PerformLeafAnalysis(p_labeledData, p_unlabeledData);
		        
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
		        //TODO TELL THE GRAPH THAT IT NEEDS TO MAKE A "PARENT"
		        int[] child = new int[2];
		        child[0] = -1;
		        child[1] = -1;
		        Debugger.DebugPrint("=====ID: " + m_id + " ======", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		        for (int i = 0; i < bestDists.length; i++) 
		        {
		        	child[i] = ++m_counter;
		          m_Successors[i] = new InnerTree();
		          m_Successors[i].buildTree(subsets[i], unlabeledSubset[i], bestDists[i], p_attIndicesWindow,
		            p_labeledData.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i],
		            p_random, p_depth + 1, minVariance, m_id, child[i]);		          	 
		        }
		        Debugger.DebugPrint("=====END: " + m_id + " ======", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		        m_graph.AddParent(m_id, p_parentId, child[0], child[1]);
		        // If all successors are non-empty, we don't need to store the class
		        // distribution
		        boolean emptySuccessor = false;
		        int empty = 0;
		        for (int i = 0; i < subsets.length; i++) {
		          if (m_Successors[i].m_ClassDistribution == null) {
		            emptySuccessor = true;		    
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
		    	  m_Attribute = -1;
		    	  Instances instances = new Instances(p_labeledData);
			      instances.addAll(p_unlabeledData);
			      m_covarianceMatrix = new double[instances.numAttributes()-1][instances.numAttributes()-1];
			      Utilities.CalculateCovarianceMatrix( instances, m_covarianceMatrix, m_center);
			      
			      if(Utilities.g_clusterAnalysis)
			    	  PerformLeafAnalysis(p_labeledData, p_unlabeledData);
			      

			      m_graph.AddLeaf(p_labeledData, p_unlabeledData, m_covarianceMatrix, p_parentId, m_id);

			      
				  m_plotter.Set2dPlotValues(p_unlabeledData, p_labeledData);


		      }
		    }

		//calculate Purity and VarianceDiff
		private void PerformLeafAnalysis(Instances p_labeledData,
				Instances p_unlabeledData) {
			
			m_FPInstances = new Instances(p_labeledData);
			double labeledMean = 0.0;
			double unlabeledMean = 0.0;
			p_unlabeledData.setClassIndex(p_unlabeledData.numAttributes() - 1);
			
			if(p_labeledData.numInstances() != 0 && p_unlabeledData.numInstances() != 0)
			{
			    for(int i = 0; i < p_labeledData.numInstances(); i++)
			    {
			    	labeledMean += p_labeledData.instance(i).classValue();
			    }
			    labeledMean /= p_labeledData.numInstances();
			    
			    for(int i = 0; i < p_unlabeledData.numInstances(); i++)
			    {
			    	unlabeledMean += p_unlabeledData.instance(i).classValue();
			    }
			    unlabeledMean /= p_unlabeledData.numInstances();
			    
			    m_classVariance = 0.0;
			    for(int i = 0; i <  p_labeledData.numInstances(); i++)
			    {
			    	m_classVariance += Math.pow(p_labeledData.instance(i).classValue() - labeledMean, 2);
			    }
			    //Divide by n if only one instance, otherwise divide by n-1
			    m_classVariance /= p_labeledData.numInstances() == 1 ? 1 : (p_labeledData.numInstances() - 1);
			    
			    double unlabaledVariance = 0.0;
			    
			    for(int i = 0; i < p_unlabeledData.numInstances(); i++)
			    {
			    	if((m_ClassDistribution[0] - m_classVariance) < p_unlabeledData.instance(i).classValue() && p_unlabeledData.instance(i).classValue() < (m_ClassDistribution[0] + m_classVariance))
			    		m_purity ++;
			    	else
			    		m_FPInstances.add(p_unlabeledData.instance(i));
			    	unlabaledVariance += Math.pow(p_unlabeledData.instance(i).classValue() - unlabeledMean, 2);
			    	
			    }
			    //Divide by n if only one instance, otherwise divide by n-1
			    unlabaledVariance /= p_unlabeledData.numInstances() == 1 ? 1 : (p_unlabeledData.numInstances() - 1);
			    
			    m_varianceDiff = Math.abs(m_classVariance - unlabaledVariance);
			    m_TP = Utilities.CalculateCombination((int)m_purity, 2);
			    m_FP = Utilities.CalculateCombination(p_unlabeledData.numInstances(), 2) - m_TP;
			    m_purity /= p_unlabeledData.numInstances(); 
			}
			else
			{
				m_purity = -1;
			}
			p_unlabeledData.setClassIndex(-1);
		}
		
		protected double numericDistribution(double[][] props, double[][][] dists,
			      int att, double[][] subsetWeights, Instances p_labeledData, Instances p_unlabeledData, double[] vals)
			      throws Exception {
				
			      double splitPoint = Double.NaN;
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
			        	
			          if (inst.value(att) > currSplit) {
			        	double k = variance(currSums, currSumSquared,
					              currSumOfWeights);
			        	double c = (m_alpha * Covariance(clusterData.numInstances(), splitData(clusterData, inst.value(att), att)));
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
			      Instances clusterInstances = new Instances(p_labeledData);
			      clusterInstances.addAll(p_unlabeledData);
			      clusterInstances.setClassIndex(-1);
			      double clusterPrior = SingleCovariance(clusterInstances);
			      double clusterVar = Covariance(clusterInstances.numInstances(), splitData(clusterInstances, splitPoint, att));
			      double gain = (priorVar - var) + m_alpha *(clusterPrior - clusterVar);

			      // Return distribution and split point
			      subsetWeights[att] = sumOfWeights;
			      dists[0] = dist;
			      vals[att] = gain;

			      return splitPoint;
			    }
		
		private double Covariance(int p_sumParentInstances, Instances[] p_instances) throws Exception
		{
			double hejhoppiklingonskogen = 0.0, parentByChild = 0.0, singleResult = 0.0;
			Debugger.DebugPrint("Entering Covariance", Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			Debugger.DebugPrint("SumParents = " + p_sumParentInstances + "\n" + "Sum child1 = " + p_instances[0].numInstances() + "\n" + "Sum child2 = " + p_instances[1].numInstances(),
								Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			for(int i = 0; i < 2; i++)
			{
				parentByChild = (double)p_instances[i].numInstances() / (double)p_sumParentInstances;
				singleResult =  SingleCovariance(p_instances[i]);			
				hejhoppiklingonskogen += (parentByChild * singleResult) ;
				Debugger.DebugPrint("Covariance value= " + hejhoppiklingonskogen, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			}
			Debugger.DebugPrint("Leaving Covariance", Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			return hejhoppiklingonskogen;
		}
		
		private double SingleCovariance(Instances p_instances) throws Exception
		{

			double[][] covarianceMatrix = new double[p_instances.numAttributes() -1][p_instances.numAttributes() - 1];
			Utilities.CalculateCovarianceMatrix(p_instances, covarianceMatrix, m_center);
			
			double det = Utilities.CalculateDeterminant(covarianceMatrix);
			Debugger.DebugPrint("Determinant: "+ det, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			det = Math.abs(det);
			
			if(det <= 0)
				return 0.0;
			double ret = (Math.log(det)/Math.log(2));
			return ret;
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
	
