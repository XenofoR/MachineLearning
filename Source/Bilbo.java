/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Bagging.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.ArrayList;

import javax.swing.text.Utilities;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.core.AdditionalMeasureProducer;
import weka.core.Aggregateable;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.PartitionGenerator;

/**
 <!-- globalinfo-start -->
 * Class for bagging a classifier to reduce variance. Can do classification and regression depending on the base learner. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). Bagging predictors. Machine Learning. 24(2):123-140.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {Bagging predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 * 
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 * 
 * <pre> -represent-copies-using-weights
 *  Represent copies of instances using weights rather than explicitly.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 * 
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 * 
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * <pre> -P
 *  No pruning.</pre>
 * 
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 * 
 * <pre> -I
 *  Initial class value count (default 0)</pre>
 * 
 * <pre> -R
 *  Spread initial count over all class values (i.e. don't use 1 per value)</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (len@reeltwo.com)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class Bilbo
  extends NewRPISCE 
  implements WeightedInstancesHandler, AdditionalMeasureProducer,
             TechnicalInformationHandler, PartitionGenerator, Aggregateable<Bilbo> {

  /** for serialization */
  static final long serialVersionUID = -115879962237199703L;
  
  /** The size of each bag sample, as a percentage of the training size */
  protected int m_BagSizePercent = 100;

  /** Whether to calculate the out of bag error */
  protected boolean m_CalcOutOfBag = false;

  /** Whether to represent copies of instances using weights rather than explicitly */
  protected boolean m_RepresentUsingWeights = false;

  /** The out of bag error that has been calculated */
  protected double m_OutOfBagError;  
  protected double m_MaxOutOfBagError;  
  Instances toOracle;
  /**
   * Constructor.
   */
  public Bilbo() {
    
    m_Classifier = new NewTree();
  }
  
  public Vector<Vector<double[]>> GetPurityAndVardiff()
	{
	  Vector<Vector<double[]>> returnVector = new Vector<Vector<double[]>>();
	  double[] mean = new double[2];
	  for(int i = 0; i < m_Classifiers.length; i++)
	  {
		  returnVector.add(((NewTree)m_Classifiers[i]).GetPurityAndVardiff());
		  mean[0] += returnVector.lastElement().lastElement()[0];
		  mean[1] += returnVector.lastElement().lastElement()[1];
	  }
	  mean[0] /= returnVector.size();
	  mean[1] /= returnVector.size();
	  
	  returnVector.add(new Vector<double[]>());
	  
	  returnVector.lastElement().add(mean);
	  return returnVector;
	}
  
  public double CalculateRandIndex()
  {
	  double RandIndex = 0.0;
	  
	  for(int i = 0; i < m_Classifiers.length; i++)
		  RandIndex += ((NewTree)m_Classifiers[i]).CalculateRandIndex();
	  
	  return RandIndex / m_Classifiers.length;
  }
  
  public double[] CalculateCorrelationPercentage()
  {
	  double index[] = {0.0, 0.0};
	  for(int i = 0; i < m_Classifiers.length; i++)
	  {
		  double[] temp = ((NewTree)m_Classifiers[i]).CalculateCorrelationPercentage();
		  index[0] += temp[0];
		  index[1] += temp[1];
	  }
	  
	  index[0] /= m_Classifiers.length;
	  index[1] /= m_Classifiers.length;
	  
	  return index;
  }
  
  public double GetAverageTransductionError()
  {
	  double retVal = 0;
	  for(int i = 0; i < m_Classifiers.length; i++)
	  {
		  retVal += ((NewTree)m_Classifiers[i]).GetTransductionError();
	  }
	  retVal /= m_Classifiers.length;
	  return retVal;
  }
  
  public Instances getOracleData()
  {
	  return toOracle;
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Class for bagging a classifier to reduce variance. Can do classification "
      + "and regression depending on the base learner. \n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "1996");
    result.setValue(Field.TITLE, "Bagging predictors");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "24");
    result.setValue(Field.NUMBER, "2");
    result.setValue(Field.PAGES, "123-140");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {
    
    return "weka.classifiers.trees.REPTree";
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<Option>(3);

    newVector.addElement(new Option(
              "\tSize of each bag, as a percentage of the\n" 
              + "\ttraining set size. (default 100)",
              "P", 1, "-P"));
    newVector.addElement(new Option(
              "\tCalculate the out of bag error.",
              "O", 0, "-O"));
    newVector.addElement(new Option(
              "\tRepresent copies of instances using weights rather than explicitly.",
              "-represent-copies-using-weights", 0, "-represent-copies-using-weights"));

    newVector.addAll(Collections.list(super.listOptions()));
 
    return newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P
   *  Size of each bag, as a percentage of the
   *  training set size. (default 100)</pre>
   * 
   * <pre> -O
   *  Calculate the out of bag error.</pre>
   * 
   * <pre> -represent-copies-using-weights
   *  Represent copies of instances using weights rather than explicitly.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -num-slots &lt;num&gt;
   *  Number of execution slots.
   *  (default 1 - i.e. no parallelism)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.REPTree)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.REPTree:
   * </pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf (default 2).</pre>
   * 
   * <pre> -V &lt;minimum variance for split&gt;
   *  Set minimum numeric class variance proportion
   *  of train variance for split (default 1e-3).</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Number of folds for reduced error pruning (default 3).</pre>
   * 
   * <pre> -S &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   * <pre> -P
   *  No pruning.</pre>
   * 
   * <pre> -L
   *  Maximum tree depth (default -1, no maximum)</pre>
   * 
   * <pre> -I
   *  Initial class value count (default 0)</pre>
   * 
   * <pre> -R
   *  Spread initial count over all class values (i.e. don't use 1 per value)</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {

    String bagSize = Utils.getOption('P', options);
    if (bagSize.length() != 0) {
      setBagSizePercent(Integer.parseInt(bagSize));
    } else {
      setBagSizePercent(100);
    }

    setCalcOutOfBag(Utils.getFlag('O', options));

    setRepresentCopiesUsingWeights(Utils.getFlag("represent-copies-using-weights", options));

    super.setOptions(options);
    
    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String [] getOptions() {

    Vector<String> options = new Vector<String>();
    
    options.add("-P"); 
    options.add("" + getBagSizePercent());

    if (getCalcOutOfBag()) { 
        options.add("-O");
    }

    if (getRepresentCopiesUsingWeights()) {
        options.add("-represent-copies-using-weights");
    }

    Collections.addAll(options, super.getOptions());
    
    return options.toArray(new String[0]);
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String bagSizePercentTipText() {
    return "Size of each bag, as a percentage of the training set size.";
  }

  /**
   * Gets the size of each bag, as a percentage of the training set size.
   *
   * @return the bag size, as a percentage.
   */
  public int getBagSizePercent() {

    return m_BagSizePercent;
  }
  
  /**
   * Sets the size of each bag, as a percentage of the training set size.
   *
   * @param newBagSizePercent the bag size, as a percentage.
   */
  public void setBagSizePercent(int newBagSizePercent) {

    m_BagSizePercent = newBagSizePercent;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String representCopiesUsingWeightsTipText() {
    return "Whether to represent copies of instances using weights rather than explicitly.";
  }

  /**
   * Set whether copies of instances are represented using weights rather than explicitly.
   *
   * @param representUsingWeights whether to represent copies using weights
   */
  public void setRepresentCopiesUsingWeights(boolean representUsingWeights) {

    m_RepresentUsingWeights = representUsingWeights;
  }

  /**
   * Get whether copies of instances are represented using weights rather than explicitly.
   *
   * @return whether the out of bag error is calculated
   */
  public boolean getRepresentCopiesUsingWeights() {

    return m_RepresentUsingWeights;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String calcOutOfBagTipText() {
    return "Whether the out-of-bag error is calculated.";
  }

  /**
   * Set whether the out of bag error is calculated.
   *
   * @param calcOutOfBag whether to calculate the out of bag error
   */
  public void setCalcOutOfBag(boolean calcOutOfBag) {

    m_CalcOutOfBag = calcOutOfBag;
  }

  /**
   * Get whether the out of bag error is calculated.
   *
   * @return whether the out of bag error is calculated
   */
  public boolean getCalcOutOfBag() {

    return m_CalcOutOfBag;
  }

  /**
   * Gets the out of bag error that was calculated as the classifier
   * was built.
   *
   * @return the out of bag error 
   */
  public double measureOutOfBagError() {
    
    return m_OutOfBagError;
  }
  
  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  @Override
  public Enumeration<String> enumerateMeasures() {
    
    Vector<String> newVector = new Vector<String>(1);
    newVector.addElement("measureOutOfBagError");
    return newVector.elements();
  }
  
  /**
   * Returns the value of the named measure.
   *
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String additionalMeasureName) {
    
    if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
      return measureOutOfBagError();
    }
    else {throw new IllegalArgumentException(additionalMeasureName 
					     + " not supported (Bagging)");
    }
  }
  
  protected Random m_random;
  protected boolean[][] m_inBag;

  public Long[] GetAndAverageGraphTime()
  {
	  Long[] retVal = new Long[3];
	  Long tT, lT, tlT;
	  tT = lT = tlT = 0L;
	  Long[] cont; 	  
	  for(int i = 0; i < m_Classifiers.length; i++)
	  {
		  cont = ((NewTree)m_Classifiers[i]).GetGraphTime();
		  lT += cont[0];
		  tlT += cont[1];
		  tT += cont[2];
	  }
	  retVal[0] = lT / m_Classifiers.length;
	  retVal[1] = tlT / m_Classifiers.length;
	  retVal[2] = tT / m_Classifiers.length;
	  return retVal;
  }
  
  /**
   * Returns a training set for a particular iteration.
   * 
   * @param iteration the number of the iteration for the requested training set.
   * @return the training set for the supplied iteration number
   * @throws Exception if something goes wrong when generating a training set.
   */
  @Override
  protected synchronized Instances getTrainingSet(Instances p_data, int iteration) throws Exception {
    int bagSize = (int) (p_data.numInstances() * (m_BagSizePercent / 100.0));
    Instances bagData = null;
    Random r = new Random(m_Seed + iteration);

    // create the in-bag dataset
    if (m_CalcOutOfBag && p_data.classIndex() != -1) {
      m_inBag[iteration] = new boolean[p_data.numInstances()];
      bagData = p_data.resampleWithWeights(r, m_inBag[iteration], getRepresentCopiesUsingWeights());
    } else {
      bagData = p_data.resampleWithWeights(r, getRepresentCopiesUsingWeights());
      if (bagSize < p_data.numInstances()) {
        bagData.randomize(r);
        Instances newBagData = new Instances(bagData, 0, bagSize);
        bagData = newBagData;
      }
    }
    
    return bagData;
  }
  
  /**
   * Bagging method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data, Instances p_unlabeledData) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // Has user asked to represent copies using weights?
    if (getRepresentCopiesUsingWeights() && !(m_Classifier instanceof WeightedInstancesHandler)) {
      throw new IllegalArgumentException("Cannot represent copies using weights when " +
                                         "base learner in bagging does not implement " +
                                         "WeightedInstancesHandler.");
    }

    // get fresh Instances object
    m_data = new Instances(data);
    m_unlabeledData = new Instances(p_unlabeledData);
    super.buildClassifier(m_data);

    if (m_CalcOutOfBag && (m_BagSizePercent != 100)) {
      throw new IllegalArgumentException("Bag size needs to be 100% if " +
					 "out-of-bag error is to be calculated!");
    }

    m_random = new Random(m_Seed);
    
    m_inBag = null;
    if (m_CalcOutOfBag)
      m_inBag = new boolean[m_Classifiers.length][];
    
    for (int j = 0; j < m_Classifiers.length; j++) {      
      if (m_Classifier instanceof Randomizable) {
	((Randomizable) m_Classifiers[j]).setSeed(m_random.nextInt());
      }
    }
     
    buildClassifiers();
    Instances inst = new Instances(m_data , 0);
    
	for(int i = 0 ; i < m_Classifiers.length; i++)
	{
		inst.clear();
		Debugger.DebugPrint("Forest done! starting Induction \n", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
    	((NewTree)m_Classifiers[i]).GetTransductedInstances(inst);
    	((NewTree)m_Classifiers[i]).DoInduction(inst);
    	Debugger.DebugPrint("Induction finished! \n", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
    	// Ehm, do something boyski TODO: Remove this comment
	}

    // calc OOB error?
    if (getCalcOutOfBag()) {
      m_OutOfBagError = CalculateOutOfBagError();
    }
    else {
      m_OutOfBagError = 0;
    }

    if(m_unlabeledData.numInstances() != 0)
    {
    		toOracle = new Instances(m_unlabeledData,0);
    		switch(OurUtil.g_activeTech)
    		{
    		case Random:
    			SelectAtRandom(OurUtil.g_activeNumber, toOracle);
    			break;
    		case Worst:
    			SelectWorst(toOracle);
    			break;
    		case AllWorst:
    			SelectAllWorst(toOracle);
    			break;
    		case Ensemble:
    			SelectEnsemble(OurUtil.g_activeNumber, toOracle);
    			break;
    		case NONE:
			default:
				throw new Exception("No or NONE active learning tech chosen, please pick one of the following: Random, Worst, Allworst, Ensemble");
    		}
    }
    // save memory
    m_data = null;
    m_unlabeledData = null;
    inst = null;
  }
  
  

  public void SelectAtRandom(int p_numRan, Instances p_retInst)
  {
	  Random rand = new Random();
	 
	  for(int i = 0; i < p_numRan; i++)
	  {
		  int index = rand.nextInt(m_unlabeledData.size());
		  p_retInst.add(m_unlabeledData.instance(index));
		  m_unlabeledData.remove(index);
	  }
  }
  public void SelectWorst(Instances p_retInst)
  {
	  double worstDist = 0;
	  Instance inst = null;
	  for(int i = 0; i < m_Classifiers.length; i++)
	  {
		  if(((NewTree) m_Classifiers[i]).GetWorstDistance() > worstDist)
		  {
			  worstDist = ((NewTree) m_Classifiers[i]).GetWorstDistance();
			  inst = ((NewTree) m_Classifiers[i]).GetWorstInstance();
		  }
	  }
	  InstanceComparator comp = new InstanceComparator();
	  for(int i = 0; i < m_unlabeledData.size(); i++)
		  if(comp.compare(inst, m_unlabeledData.instance(i)) == 0)
		  {
			  m_unlabeledData.remove(i);
			  break;
		  }
	  p_retInst.add(inst);
  }
  
  public void SelectEnsemble(int p_number, Instances p_retInst)
  {
	  Instances worst = new Instances(m_unlabeledData,0);
	  SelectAllWorst(worst);
	  InstanceComparator comp = new InstanceComparator();
	  comp.setIncludeClass(false);
	  int[] counter = new int[worst.size()];
	  for(int i = 0 ; i < worst.size(); i++)
		  for(int j = i+1 ; j < worst.size(); j++)
			  if(comp.compare(worst.instance(i), worst.instance(j)) == 0)
			  {
				  counter[i]++;
				  worst.remove(j);
				  j--;
			  }
	  int[] topChoices = new int[p_number];
	  Arrays.fill(topChoices, -1);
	  for(int i = 0; i < p_number; i++)
	  {
		  if(i == worst.numInstances())
			  break;
		  int highestVote = -1;
		  for(int j = 0; j < counter.length; j++)
		  {
			  if(counter[j] > highestVote)
			  {
				  topChoices[i] = j;
				  highestVote = counter[j];
			  }
		  }
		  counter[topChoices[i]] = -1;
		  p_retInst.add(worst.instance(topChoices[i]));
	  }
	  
	  worst = null;
  }
  
  public void SelectAllWorst(Instances p_retInst)
  {
	  for(int i = 0; i < m_Classifiers.length; i++)
	  {
		  if(m_Classifiers[i] == null)
			  continue;
		  p_retInst.add(((NewTree) m_Classifiers[i]).GetWorstInstance());
	  }
	  InstanceComparator comp = new InstanceComparator();
	  for(int j = 0; j < p_retInst.size(); j++)
		  for(int i = 0; i < m_unlabeledData.size(); i++)
		  {
			  if(comp.compare(p_retInst.instance(j), m_unlabeledData.instance(i)) == 0)
					  m_unlabeledData.remove(i);
		  }
  }

  
  public double CalculateOutOfBagError() throws Exception
  {
	  double outOfBagCount = 0.0;
      double errorSum = 0.0;
      boolean numeric = m_data.classAttribute().isNumeric();
      double retVal = 0.0;
      for (int i = 0; i < m_data.numInstances(); i++) {
        double vote;
        double[] votes;
        if (numeric)
          votes = new double[1];
        else
          votes = new double[m_data.numClasses()];
        
        // determine predictions for instance
        int voteCount = 0;
        for (int j = 0; j < m_Classifiers.length; j++) {
          if (m_inBag[j][i])
            continue;
          
         
          if (numeric) {
            double pred = ((NewTree)m_Classifiers[j]).classifyInstance(m_data.instance(i));
            if (!Utils.isMissingValue(pred)) {
              votes[0] += pred;
              voteCount++;
            }
          } else {
            voteCount++;
            double[] newProbs = ((NewTree)m_Classifiers[j]).distributionForInstance(m_data.instance(i));
            // average the probability estimates
            for (int k = 0; k < newProbs.length; k++) {
              votes[k] += newProbs[k];
            }
          }
        }
        
        // "vote"
        if (numeric) {
          if (voteCount == 0) {
            vote = Utils.missingValue();
          } else {
            vote = votes[0] / voteCount;    // average
          }
        } else {
          if (Utils.eq(Utils.sum(votes), 0)) {            
            vote = Utils.missingValue();
          } else {
            vote = Utils.maxIndex(votes);   // predicted class
            Utils.normalize(votes);
          }
        }
        
        // error for instance
        if (!Utils.isMissingValue(vote) && !m_data.instance(i).classIsMissing()) {
          outOfBagCount += m_data.instance(i).weight();
          if (numeric) {
        	  errorSum += StrictMath.abs(vote - m_data.instance(i).classValue() 
                      * m_data.instance(i).weight()) ;
            /*errorSum += (StrictMath.abs(vote - m_data.instance(i).classValue()) / m_data.instance(i).classValue() 
              * m_data.instance(i).weight()) ;*/
          }
          else {
            if (vote != m_data.instance(i).classValue())
              errorSum += m_data.instance(i).weight();
          }
        }
      }
      
      if (outOfBagCount > 0) {
    	  retVal = errorSum / outOfBagCount;
      }
	  return retVal;
  }
  
  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    double [] sums = new double [instance.numClasses()], newProbs; 
    
    double numPreds = 0;
    for (int i = 0; i < m_NumIterations; i++) {
      if (instance.classAttribute().isNumeric() == true) {
        double pred = ((NewTree)m_Classifiers[i]).classifyInstance(instance);
        if (!Utils.isMissingValue(pred)) {
          sums[0] += pred;
          numPreds++;
        }
      } else {
	newProbs = ((NewTree)m_Classifiers[i]).distributionForInstance(instance);
	for (int j = 0; j < newProbs.length; j++)
	  sums[j] += newProbs[j];
      }
    }
    if (instance.classAttribute().isNumeric() == true) {
      if (numPreds == 0) {
        sums[0] = Utils.missingValue();
      } else {
        sums[0] /= numPreds;
      }
      return sums;
    } else if (Utils.eq(Utils.sum(sums), 0)) {
      return sums;
    } else {
      Utils.normalize(sums);
      return sums;
    }
  }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  @Override
  public String toString() {
    
    if (m_Classifiers == null) {
      return "Bagging: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("All the base classifiers: \n\n");
    for (int i = 0; i < m_Classifiers.length; i++)
      text.append(m_Classifiers[i].toString() + "\n\n");
    
    if (m_CalcOutOfBag) {
      text.append("Out of bag error: "
		  + Utils.doubleToString(m_OutOfBagError, 4)
		  + "\n\n");
    }

    return text.toString();
  }
  
  /**
   * Builds the classifier to generate a partition.
   */
  @Override
  public void generatePartition(Instances data) throws Exception {
    
    if (m_Classifier instanceof PartitionGenerator)
      buildClassifier(data);
    else throw new Exception("Classifier: " + getClassifierSpec()
			     + " cannot generate a partition");
  }
  
  /**
   * Computes an array that indicates leaf membership
   */
  @Override
  public double[] getMembershipValues(Instance inst) throws Exception {
    
    if (m_Classifier instanceof PartitionGenerator) {
      ArrayList<double[]> al = new ArrayList<double[]>();
      int size = 0;
      for (int i = 0; i < m_Classifiers.length; i++) {
        double[] r = ((PartitionGenerator)m_Classifiers[i]).
          getMembershipValues(inst);
        size += r.length;
        al.add(r);
      }
      double[] values = new double[size];
      int pos = 0;
      for (double[] v: al) {
        System.arraycopy(v, 0, values, pos, v.length);
        pos += v.length;
      }
      return values;
    } else throw new Exception("Classifier: " + getClassifierSpec()
                               + " cannot generate a partition");
  }
  
  /**
   * Returns the number of elements in the partition.
   */
  @Override
  public int numElements() throws Exception {
    
    if (m_Classifier instanceof PartitionGenerator) {
      int size = 0;
      for (int i = 0; i < m_Classifiers.length; i++) {
        size += ((PartitionGenerator)m_Classifiers[i]).numElements();
      }
      return size;
    } else throw new Exception("Classifier: " + getClassifierSpec()
                               + " cannot generate a partition");
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

  /**
   HI KIM IAM A COMMENT. I NEED YOU TO SEND YOUR MONY TO ME IN CHINA: IN CHINA I CAN BVRING YOU VERY BIG HOUSE. 
   EVEN BIGGER THAN U THINK. IT IS CALLED THE FORBIDDEN CITY AND IT CAN BE ALL YOURS
   */

  
  protected List<Classifier> m_classifiersCache;

  /**
   * Aggregate an object with this one
   * 
   * @param toAggregate the object to aggregate
   * @return the result of aggregation
   * @throws Exception if the supplied object can't be aggregated for some
   *           reason
   */
  @Override
  public Bilbo aggregate(Bilbo toAggregate) throws Exception {
    if (!m_Classifier.getClass().isAssignableFrom(toAggregate.m_Classifier.getClass())) {
      throw new Exception("Can't aggregate because base classifiers differ");
    }
    
    if (m_classifiersCache == null) {
      m_classifiersCache = new ArrayList<Classifier>();
      m_classifiersCache.addAll(Arrays.asList(m_Classifiers));
    }
    m_classifiersCache.addAll(Arrays.asList(toAggregate.m_Classifiers));
    
    return this;
  }

  /**
   * Call to complete the aggregation process. Allows implementers to do any
   * final processing based on how many objects were aggregated.
   * 
   * @throws Exception if the aggregation can't be finalized for some reason
   */
  @Override
  public void finalizeAggregation() throws Exception {    
    m_Classifiers = m_classifiersCache.toArray(new NewTree[1]);
    m_NumIterations = m_Classifiers.length;
    
    m_classifiersCache = null;
  }
}
