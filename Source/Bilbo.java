import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.core.AdditionalMeasureProducer;
import weka.core.Aggregateable;
import weka.core.Instance;
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

//https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/meta/Bagging.java
public class Bilbo extends weka.classifiers.meta.Bagging
{
	 public void buildClassifier(Instances p_labeledData) throws Exception {
		// can classifier handle the data?
	    super.getCapabilities().testWithFail(p_labeledData);
	
	    // Has user asked to represent copies using weights?
	    if (super.getRepresentCopiesUsingWeights() && !(m_Classifier instanceof WeightedInstancesHandler)) {
	      throw new IllegalArgumentException("Cannot represent copies using weights when " +
	                                         "base learner in bagging does not implement " +
	                                         "WeightedInstancesHandler.");
	    }
	
	    // get fresh Instances object
	    m_data = new Instances(p_labeledData);
	   
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
	    
	    super.buildClassifiers();
	    
	    // calc OOB error?
	    if (super.getCalcOutOfBag()) {
	      double outOfBagCount = 0.0;
	      double errorSum = 0.0;
	      boolean numeric = m_data.classAttribute().isNumeric();
	      
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
	            double pred = m_Classifiers[j].classifyInstance(m_data.instance(i));
	            if (!Utils.isMissingValue(pred)) {
	              votes[0] += pred;
	              voteCount++;
	            }
	          } else {
	            voteCount++;
	            double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
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
	            errorSum += StrictMath.abs(vote - m_data.instance(i).classValue()) 
	              * m_data.instance(i).weight();
	          }
	          else {
	            if (vote != m_data.instance(i).classValue())
	              errorSum += m_data.instance(i).weight();
	          }
	        }
	      }
	      
	      if (outOfBagCount > 0) {
	        m_OutOfBagError = errorSum / outOfBagCount;
	      }
	    }
	    else {
	      m_OutOfBagError = 0;
	    }
	    
	    // save memory
	    m_data = null;
  }

}