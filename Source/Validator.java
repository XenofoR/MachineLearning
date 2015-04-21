import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.trees.RandomForest;

import java.math.*;
public class Validator
{
	RandomForest m_model = null;
	Instances m_validationSet = null;
	double m_MAE = 0.0;
	double m_MAPE = 0.0;
	double m_errorVariance = 0.0;
	double m_errorDiviation = 0.0;
	Validator()
	{
		
	}
	
	void Init(Instances p_validationSet, RandomForest p_model)
	{
		m_validationSet = p_validationSet;
		m_validationSet.setClassIndex(m_validationSet.numAttributes()-1);
		m_model = p_model;

		
	}
	
	void ValidateModel() throws Exception
	{
		double tempMAE = 0.0;
		double tempMAPE = 0.0;
		double tempSMAPE = 0.0;
		double[] predictions = new double[m_validationSet.numInstances()];
		for(int i = 0; i < m_validationSet.numInstances(); i++)
		{
			double prediction = m_model.classifyInstance(m_validationSet.instance(i));
			
			predictions[i] = Math.abs(prediction - m_validationSet.instance(i).classValue());
			tempMAE += Math.abs(prediction - m_validationSet.instance(i).classValue());
			tempSMAPE += prediction + m_validationSet.instance(i).classValue();
			tempMAPE += Math.abs(prediction - m_validationSet.instance(i).classValue()) / m_validationSet.instance(i).classValue();
		}
		
		
		m_MAE = tempMAE / m_validationSet.numInstances();
		m_MAPE = tempMAPE / m_validationSet.numInstances();
		
		for(int i = 0; i < m_validationSet.numInstances(); i++)
			m_errorVariance += Math.pow(predictions[i] - m_MAE, 2);
		
		m_errorVariance /= m_validationSet.numInstances();
		m_errorDiviation = Math.sqrt(m_errorVariance);
	}
	
	double GetMAE()
	{
		return m_MAE;
	}
	
	double GetMAPE()
	{
		return m_MAPE;
	}
	
	double GetErrorVar()
	{
		return m_errorVariance;
	}
	
	double GetErrorDiv()
	{
		return m_errorDiviation;
	}
	
}