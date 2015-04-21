import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.trees.RandomForest;

import java.math.*;
public class Validator
{
	RandomForest m_model = null;
	Instances m_validationSet = null;
	double m_variance = 0.0;
	double m_precision = 0.0;
	double m_MAE = 0.0;
	double m_MAPE = 0.0;
	double m_SMAPE = 0.0;
	Validator()
	{
		
	}
	
	void Init(Instances p_validationSet, RandomForest p_model)
	{
		m_validationSet = p_validationSet;
		m_validationSet.setClassIndex(m_validationSet.numAttributes()-1);
		m_model = p_model;
		
		double mean = m_validationSet.meanOrMode(m_validationSet.classAttribute());
		
		for(int i = 0; i < m_validationSet.numInstances(); i++)
			m_variance += Math.pow(m_validationSet.instance(i).classValue() - mean,2);
		m_variance /= m_validationSet.numInstances();
	}
	
	void ValidateModel() throws Exception
	{
		double tempMAE = 0.0;
		double tempMAPE = 0.0;
		double tempSMAPE = 0.0;
		double truePositive = 0.0;
		double falsePositive = 0.0;
		double trueNegative = 0.0;
		for(int i = 0; i < m_validationSet.numInstances(); i++)
		{
			double prediction = m_model.classifyInstance(m_validationSet.instance(i));
			
			if(m_validationSet.instance(i).classValue() - m_variance < prediction && prediction < m_validationSet.instance(i).classValue() + m_variance)
			{
				truePositive++;
			}
			else
			{
				falsePositive++;
				//continue;
			}
			
			tempMAE += Math.abs(prediction - m_validationSet.instance(i).classValue());
			tempSMAPE += prediction + m_validationSet.instance(i).classValue();
			tempMAPE += Math.abs(prediction - m_validationSet.instance(i).classValue()) / m_validationSet.instance(i).classValue();
		}
		
		m_precision = truePositive / (truePositive+falsePositive);
		
		m_MAE = tempMAE / m_validationSet.numInstances();
		m_MAPE = tempMAPE / m_validationSet.numInstances();
		m_SMAPE = tempMAE / tempSMAPE;
	}
	
	double GetMAE()
	{
		return m_MAE;
	}
	
	double GetMAPE()
	{
		return m_MAPE;
	}
	
	double GetPrecision()
	{
		return m_precision;
	}
	
	double GetSMAPE()
	{
		return m_SMAPE;
	}
}