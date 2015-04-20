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
		for(int i = 0; i < m_validationSet.numInstances(); i++)
		{
			double prediction = m_model.classifyInstance(m_validationSet.instance(i));
			
			tempMAE += Math.abs(prediction - m_validationSet.instance(i).classValue());
			tempMAPE += Math.abs(prediction - m_validationSet.instance(i).classValue()) / m_validationSet.instance(i).classValue();
		}
		m_MAE = tempMAE / m_validationSet.numInstances();
		m_MAPE = tempMAPE / m_validationSet.numInstances();
	}
	
	double GetMAE()
	{
		return m_MAE;
	}
	
	double GetMAPE()
	{
		return m_MAPE;
	}
	
}