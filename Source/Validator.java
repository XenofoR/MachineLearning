import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.trees.RandomForest;

import java.math.*;
public class Validator
{
	Instances[] m_validationSet = null;
	int m_validationIndex;
	double m_MAE = 0.0;
	double m_MSE = 0.0;
	double m_SMAPE = 0.0;
	double m_errorVariance = 0.0;
	double m_errorDiviation = 0.0;
	Validator(int p_numFolds)
	{
		m_validationSet = new Instances[p_numFolds];
	}
	
	void Init(Instances[] p_validationSet)//, RandomForest p_model)
	{
		for(int i = 0; i < p_validationSet.length; i++)
		{
			m_validationSet[i] = p_validationSet[i];
			m_validationSet[i].setClassIndex(m_validationSet[i].numAttributes()-1);
		}
		m_validationIndex = -1;

	}
	
	void GetTrainingSet(Instances p_trainingSet)
	{
		m_validationIndex++;
		m_validationIndex = (m_validationIndex == m_validationSet.length) ? 0 : m_validationIndex;
		for(int i = 0; i < m_validationSet.length; i++)
		{
			if(i != m_validationIndex)
				p_trainingSet.addAll(m_validationSet[i]);
		}
	}
	
	void ValidateModel(RandomForest p_model) throws Exception
	{
		double tempMAE = 0.0;
		double tempMSE = 0.0;
		double tempSMAPE = 0.0;
		double tempSMAPEdiv = 0.0;
		double[] predictions = new double[m_validationSet[m_validationIndex].numInstances()];
		for(int i = 0; i < m_validationSet[m_validationIndex].numInstances(); i++)
		{
			double prediction = p_model.classifyInstance(m_validationSet[m_validationIndex].instance(i));
			
			predictions[i] = Math.abs(prediction - m_validationSet[m_validationIndex].instance(i).classValue());
			tempMAE += Math.abs(prediction - m_validationSet[m_validationIndex].instance(i).classValue());
			tempMSE += Math.sqrt(prediction - m_validationSet[m_validationIndex].instance(i).classValue());
			tempSMAPE += Math.abs(prediction - m_validationSet[m_validationIndex].instance(i).classValue()) / ((Math.abs(prediction) + (Math.abs(m_validationSet[m_validationIndex].instance(i).classValue()))) / 2);
		}
		
		m_MAE = tempMAE / m_validationSet[m_validationIndex].numInstances();
		m_MSE = tempMSE / m_validationSet[m_validationIndex].numInstances();
		m_SMAPE = tempSMAPE / m_validationSet[m_validationIndex].numInstances();
		
		for(int i = 0; i < m_validationSet[m_validationIndex].numInstances(); i++)
			m_errorVariance += Math.pow(predictions[i] - m_MAE, 2);
		
		m_errorVariance /= m_validationSet[m_validationIndex].numInstances();
		m_errorDiviation = Math.sqrt(m_errorVariance);
		
		

	}
	
	double GetMAE()
	{
		return m_MAE;
	}
	
	double GetMSE()
	{
		return m_MSE;
	}
	
	double GetMAPE()
	{
		return m_SMAPE;
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