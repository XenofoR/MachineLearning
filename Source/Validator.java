import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.trees.RandomForest;

import java.math.*;
public class Validator
{
	Instances[] m_validationSet = null;
	int m_trainingIndex, m_validationIndex;
	double m_MAE = 0.0;
	double m_MAPE = 0.0;
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
		m_trainingIndex = 0;
		m_validationIndex = 1;

	}
	
	Instances GetTrainingSet()
	{
		return m_validationSet[m_trainingIndex];
	}
	
	void ValidateModel(RandomForest p_model) throws Exception
	{
		double tempMAE = 0.0;
		double tempMAPE = 0.0;
		//double tempSMAPE = 0.0;
		double[] predictions = new double[m_validationSet[m_validationIndex].numInstances()];
		for(int i = 0; i < m_validationSet[m_validationIndex].numInstances(); i++)
		{
			double prediction = p_model.classifyInstance(m_validationSet[m_validationIndex].instance(i));
			
			predictions[i] = Math.abs(prediction - m_validationSet[m_validationIndex].instance(i).classValue());
			tempMAE += Math.abs(prediction - m_validationSet[m_validationIndex].instance(i).classValue());
			;//tempSMAPE += prediction + m_validationSet[m_validationIndex].instance(i).classValue();
			tempMAPE += Math.abs(prediction - m_validationSet[m_validationIndex].instance(i).classValue()) / (m_validationSet[m_validationIndex].instance(i).classValue() + prediction);
		}
		
		
		m_MAE = tempMAE / m_validationSet[m_validationIndex].numInstances();
		m_MAPE = tempMAPE / m_validationSet[m_validationIndex].numInstances();
		
		for(int i = 0; i < m_validationSet[m_validationIndex].numInstances(); i++)
			m_errorVariance += Math.pow(predictions[i] - m_MAE, 2);
		
		m_errorVariance /= m_validationSet[m_validationIndex].numInstances();
		m_errorDiviation = Math.sqrt(m_errorVariance);
		
		m_trainingIndex = m_validationIndex;
		m_validationIndex++;
		m_validationIndex = (m_validationIndex == m_validationSet.length) ? 0 : m_validationIndex;

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