import java.util.Iterator;
import java.util.Random;


import weka.classifiers.Evaluation;
import weka.core.Instances;
//https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/trees/RandomForest.java
public class SupervisedForest extends weka.classifiers.trees.RandomForest {

	String m_info;
	public SupervisedForest() {
		super();
	}
	
	public void buildClassifier(Instances p_data) throws Exception
	{
		
		super.buildClassifier(p_data);
		
		m_info = toString();
		//Insert Future stuff here?
		
	}
	public String GetInfo()
	{
		return m_info;
	}
	
/*
	void SetData(Instances p_data) throws Exception
	{
		m_structure = p_data;
		
		m_evaluator = new Evaluation(m_structure);
		m_forest.setNumTrees(m_numTrees);
		m_forest.setMaxDepth(m_maxDepth);
	}
	
	String Train() throws Exception {
		
		m_forest.buildClassifier(m_structure);

		return m_forest.toString();
	}
	
	String CrossValidate() throws Exception
	{
		m_evaluator.crossValidateModel(m_forest, m_structure, 10, new Random());
		
		return m_evaluator.toSummaryString();
	}

	double[] Run() throws Exception {
		
		double[] returnValue = new double[m_structure.numInstances()];
		
		for(int i = 0; i < m_structure.numInstances(); i++)
		{
			returnValue[i] = m_forest.classifyInstance(m_structure.get(i));
		}
		
		return returnValue;
	}
*/


}
