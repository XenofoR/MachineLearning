import java.io.File;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

public abstract class RegressionForest extends weka.classifiers.trees.RandomForest
{
	
	public RegressionForest()
	{
		super();
	}
	
	public void buildClassifier(Instances p_data) throws Exception
	{
		super.buildClassifier(p_data);
	}
	abstract public String GetInfo();
	
	
	
}
