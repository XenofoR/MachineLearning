import weka.core.Instance;
import weka.core.Instances;


public class Utilities
{
	public enum DebugLevel
	{
		HIGH,
		MEDIUM,
		LOW,
		NONE
	}
	public enum DebugType
	{
		FILE,
		CONSOLE,
		NONE
	}
	
	//http://professorjava.weebly.com/matrix-determinant.html
	static public double determinant(double p_matrix[][])
	 {
		int sum = 0; 
	    int s;
	    if(p_matrix.length==1)
	    {  //bottom case of recursion. size 1 matrix determinant is itself.
	      return(p_matrix[0][0]);
	    }
	    for(int i = 0;i < p_matrix.length; i++)
	    { //finds determinant using row-by-row expansion
	      double[][]smaller= new double[p_matrix.length - 1][p_matrix.length - 1]; //creates smaller matrix- values not in same row, column
	      for(int a = 1; a < p_matrix.length; a++)
	      {
	        for(int b = 0; b < p_matrix.length; b++){
	          if(b<i){
	            smaller[a-1][b]=p_matrix[a][b];
	          }
	          else if(b>i){
	            smaller[a-1][b-1]=p_matrix[a][b];
	          }
	        }
	      }
	      if(i%2==0){ //sign changes based on i
	        s = 1;
	      }
	      else{
	        s =-1;
	      }
	      sum+=s*p_matrix[0][i]*(determinant(smaller)); //recursive step: determinant of larger determined by smaller.
	    }
	    return(sum); //returns determinant value. once stack is finished, returns final determinant.
	}
	
	static public void CalculateCovarianceMatrix(Instances p_instances, double[][] p_destination)
	{
		
		double[] mean = Mean(p_instances);
		for(int i = 0; i < p_instances.numInstances() ;i++)
		{			
			double[] tempVector = new double[p_instances.instance(i).numAttributes() -1];
			Subtract(p_instances.instance(i).toDoubleArray(), mean, tempVector);
			double[][] tempMatrix = new double[p_instances.numAttributes() - 1][p_instances.numAttributes() - 1];
			OuterProduct(tempVector, tempVector, tempMatrix);
			Add(p_destination, tempMatrix, p_destination);
		}
		Scale(p_destination, 1.0/(p_instances.numAttributes()-2));
	}
	
	private static void OuterProduct(double[] p_row, double[] p_column, double[][] p_destination)
	{
		for(int i = 0; i < p_row.length; i++)
			for(int j = 0; j < p_column.length; j++)
				p_destination[i][j] = p_row[i] * p_column[j];
	}
	//Subtract p_value from each attribute in p_instance and save results in p_destination
	private static void Subtract(double[] p_source, double[] p_value, double[] p_destination)
	{
		for(int i = 0; i < p_source.length - 1; i++)
			p_destination[i] = p_source[i] - p_value[i];
	}
	//Adds p_first[][] and p_second[][] and stores it in p_destination
	private static void Add(double[][] p_first, double[][] p_second, double[][] p_destination	)
	{
		for(int i = 0; i < p_first.length; i++)
			for(int j = 0; j < p_first[i].length ; j++)
				p_destination[i][j] = p_first[i][j] +  p_second[i][j];
	}
	//
	private static double MeanOfAttribute(Instances p_instances, int p_index)
	{
		double retVal = 0.0;
		for(int i = 0; i < p_instances.numInstances(); i++)
			retVal += p_instances.instance(i).toDoubleArray()[p_index];
		return (retVal / p_instances.numInstances());
	}
	private static double[] Mean(Instances p_instances)
	{
		double[] retVal = new double[p_instances.numAttributes() -1];
		for(int i = 0; i < p_instances.numAttributes() -1; i++)
		{
			retVal[i] = MeanOfAttribute(p_instances, i);
		}
		
		
		return retVal;
	}
	private static void Scale(double[][] p_matrix, double p_scale)
	{
		for(int i = 0; i < p_matrix.length; i++)
			for(int j = 0; j < p_matrix[i].length; j++)
				p_matrix[i][j] *= p_scale;
	}
}