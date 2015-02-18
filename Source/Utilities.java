import weka.core.Instance;
import weka.core.Instances;


public class Utilities
{
	
	
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
	
	//http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
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
		Scale(p_destination, 1.0/(p_instances.numInstances()-1));
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
	
	//http://introcs.cs.princeton.edu/java/95linear/Cholesky.java.html
	public static double cholesky(double[][] A) {
		double det = 0.0;
        if (!isSquare(A)) {
            throw new RuntimeException("Matrix is not square");
        }
        if (!isSymmetric(A)) {
            throw new RuntimeException("Matrix is not symmetric");
        }

        int N  = A.length;
        double[][] L = new double[N][N];

        for (int i = 0; i < N; i++)  {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) L[i][i] = Math.sqrt(A[i][i] - sum);
                else        L[i][j] = 1.0 / L[j][j] * (A[i][j] - sum);
            }
            if (L[i][i] <= 0) {
                throw new RuntimeException("Matrix not positive definite");
            }
        }
        det = L[0][0];
        for(int i = 1; i < L.length; i++)
        {
        	det*= L[i][i];
        }
        
        return det;
    }
	
	private static boolean isSymmetric(double[][] A) {
        int N = A.length;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < i; j++) {
                if (A[i][j] != A[j][i]) return false;
            }
        }
        return true;
    }

    // is symmetric
    private static boolean isSquare(double[][] A) {
        int N = A.length;
        for (int i = 0; i < N; i++) {
            if (A[i].length != N) return false;
        }
        return true;
    }
    
    
	
}