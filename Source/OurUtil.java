import java.util.Arrays;
import java.math.BigInteger;


import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.core.matrix.SingularValueDecomposition;


public class OurUtil
{
	static boolean g_debug;
	static boolean g_clusterAnalysis;
	static boolean g_useMahalanobis;
	static boolean g_useWeightedTransduction;
	static boolean g_forceCompleteGraph;
	static double g_machineEpsilion;
	static double g_alphaValue;
	static int g_numTrees;
    static int g_activeNumber;
    static ActiveTechnique g_activeTech;
    
	public enum ActiveTechnique
	{
		Random,
		Worst,
		AllWorst,
		Ensemble,
		NONE
	}
	
	static public int CalculateCombination(int p_numenator, int p_denomenator)
	{
		BigInteger interNum = new BigInteger(p_numenator+"");
		BigInteger interDen = new BigInteger(p_denomenator+"");
		
		
		for(int i = p_numenator - 1; i > 0; i--)
		{
			BigInteger temp = new BigInteger(i+"");
			interNum.multiply(temp);
		}
		
		for(int i = p_denomenator - 1; i > 0; i--)
		{
			BigInteger temp = new BigInteger(i+"");
			interDen.multiply(temp);
		}
		
		for(int i = p_numenator - p_denomenator - 1; i > 0; i--)
		{
			BigInteger temp = new BigInteger(i+"");
			interDen.multiply(temp);
		}
		
 		return interNum.divide(interDen).intValue();
	}
	//http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
	static public void CalculateCovarianceMatrix(Instances p_instances, double[][] p_destination, double[] p_meanDestination, boolean p_unlabeled)
	{
			
		double[] mean = Mean(p_instances, p_unlabeled);
		if(p_meanDestination != null)
			for(int  i = 0; i < mean.length; i++)
				p_meanDestination[i] = mean[i];
		for(int i = 0; i < p_instances.numInstances() ;i++)
		{			
			double[] tempVector;
			double[][] tempMatrix; 
			if(p_unlabeled)
			{
				tempVector = new double[p_instances.instance(i).numAttributes() -1];
				tempMatrix = new double[p_instances.numAttributes() - 1][p_instances.numAttributes() - 1];
			}
			else
			{
				tempVector = new double[p_instances.instance(i).numAttributes()];
				tempMatrix = new double[p_instances.numAttributes()][p_instances.numAttributes()];
			}
			Subtract(p_instances.instance(i).toDoubleArray(), mean, tempVector);
			OuterProduct(tempVector, tempVector, tempMatrix);
			Add(p_destination, tempMatrix, p_destination);

		}
		double scaleVar = 1.0/(p_instances.numInstances()-1);
		if(!Double.isInfinite(scaleVar) && !Double.isNaN(scaleVar))
		{
			Scale(p_destination, scaleVar );
		}
		else
		{
			//for(int i = 0; i < p_destination.length;i++)
				//p_destination[i][i] = 1;
			Debugger.DebugPrint("CalculateCovarianceMatrix only recieved 1 instance  results will be a zero matrix", Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			Scale(p_destination, 1 );
		}
		
	}
	
	static public double[][] CalculateInverse(double[][] p_matrix)
	{
		//Deep copy matrix
		Matrix matrix = Matrix.constructWithCopy(p_matrix);
		SingularValueDecomposition SVD = new SingularValueDecomposition(matrix);
		Matrix S,V,U;
		S = SVD.getS();
		V = SVD.getV();
		U = SVD.getU();
		//calculate tolerance
		double tolerance = g_machineEpsilion * Math.max(S.getColumnDimension(), S.getRowDimension()) * S.norm2();
		
		//Pseudo invert S
		for(int i = 0; i < S.getColumnDimension(); i++)
			if(S.get(i, i) >= tolerance) //tolerance should remove floating point errors on variables smaller than a really small value
				S.set(i, i, 1/S.get(i, i));
			else
				S.set(i, i, 0);
		S = S.transpose();
		//m^+ = V * S^+ * U'
		matrix = V.times(S);
		matrix = matrix.times(U.transpose());
		
		return matrix.getArray();
	}
	
	static public double CalculateDeterminant(double[][] p_matrix)
	{
		//In case we ever implement another method for calculating
		return CalcDeterminantWithLU(p_matrix);
	}
	
	static public double[][] NormalizeMatrix(double[][] p_matrix)
	{
		
		
		double[][] temp = new double[p_matrix.length][];			
		for(int i = 0; i < p_matrix.length; i++)
		{
			temp[i] = Arrays.copyOf(p_matrix[i], p_matrix[i].length );
			for(int j = 0; j < p_matrix.length; j++)
			{
				if(p_matrix[i][i] != 0 && p_matrix[j][j] != 0)
					temp[i][j] /= Math.sqrt(p_matrix[i][i]*p_matrix[j][j]);
				else
				{
					temp[i][j] = 0;
				}
			}
		}
		return temp;
	}
	
	
	private static void OuterProduct(double[] p_row, double[] p_column, double[][] p_destination)
	{
		for(int i = 0; i < p_row.length; i++)
			for(int j = 0; j < p_column.length; j++)
				p_destination[i][j] = p_row[i] * p_column[j];
	}
	//Subtract p_value from each attribute in p_instance and save results in p_destination
	public static void Subtract(double[] p_source, double[] p_value, double[] p_destination)
	{
		for(int i = 0; i < p_destination.length; i++)
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
	private static double[] Mean(Instances p_instances, boolean p_unlabeled)
	{
		double[] retVal;
		if(p_unlabeled)
			retVal = new double[p_instances.numAttributes() -1];
		else
			retVal = new double[p_instances.numAttributes()];
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
	

	

    
    private static double CalcDeterminantWithLU(double[][] p_matrix)
    {
    	double det = 0.0;
    	double[][] matrix = new double[p_matrix.length][];
    	int pivSign = 1;
    	int[] piv = new int[matrix.length];
    	
    	for(int i = 0; i < matrix.length; i++)
    		piv[i] = i;
    	
    	double[] LUrowi;
    	double[] LUcolj = new double[matrix.length];
    	//Deepcopy matrix
    	for(int i = 0; i < p_matrix.length; i++)
    		matrix[i] = Arrays.copyOf(p_matrix[i], p_matrix[i].length);
    	// Outer loop
    	for(int j = 0; j < matrix.length; j++)
    	{
    		
    		for(int i = 0; i < matrix.length; i++)
    		{
    		LUcolj[i] = matrix[i][j];
    		}
    		
    		//Apply previous transformations
    		for(int i = 0; i < matrix.length; i++)
    		{
    			LUrowi = matrix[i];
    			
    			
    			//Dot product
    			int kmax = Math.min(i, j);
    			double s = 0.0;
    			for(int k = 0; k < kmax; k++)
    			{
    				s += LUrowi[k] * LUcolj[k];
    			}
    			
    			LUrowi[j] = LUcolj[i] -= s;
    		}
    		
    		//Find pivot and exchange if necessary
    		
    		int p = j;
    		for(int i = j + 1; i < matrix.length; i++)
    		{
    			if(Math.abs(LUcolj[i]) > Math.abs(LUcolj[p]))
    				p = i;
    		}
    		if(p != j)
    		{
    			for(int k = 0; k < matrix.length; k++)
    			{
    				double t = matrix[p][k];
    				matrix[p][k] = matrix[j][k];
    				matrix[j][k] = t;
    			}
    			int k = piv[p];
    			piv[p] = piv[j];
    			piv[j] = k;
    			pivSign = -pivSign;
    		}
    		
    		//Compute multipliers
    		if(j < matrix.length & matrix[j][j] != 0.0)
    		{
    			for(int i = j + 1; i < matrix.length; i++)
    			{
    				matrix[i][j] /= matrix[j][j];
    			}
    		}
    	}
    	det = matrix[0][0];
    	for(int i = 1; i < matrix.length; i++)
    		det *= matrix[i][i];
 
    	return det * pivSign;
    }
    
    private static boolean isNonsingular(double[][] p_matrix) {
        for (int j = 0; j < p_matrix.length; j++) {
          if (p_matrix[j][j] == 0)
            return false;
        }
        return true;
      }
    
   static public class Pair<F,S>
	{
		private F first;
		private S second;
		
		public Pair(F p_first, S p_second)
		{
			first = p_first;
			second = p_second;
		}
		public F GetFirst()
		{
			return first;
		}
		public S GetSecond()
		{
			return second;
		}
		
	}
	
}