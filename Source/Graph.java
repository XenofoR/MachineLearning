import java.util.Arrays;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;



public class Graph
{
	Vector<Point> m_Points;
	Vector<Integer> m_labeledIndices;
	Vector<double[][]> m_covarianeMatrices;
	Graph()
	{
		
	}
	public void Init()
	{
		m_Points = new Vector<Point>();
		m_labeledIndices = new Vector<Integer>();
		m_covarianeMatrices = new Vector<double[][]>();
	}
	//TODO
	public void AddCluster(Instances p_labeled, Instances p_unlabeled, double[][] p_covariance)
	{
		
		int index = m_covarianeMatrices.size();
		m_covarianeMatrices.add(p_covariance);
		for(int i = 0; i < p_labeled.size(); i++)
		{
			//Remember that this is a labeled instance
			m_labeledIndices.add(m_Points.size());
			Point point = new Point(p_labeled.instance(i), true, index);
			//Build edges
			ConstructEdges(point);
			m_Points.add(point);	
		}
		for(int i = 0; i < p_labeled.size(); i++)
		{
			Point point = new Point(p_unlabeled.instance(i), false, index);
			//Build edges
			ConstructEdges(point);
			m_Points.add(point);	
		}
	}
	
	//TODO
	public double CalculateHighestUncertainty(Instance p_outInstance)
	{
		return 0.0;
	}
	//TODO
	private double Djikstra(int p_unlabeled, int p_labeledIndex)
	{
		return 0.0;
	}
	
	public void ConstructEdges(Point p_currPoint)
	{
		double[] currPointArray, pointArray;
		//Index that the current point will have after it's added
		int myIndex = m_Points.size();
		
		
		//If labeled ignore last attribute since it is a label
		currPointArray  = p_currPoint.m_labeled ? new double [p_currPoint.m_instance.numAttributes() -1] : new double [p_currPoint.m_instance.numAttributes()];
		//Change from instance to array
		for(int i = 0; i < currPointArray.length; i++)
			currPointArray[i] = p_currPoint.m_instance.toDoubleArray()[i];
		
		for(int i = 0; i < m_Points.size(); i++)
		{
			//If labeled ignore last attribute since it is a label
			pointArray = m_Points.elementAt(i).m_labeled ? new double [m_Points.elementAt(i).m_instance.numAttributes() -1] : new double [m_Points.elementAt(i).m_instance.numAttributes()];
			//Change from instance to array
			for(int j = 0; j < pointArray.length; j++)
				pointArray[j] = m_Points.elementAt(i).m_instance.toDoubleArray()[j];
			
			
			Edge edge = new Edge();
			
			edge.m_pointIndex1 = myIndex;
			
			edge.m_pointIndex2 = i;
			
			//calculate the mean of the mahalanobis distance using both covariance matrices. P.200 in Criminisi 2011
			double mahalanobis = CalculateMahalanobisDistance(currPointArray, pointArray, p_currPoint.m_covarianceIndex);
			mahalanobis +=  CalculateMahalanobisDistance(currPointArray, pointArray, m_Points.elementAt(i).m_covarianceIndex);
			mahalanobis /= 2;
			
			edge.m_weight = mahalanobis;
			p_currPoint.m_edges.add(edge);
			m_Points.elementAt(i).m_edges.add(edge);
		}
	}
	//TODO
	private double CalculateMahalanobisDistance(double[] p_first, double[] p_second, int p_covMatIndex)
	{
		double retVal = 0;
		
		//retval = D^T * M^-1 * D
		double[] distanceVec = new double[p_first.length];
		Utilities.Subtract(p_first, p_second, distanceVec);
		double[][] matrix = Arrays.copyOf(m_covarianeMatrices.elementAt(p_covMatIndex));
		//D^T * M^-1
		for(int i = 0; i < 20; i++)
		{
			
		}
		
		
		return retVal;
	}
	
	
	
	
	//==================INTERNAL STRUCTS=======================================
	private class Point
	{
		Point()
		{
			m_edges = new Vector<Edge>();
		}
		Point(Instance p_instance, boolean p_labeled, int p_covarianceIndex)
		{
			m_instance = p_instance;
			m_labeled = p_labeled;
			m_covarianceIndex = p_covarianceIndex;
			m_edges = new Vector<Edge>();
		}
		
		public boolean m_labeled;
		public int m_covarianceIndex;
		public Instance m_instance;
		public Vector<Edge> m_edges;
	}
	private class Edge
	{
		Edge()
		{
			
		}
		public int m_pointIndex1, m_pointIndex2;
		//The weight will be the mahalanobis distance between points
		public double m_weight;
	}
}