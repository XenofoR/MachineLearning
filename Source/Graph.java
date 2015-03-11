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
	
	//TODO
	private double CalculateMahalanobisDistance()
	{
		return 0.0;
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