import java.util.Vector;

import weka.core.Instance;


public class Graph
{
	Vector<Point> m_points;
	Graph()
	{
		
	}
	public void Init()
	{
		m_points = new Vector<Point>();
	}
	public double CalculateHighestUncertainty(Instance p_outInstance)
	{
		
		
		
		return 0.0;
	}
	
	
	private class Point
	{
		Point()
		{
			
		}
		Point(Instance p_instance, boolean p_labeled, int p_index)
		{
			m_instance = p_instance;
			m_labeled = p_labeled;
			m_index = p_index;
			m_edges = new Vector<Edge>();
		}
		public boolean m_labeled;
		public int m_index;
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