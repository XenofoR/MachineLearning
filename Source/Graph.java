import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;
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
	public void GetInstances(Instances p_retInstances)
	{
		for(int i = 0; i < m_Points.size(); i++)
			p_retInstances.add(m_Points.elementAt(i).m_instance);
		
	}
	
	public void AddCluster(Instances p_labeled, Instances p_unlabeled, double[][] p_covariance)
	{
		
		int index = m_covarianeMatrices.size();
		m_covarianeMatrices.add(p_covariance);
		Instances tempL = new Instances(p_labeled);
		Instances tempU = new Instances(p_unlabeled);
		tempL.addAll(p_labeled);
		tempU.addAll(p_unlabeled);
		for(int i = 0; i < tempL.size(); i++)
		{
			//Remember that this is a labeled instance
			m_labeledIndices.add(m_Points.size());
			Point point = new Point(tempL.instance(i), true, index);
			//Build edges
			ConstructEdges(point);
			m_Points.add(point);	
		}
		for(int i = 0; i < tempU.size(); i++)
		{
			Point point = new Point(tempU.instance(i), false, index);
			//Build edges
			ConstructEdges(point);
			m_Points.add(point);	
		}
	}
	
	public double CalculateHighestUncertaintyAndPropagateLabels(Instance p_outInstance)
	{
		double retVal = 0;
		double localShortest = 0, totalDist = 0, label = 0;
		//TODO This is the place to insert threading in this algorithm if we intend to do so
		for(int i = 0; i < m_Points.size(); i++)
		{
			localShortest = Double.MAX_VALUE;
			totalDist = 0;
			label = 0;
			double [] distances = new double[m_labeledIndices.size()];
			if(m_Points.elementAt(i).m_labeled)
				continue;
			//Calculate distance to each label
			for(int j = 0; j < m_labeledIndices.size(); j++)
			{
				//TODO No need to iterate over each label, just return the äntire arräy yå
				double temp = Dijkstra(i, m_labeledIndices.elementAt(j));
				totalDist += temp;
				 if(temp < localShortest)
				 {
					 localShortest = temp;
				 }
				 distances[j] = temp;
			}
			//Save the longest shortest path
			if(localShortest > retVal)
			{
				retVal = localShortest;
				p_outInstance = m_Points.elementAt(i).m_instance;
			}
			//Calculate propagated label for instance
			for(int j = 0; j < distances.length; j++)
			{
				label += (distances[j] / totalDist) * m_Points.elementAt(m_labeledIndices.elementAt(j)).m_instance.classValue();
			}
			m_Points.elementAt(i).m_instance.setClassValue(label);
		}
		
		return retVal;
	}
	//http://rosettacode.org/wiki/Dijkstra%27s_algorithm#C.2B.2B
	private double Dijkstra(int p_start, int p_target)
	{
		//Shortest distance to each point from origin
		double[] minDist = new double[m_Points.size()];
		Arrays.fill(minDist, Double.MAX_VALUE);
		minDist[p_start] = 0; //Origin dist
		Set<Pair<Point, Double>> queue = new LinkedHashSet<Pair<Point, Double>>();
		queue.add(new Pair<Point, Double>(m_Points.elementAt(p_start),0.0));
		
		while(!queue.isEmpty())
		{
			// Java.... why are you not c++
			Pair<Point, Double> itr = queue.iterator().next();
			Point currPoint =  itr.GetFirst();
			double currDist =  itr.GetSecond();
			queue.remove(itr);
			//Visit currPoints edges
			for(Iterator<Edge> it = currPoint.m_edges.iterator(); it.hasNext();)
			{
				Edge temp = (Edge) it.next();
				Point target = m_Points.elementAt(temp.m_pointIndex2);
				double localDist = temp.m_weight;
				double totalDist = currDist + localDist;
				if(totalDist < minDist[temp.m_pointIndex2])
				{
					queue.remove(new Pair<Point, Double>(target, minDist[temp.m_pointIndex2]));
					
					minDist[temp.m_pointIndex2] = totalDist;
					queue.add(new Pair<Point, Double>(target, totalDist));
				}
			}
		}
		return minDist[p_target];
	}
	
	private void InvertMatrix(int p_matrixIndex, double[][] p_output)
	{
		//http://en.wikipedia.org/wiki/Gaussian_elimination#Finding_the_inverse_of_a_matrix
		//Note to self, only use ? operator...yeeessss.yeeesssssssssssssssssss
	}

	private void ConstructEdges(Point p_currPoint)
	{
		double[] currPointArray, pointArray;
		//Index that the current point will have after it's added
		int myIndex = m_Points.size();
		
		
		//If labeled ignore last attribute since it is a label
		currPointArray  = p_currPoint.m_labeled ? new double [p_currPoint.m_instance.numAttributes() -1] : new double [p_currPoint.m_instance.numAttributes()];
		//Change from instance to array
		for(int i = 0; i < currPointArray.length; i++)
			currPointArray[i] = p_currPoint.m_instance.toDoubleArray()[i];
		
		//since it is a complete graph we will need edges to all other points
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
			//TODO check so that this doesn't change the edge added to currpoint
			edge.m_pointIndex1 = i;
			edge.m_pointIndex2 = myIndex;
			m_Points.elementAt(i).m_edges.add(edge);
		}
	}
	//TODO FIX INVERSE MATRIX CALC
	private double CalculateMahalanobisDistance(double[] p_first, double[] p_second, int p_covMatIndex)
	{
		double retVal = 0;
		
		//retval = D^T * M^-1 * D
		double[] distanceVec = new double[p_first.length];
		Utilities.Subtract(p_first, p_second, distanceVec);
		double[][] matrix = new double[m_covarianeMatrices.elementAt(p_covMatIndex).length][];
		//Deep copy matrix
    	for(int i = 0; i < m_covarianeMatrices.elementAt(p_covMatIndex).length; i++)
    		matrix[i] = Arrays.copyOf(m_covarianeMatrices.elementAt(p_covMatIndex)[i], m_covarianeMatrices.elementAt(p_covMatIndex)[i].length);
		 
    	double[] tempVec = new double[distanceVec.length];
		//D^T * M^-1
		for(int i = 0; i < matrix.length; i++)
		{
			for(int j = 0; j < matrix[i].length; j++)
			{
				tempVec[i] += distanceVec[j] * matrix[j][i]; // TODO THIS SHOULD BE INVERSE MATRIX
			}
		}
		//(D^T * M^-1) * D
		for(int i= 0; i < tempVec.length; i++)
		{
			retVal += tempVec[i] * distanceVec[i]; 
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
	private class Pair<F,S>
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