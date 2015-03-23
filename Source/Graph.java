import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.core.matrix.SingularValueDecomposition;




public class Graph implements Serializable
{
	
	Vector<InnerGraph> m_innerGraphs;
	Vector<double[][]> m_covarianeMatrices;
	Graph()
	{
		
	}
	void Init()
	{
		m_covarianeMatrices = new Vector<double[][]>();		
		m_innerGraphs = new Vector<InnerGraph>();
	}
	
	/*public void GetInstances(Instances p_retInstances)
	{
		for(int i = 0; i < m_Points.size(); i++)
			p_retInstances.add(m_Points.elementAt(i).m_instance);
		
	}
	*/
	
	public void AddCluster(Instances p_labeled, Instances p_unlabeled, double[][] p_covariance, int p_parent, int p_id)
	{
		InnerGraph temp = new InnerGraph(p_parent, p_id);
		temp.AddCluster(p_labeled, p_unlabeled, p_covariance);
		m_innerGraphs.add(temp);
	}
	
	
	//http://rosettacode.org/wiki/Dijkstra%27s_algorithm#C.2B.2B
	
	


	
	//TODO FIX INVERSE MATRIX CALC
	


	//==================INTERNAL STRUCTS=======================================
	private class InnerGraph
	{
		Vector<Point> m_Points;
		int m_id;
		int m_parentId;
		Vector<Integer> m_labeledIndices;
		InnerGraph(int p_parentId, int p_id)
		{
			m_id = p_id;
			m_parentId = p_parentId;
		}
		public void Init()
		{
			m_Points = new Vector<Point>();
			m_labeledIndices = new Vector<Integer>();
		}
		public void AddCluster(Instances p_labeled, Instances p_unlabeled, double[][] p_covariance)
		{
			int index = m_covarianeMatrices.size();
			m_covarianeMatrices.add(p_covariance);
			Instances tempL = new Instances(p_labeled);
			Instances tempU = new Instances(p_unlabeled);
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
		private void ConstructEdges(Point p_currPoint)
		{
			double[] currPointArray, pointArray;
			//Index that the current point will have after it's added
			int myIndex = m_Points.size();
			
			
			//If labeled ignore last attribute since it is a label
			currPointArray  = new double [p_currPoint.m_instance.numAttributes() -1];
			for(int i = 0; i < currPointArray.length; i++)
				currPointArray[i] = p_currPoint.m_instance.toDoubleArray()[i];
			
			//since it is a complete graph we will need edges to all other points
			for(int i = 0; i < m_Points.size(); i++)
			{
				//If labeled ignore last attribute since it is a label
				pointArray =  new double [m_Points.elementAt(i).m_instance.numAttributes() -1];
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
				Edge edge2 = new Edge();
				edge2.m_pointIndex1 = i;
				edge2.m_pointIndex2 = myIndex;
				edge2.m_weight = mahalanobis;
				m_Points.elementAt(i).m_edges.add(edge2);
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
				if(m_Points.elementAt(i).m_labeled)
					continue;
				//Calculate distance to each label
				
				//TODO No need to iterate over each label, just return the äntire arräy yå
				double[] distances = Dijkstra(i);
				for(int j = 0; j < m_labeledIndices.size(); j++)
				{
					totalDist += distances[j];
					 if(distances[j] < localShortest)
					 {
						 localShortest = distances[j];
					 }
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
				//WHY CAN'T YOU SET A BLOODY CLASS VALUE TO AN INSTANCE GRRRRRR
				//m_Points.elementAt(i).m_instance.setClassValue(label);
				//I guess this would work the same though, TODO ask kim if it's true.
				System.out.println("Checked point: " + i + "of: " + m_Points.size());
				m_Points.elementAt(i).m_instance.setValue(m_Points.elementAt(i).m_instance.numAttributes()-1, label);
			}
			
			return retVal;
		}
		private double[] Dijkstra(int p_start)
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
			double[] retMat = new double[m_labeledIndices.size()];
			for(int i = 0; i < m_labeledIndices.size(); i++)
				retMat[i] = minDist[m_labeledIndices.elementAt(i)];
			return retMat;
		}
		
		private double CalculateMahalanobisDistance(double[] p_first, double[] p_second, int p_covMatIndex)
		{
			double retVal = 0;
			
			//retval = D^T * M^-1 * D
			double[] distanceVec = new double[p_first.length];
			Utilities.Subtract(p_first, p_second, distanceVec);
			//double[][] matrix = new double[m_covarianeMatrices.elementAt(p_covMatIndex).length][];
			//Deep copy matrix
			Matrix matrix = Matrix.constructWithCopy(m_covarianeMatrices.elementAt(p_covMatIndex));
			SingularValueDecomposition SVD = new SingularValueDecomposition(matrix);
			Matrix S,V,U;
			S = SVD.getS();
			V = SVD.getV();
			U = SVD.getU();
			//calculate tolerance
			double tolerance = Utilities.g_machineEpsilion * Math.max(S.getColumnDimension(), S.getRowDimension()) * S.norm2();
			
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
					
	    	double[] tempVec = new double[distanceVec.length];
	    	double[][] inverseMat = matrix.getArray();
			//D^T * M^-1
	    	//double[][] inverse = GaussJordan(matrix);
			for(int i = 0; i < inverseMat.length; i++)
			{
				for(int j = 0; j < inverseMat[i].length; j++)
				{
					tempVec[i] += distanceVec[j] * inverseMat[j][i]; 
				}
			}
			//(D^T * M^-1) * D
			for(int i= 0; i < tempVec.length; i++)
			{
				retVal += tempVec[i] * distanceVec[i]; 
			}
			if(retVal < 0)
				System.out.println("HOUSTON WE HAVE A PROBLEM");
			return retVal;
		}
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
}