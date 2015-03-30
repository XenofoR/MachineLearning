
import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.Vector;


import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.core.matrix.SingularValueDecomposition;




public class Graph implements Serializable
{
	
	Vector<InnerGraph> m_graphs;
	Vector<double[][]> m_covarianeMatrices;
	Vector<Utilities.Pair<Integer, Integer>> m_idToIndexMap;
	boolean m_forceRootMerge = false;
	Graph()
	{
		
	}
	void Init()
	{
		m_covarianeMatrices = new Vector<double[][]>();		
		m_graphs = new Vector<InnerGraph>();
		m_idToIndexMap = new Vector<Utilities.Pair<Integer, Integer>>();
	}
	
	/*public void GetInstances(Instances p_retInstances)
	{
		for(int i = 0; i < m_Points.size(); i++)
			p_retInstances.add(m_Points.elementAt(i).m_instance);
		
	}
	*/
	public double GetAverageErrorRate()
	{
		double retVal = 0;
		double total = 0;
		for(int i = 0; i < m_graphs.size(); i++)
		{
			if((m_graphs.elementAt(i).HasBeenMerged() == false) && m_graphs.elementAt(i).HasLabeled())
				for(int j = 0; j < m_graphs.elementAt(i).m_Points.size(); j++)
				{
					if(m_graphs.elementAt(i).m_Points.elementAt(j).m_labeled == false)
					{
						retVal += m_graphs.elementAt(i).m_Points.elementAt(j).m_errorPercentage;
						total++;
					}
				}
		}
		return retVal/total;
	}
	public void AddLeaf(Instances p_labeled, Instances p_unlabeled, double[][] p_covariance, int p_parentId, int p_id)
	{
		Debugger.DebugPrint("Added leaf node: " + p_id + " With parent: " + p_parentId, Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		InnerGraph graph = new InnerGraph(p_parentId, p_id, -1 , -1);
		graph.Init();
		graph.AddCluster(p_labeled, p_unlabeled, p_covariance);
		Utilities.Pair<Integer, Integer> temp = new Utilities.Pair<Integer, Integer>(p_id, m_graphs.size());
		m_idToIndexMap.add(temp);
		m_graphs.add(graph);
	}
	public void AddParent(int p_id, int p_parentId, int p_childId1, int p_childId2)
	{
		Debugger.DebugPrint("Added split node: " + p_id + " With Parent: " + p_parentId + " And children: " + p_childId1 + " " + p_childId2, Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);

		InnerGraph graph = new InnerGraph(p_parentId, p_id, p_childId1 , p_childId2);
		graph.Init();
		Utilities.Pair<Integer, Integer> temp = new Utilities.Pair<Integer, Integer>(p_id, m_graphs.size());
		m_idToIndexMap.add(temp);
		m_graphs.add(graph);
	}
	public double CalculateHighestUncertaintyAndPropagateLabels(Instance p_outInstance) throws Exception
	{
		if(m_forceRootMerge)
			return UncertaintyCompleteGraph(p_outInstance);
		Instance retInst = null;
		double retVal = 0;
		for(int i = 0; i < m_graphs.size(); i++)
		{
			if(m_graphs.elementAt(i).GetChildren()[0] != -1) // Non-leaf
				continue;
			if(m_graphs.elementAt(i).HasBeenMerged())
				continue;
			Instance temp = null;
			double val = 0;
			if(m_graphs.elementAt(i).HasLabeled())
				val = m_graphs.elementAt(i).CalculateHighestUncertaintyAndPropagateLabels(temp);
			else
			{
				Debugger.DebugPrint("No labeled, going into merge mode", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
				int parent = m_graphs.elementAt(i).GetParentId();
				int index = MergeChildren(parent);
				//parent = MergeChildren(index);
				val = m_graphs.elementAt(index).CalculateHighestUncertaintyAndPropagateLabels(temp);
				
			}
			if(val > retVal)
			{
				retInst = temp;
				retVal = val;
			}
		
			Debugger.DebugPrint("Checked graph: " + i + "of: " + m_graphs.size(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		}
		p_outInstance = retInst;
		return retVal;
	}
	
	public void ForceRootMerge(boolean p_force)
	{
		m_forceRootMerge = p_force;
	}
	private int MergeChildren(int p_id) throws Exception
	{
		if(p_id == -1)
			throw new Exception("MergeException:NoLabeledData");
		int retVal = p_id;
		int index = FindIndexFromId(p_id);
		if(!m_graphs.elementAt(index).m_Points.isEmpty())
			return p_id; //Is leaf or is already merged
		int[] children = m_graphs.elementAt(index).GetChildren();
		
		for(int i = 0; i < children.length; i++)
			MergeChildren(children[i]);
		int childIndex1 = FindIndexFromId(children[0]);
		int childIndex2 = FindIndexFromId(children[1]);
		m_graphs.elementAt(index).MergeClusters(m_graphs.elementAt(childIndex1), m_graphs.elementAt(childIndex2));
		for(int i = 0; i < children.length; i++)
			m_graphs.elementAt(childIndex1).SetHasBeenMerged(true);
		if(!m_graphs.elementAt(index).HasLabeled())
			retVal = MergeChildren(m_graphs.elementAt(index).GetParentId());
		return retVal;
	}
	private double UncertaintyCompleteGraph(Instance p_outInstance) throws Exception
	{
		double retVal = 0;
		int index = FindIndexFromId(0);
		if(m_graphs.elementAt(index).m_Points.isEmpty())
			MergeChildren(0);
		Debugger.DebugPrint("Started Dikjstras on root graph, this is going to take a while", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		retVal = m_graphs.elementAt(index).CalculateHighestUncertaintyAndPropagateLabels(p_outInstance);
		
		Debugger.DebugPrint("Dujkstra funished", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		return retVal;
	}
	private int FindIndexFromId(int p_id) throws Exception
	{
		int retVal = -1;
		for(int i = 0; i < m_idToIndexMap.size(); i++)
			if(p_id == m_idToIndexMap.elementAt(i).GetFirst())
			{
				retVal = m_idToIndexMap.elementAt(i).GetSecond();
				break;
			}
		if(retVal == -1)
			throw new Exception("GraphException:ParentNotFound " + p_id);
		return retVal;
	}
	
	private class InnerGraph
	{
		Vector<Point> m_Points;
		private int m_id;
		private int m_parentId;
		private int[] m_child;
		Vector<Integer> m_labeledIndices;
		private boolean m_hasBeenMerged;
		InnerGraph(int p_parentId, int p_id, int p_child1, int p_child2)
		{
			m_id = p_id;
			m_parentId = p_parentId;
			m_child = new int[2];
			m_child[0] = p_child1;
			m_child[1] = p_child2;
			m_hasBeenMerged = false;
		}
		int GetId()
		{
			return m_id;
		}
		int GetParentId()
		{
			return m_parentId;
		}
		int[] GetChildren()
		{
			return m_child;
		}
		boolean HasLabeled()
		{
			return !m_labeledIndices.isEmpty();
		}
		boolean HasBeenMerged()
		{
			return m_hasBeenMerged;
		}
		void SetHasBeenMerged(boolean p_merged)
		{
			m_hasBeenMerged = p_merged;
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
		//TODO Fix something that makes the graph only do things on the one with labels 
		public void MergeClusters(InnerGraph p_graph1, InnerGraph p_graph2) throws Exception
		{					
			Debugger.DebugPrint("Merging clusters", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			for(int i = 0; i < p_graph1.m_Points.size(); i++)
			{
				Point temp = p_graph1.m_Points.elementAt(i).Clone();
				temp.m_edges.clear();
				if(temp.m_labeled)
					m_labeledIndices.add(m_Points.size());
				
				ConstructEdges(temp);
				m_Points.add(temp);
			}
			for(int i = 0; i < p_graph2.m_Points.size(); i++)
			{
				Point temp = p_graph2.m_Points.elementAt(i).Clone();
				temp.m_edges.clear();
				if(temp.m_labeled)
					m_labeledIndices.add(m_Points.size());
				
				ConstructEdges(temp);
				m_Points.add(temp);
			}
			m_labeledIndices.addAll(p_graph1.m_labeledIndices);
			for(int i = 0; i < m_Points.size(); i++)
				m_Points.elementAt(i).m_edges.clear();
			int offset = p_graph1.m_labeledIndices.size();
			Debugger.DebugPrint("Merge Complete", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			
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
			if(m_labeledIndices.size() == 0)
				return -1;
			
			
			double retVal = 0;
			double localShortest = 0, totalDist = 0, label = 0;
			for(int i = 0; i < m_Points.size(); i++)
			{
				localShortest = Double.MAX_VALUE;
				totalDist = 0;
				label = 0;
				if(m_Points.elementAt(i).m_labeled)
					continue;
				//Calculate distance to each label
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
					double percentage = 1.0 - (distances[j] / totalDist);
					// if 0 we only have 1 labeled and then we will simply apply it directly
					percentage = (percentage == 0) ? 1 : percentage; 
					label +=  percentage * m_Points.elementAt(m_labeledIndices.elementAt(j)).m_instance.classValue();
				}
				//WHY CAN'T YOU SET A BLOODY CLASS VALUE TO AN INSTANCE GRRRRRR
				//m_Points.elementAt(i).m_instance.setClassValue(label);
				//I guess this would work the same though, TODO ask kim if it's true.
				int labelIndex = m_Points.elementAt(i).m_instance.numAttributes()-1;
				double error = Math.abs((m_Points.elementAt(i).m_instance.value(labelIndex) - label)/ (m_Points.elementAt(i).m_instance.value(labelIndex)));
				m_Points.elementAt(i).m_errorPercentage = error;
				m_Points.elementAt(i).m_instance.setValue(m_Points.elementAt(i).m_instance.numAttributes()-1, label);
			}
			
			return retVal;
		}
		private double[] Dijkstra(int p_start)
		{
			//Shortest distance to each point from origin
			double[] minDist = new double[m_Points.size()];
			Arrays.fill(minDist, Double.MAX_VALUE);
			//minDist[p_start] = 0; //Origin dist
			Set<Utilities.Pair<Point, Double>> queue = new LinkedHashSet<Utilities.Pair<Point, Double>>();
			queue.add(new Utilities.Pair<Point, Double>(m_Points.elementAt(p_start),0.0));
			
			while(!queue.isEmpty())
			{
				// Java.... why are you not c++
				Utilities.Pair<Point, Double> itr = queue.iterator().next();
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
						queue.remove(new Utilities.Pair<Point, Double>(target, minDist[temp.m_pointIndex2]));
						
						minDist[temp.m_pointIndex2] = totalDist;
						queue.add(new Utilities.Pair<Point, Double>(target, totalDist));
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
			if(retVal < 0 || Double.isInfinite(retVal))
				System.out.println("HOUSTON WE HAVE A PROBLEM");
			return retVal;
		}
		//==================INTERNAL STRUCTS=======================================
		private class Point implements Cloneable, Serializable
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
				m_errorPercentage = -Double.MAX_VALUE;
				m_edges = new Vector<Edge>();
			}
			Point Clone()
			{
				Point retPoint = new Point();
				retPoint.m_instance = (Instance) m_instance.copy();
				retPoint.m_labeled = m_labeled;
				retPoint.m_covarianceIndex = m_covarianceIndex;
				retPoint.m_errorPercentage = -Double.MAX_VALUE; //This should be calculated by each graph and should therefore not be copiede
				//for(int i = 0; i < m_edges.size(); i++)
				//	retPoint.m_edges.add(m_edges.elementAt(i).Clone());
				retPoint.m_edges = new Vector<Edge>();
				return retPoint;
			}
			public boolean m_labeled;
			public int m_covarianceIndex;
			public Instance m_instance;
			public double m_errorPercentage;
			public Vector<Edge> m_edges;
		}
		private class Edge implements Cloneable, Serializable
		{
			Edge()
			{
				
			}
			Edge Clone()
			{
				Edge retEdge = new Edge();
				retEdge.m_pointIndex1 = m_pointIndex1;
				retEdge.m_pointIndex2 = m_pointIndex2;
				retEdge.m_weight = m_weight;
				return retEdge;
			}
			public int m_pointIndex1, m_pointIndex2;
			//The weight will be the mahalanobis distance between points
			public double m_weight;
		}
		
	}
}