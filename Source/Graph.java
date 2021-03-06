
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
	Long m_totalLeafTime = 0L;
	Long m_transductionTime  = 0L;
	int m_numLeafs = 0;
	Timer m_timer;
	Vector<InnerGraph> m_graphs;
	Vector<double[][]> m_covarianeMatrices;
	Vector<OurUtil.Pair<Integer, Integer>> m_idToIndexMap;
	boolean m_forceRootMerge = false;
	Graph()
	{
		
	}
	void Init()
	{
		m_timer = new Timer();
		m_covarianeMatrices = new Vector<double[][]>();		
		m_graphs = new Vector<InnerGraph>();
		m_idToIndexMap = new Vector<OurUtil.Pair<Integer, Integer>>();
		if(OurUtil.g_useMahalanobis == false)
		{
			m_graphs.add(new InnerGraph(-1, 0, -1, -1));
			m_graphs.elementAt(0).Init();
		}
	}
	void Cleanup()
	{
		for(int i = 0; i < m_graphs.size(); i++)
		{
			m_graphs.elementAt(i).Cleanup();
		}
		m_timer = null;
	}
	public Long[] GetTime()
	{
		Long[] retVal = new Long[3];
		retVal[0] = m_totalLeafTime / m_numLeafs;
		retVal[1] = m_totalLeafTime;
		retVal[2] = m_transductionTime;
		return retVal;	
	}
	public void GetInstances(Instances p_outInstances)
	{
		for(int i = 0; i < m_graphs.size(); i++)
		{
			if((m_graphs.elementAt(i).HasBeenMerged() == false) && m_graphs.elementAt(i).HasLabeled())
				m_graphs.elementAt(i).GetInstances(p_outInstances);
		}
	}
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
						double NaNcheck = m_graphs.elementAt(i).m_Points.elementAt(j).m_errorPercentage;
						if(Double.isFinite(NaNcheck))
							retVal += m_graphs.elementAt(i).m_Points.elementAt(j).m_errorPercentage;
						total++;
					}
				}
		}
		return retVal/total;
	}
	public void AddLeaf(Instances p_labeled, Instances p_unlabeled, double[][] p_covariance, int p_parentId, int p_id)
	{
		int id = m_timer.StartTimer(); 
		Debugger.DebugPrint("Added leaf node: " + p_id + " With parent: " + p_parentId, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
		if(OurUtil.g_useMahalanobis)
		{
			InnerGraph graph = new InnerGraph(p_parentId, p_id, -1 , -1);
			graph.Init();
			graph.AddCluster(p_labeled, p_unlabeled, p_covariance);
			OurUtil.Pair<Integer, Integer> temp = new OurUtil.Pair<Integer, Integer>(p_id, m_graphs.size());
			m_idToIndexMap.add(temp);
			m_graphs.add(graph);
		}
		else
		{
			m_graphs.elementAt(0).AddCluster(p_labeled, p_unlabeled, p_covariance);
		}
		m_totalLeafTime += m_timer.GetRawTime(id);
		m_timer.StopTimer(id);
		m_numLeafs += 1;
	}
	public void AddParent(int p_id, int p_parentId, int p_childId1, int p_childId2)
	{
		if(OurUtil.g_useMahalanobis == false)
			return;
		Debugger.DebugPrint("Added split node: " + p_id + " With Parent: " + p_parentId + " And children: " + p_childId1 + " " + p_childId2, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
		InnerGraph graph = new InnerGraph(p_parentId, p_id, p_childId1 , p_childId2);
		graph.Init();
		OurUtil.Pair<Integer, Integer> temp = new OurUtil.Pair<Integer, Integer>(p_id, m_graphs.size());
		m_idToIndexMap.add(temp);
		m_graphs.add(graph);
	}
	public Instance CalculateHighestUncertaintyAndPropagateLabels(double[] p_outVal) throws Exception
	{
		int tId = m_timer.StartTimer();
		Instance retInst = null;
		double val = 0;
		if(OurUtil.g_useMahalanobis)
		{
			if(m_forceRootMerge)
				return UncertaintyCompleteGraph(p_outVal);
			
			for(int i = 0; i < m_graphs.size(); i++)
			{
				if(m_graphs.elementAt(i).GetChildren()[0] != -1) // Non-leaf
					continue;
				if(m_graphs.elementAt(i).HasBeenMerged())
					continue;
				Instance temp = null;
				
				if(m_graphs.elementAt(i).HasLabeled())
					temp = m_graphs.elementAt(i).CalculateHighestUncertaintyAndPropagateLabels(p_outVal);
				else
				{
					Debugger.DebugPrint("No labeled, going into merge mode", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
					int parent = m_graphs.elementAt(i).GetParentId();
					int index = MergeChildren(parent, parent);
					index = FindIndexFromId(index);
					//parent = MergeChildren(index);
					temp = m_graphs.elementAt(index).CalculateHighestUncertaintyAndPropagateLabels(p_outVal);
				}
				if(val < p_outVal[0])
				{
					retInst = temp;
					val = p_outVal[0];
				}
			
				Debugger.DebugPrint("Checked graph: " + i + "of: " + m_graphs.size(), Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			}
			System.gc();
			p_outVal[0] = val;
			
		}
		else
			retInst = m_graphs.elementAt(0).CalculateHighestUncertaintyAndPropagateLabels(p_outVal);
		m_transductionTime += m_timer.GetRawTime(tId);
		m_timer.StopTimer(tId);
		return retInst;
	}
	
	public void ForceRootMerge(boolean p_force)
	{
		m_forceRootMerge = p_force;
	}
	private int MergeChildren(int p_id, int p_origin) throws Exception
	{
		if(p_id == -1)
			throw new Exception("MergeException:NoLabeledData");
		int retVal = p_id;
		int index = FindIndexFromId(p_id);
		if(!m_graphs.elementAt(index).m_Points.isEmpty())
			return p_id; //Is leaf or is already merged
		int[] children = m_graphs.elementAt(index).GetChildren();
		
		for(int i = 0; i < children.length; i++)
			MergeChildren(children[i], p_id);
		int childIndex1 = FindIndexFromId(children[0]);
		int childIndex2 = FindIndexFromId(children[1]);
		m_graphs.elementAt(index).MergeClusters(m_graphs.elementAt(childIndex1), m_graphs.elementAt(childIndex2));

		m_graphs.elementAt(childIndex1).SetHasBeenMerged(true);
		m_graphs.elementAt(childIndex2).SetHasBeenMerged(true);
		if(!m_graphs.elementAt(index).HasLabeled() && p_origin != m_graphs.elementAt(index).GetParentId())
			retVal = MergeChildren(m_graphs.elementAt(index).GetParentId(), p_id);
		return retVal;
	}
	private Instance UncertaintyCompleteGraph(double[] p_outVal) throws Exception
	{
		Instance retVal = null;
		int index = FindIndexFromId(0);
		if(m_graphs.elementAt(index).m_Points.isEmpty())
			MergeChildren(0, 0);
		Debugger.DebugPrint("Started Dikjstras on root graph, this is going to take a while", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
		retVal = m_graphs.elementAt(index).CalculateHighestUncertaintyAndPropagateLabels(p_outVal);
		
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
		public void Cleanup() 
		{
			for(int i =0; i < m_Points.size(); i++)
				m_Points.elementAt(i).Cleanup();
			
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
		void GetInstances(Instances p_outInstances)
		{
			for(int i = 0; i < m_Points.size(); i++)
				p_outInstances.add(m_Points.elementAt(i).m_instance);
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
			
			Matrix matrix = Matrix.constructWithCopy(p_covariance);
			SingularValueDecomposition SVD = new SingularValueDecomposition(matrix);
			Matrix S,V,U;
			S = SVD.getS();
			V = SVD.getV();
			U = SVD.getU();
			//calculate tolerance
			double tolerance = OurUtil.g_machineEpsilion * Math.max(S.getColumnDimension(), S.getRowDimension()) * S.norm2();
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
					
	    	double[][] inverseMat = matrix.getArray();
			
			int index = m_covarianeMatrices.size();
			m_covarianeMatrices.add(inverseMat);
			Instances tempL = new Instances(p_labeled);
			Instances tempU = new Instances(p_unlabeled);
			for(int i = 0; i < tempL.size(); i++)
			{
				//Remember that this is a labeled instance
				m_labeledIndices.add(m_Points.size());
				Point point = new Point(tempL.instance(i), true, index);
				//Build edges
				ConstructEdges(point, 0 , m_Points.size());
				m_Points.add(point);	
			}
			for(int i = 0; i < tempU.size(); i++)
			{
				Point point = new Point(tempU.instance(i), false, index);
				//Build edges
				ConstructEdges(point, 0 , m_Points.size());
				m_Points.add(point);	
			}
			
		}
		
		public void MergeClusters(InnerGraph p_graph1, InnerGraph p_graph2) throws Exception
		{	
			
			Debugger.DebugPrint("Merging clusters " + p_graph1.GetId() + " " + p_graph2.GetId() +" to cluster: " +m_id, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			int firstGraphlast = 0;
			for(int i = 0; i < p_graph1.m_Points.size(); i++)
			{
				Point temp = p_graph1.m_Points.elementAt(i);
				if(temp.m_labeled)
					m_labeledIndices.add(m_Points.size());
				m_Points.add(temp);
				firstGraphlast++;
			}
		
			for(int i = 0; i < p_graph2.m_Points.size(); i++)
			{
				Point temp = p_graph2.m_Points.elementAt(i);
				if(temp.m_labeled)
					m_labeledIndices.add(m_Points.size());
				for(int j = 0; j < temp.m_edges.size(); j++)
				{
					temp.m_edges.elementAt(j).m_pointIndex1 += firstGraphlast;
					temp.m_edges.elementAt(j).m_pointIndex2 += firstGraphlast;
				}
				//We only need to construct the edges between clusters once,
				//therefore we do it after the first cluster has been completed to save time.
				ConstructEdges(temp,0, firstGraphlast);
				m_Points.add(temp);
			}
			//System.gc(); 
			Debugger.DebugPrint("Merge Complete", Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
			
		}
		
		private void ConstructEdges(Point p_currPoint, int p_start, int p_end)
		{
			double[] currPointArray, pointArray;
			//Index that the current point will have after it's added
			int myIndex = m_Points.size();
			
			
			//If labeled ignore last attribute since it is a label
			currPointArray  = new double [p_currPoint.m_instance.numAttributes() -1];
			for(int i = 0; i < currPointArray.length; i++)
				currPointArray[i] = p_currPoint.m_instance.toDoubleArray()[i];
			
			//since it is a complete graph we will need edges to all other points
			int end = (p_end <= m_Points.size()) ? p_end : m_Points.size();
			for(int i = p_start; i < end; i++)
			{
				//If labeled ignore last attribute since it is a label
				pointArray =  new double [m_Points.elementAt(i).m_instance.numAttributes() -1];
				//Change from instance to array
				for(int j = 0; j < pointArray.length; j++)
					pointArray[j] = m_Points.elementAt(i).m_instance.toDoubleArray()[j];
				
				
				Edge edge = new Edge();
				Edge edge2 = new Edge();
				edge.m_pointIndex1 = myIndex;
				edge.m_pointIndex2 = i;
				edge2.m_pointIndex1 = i;
				edge2.m_pointIndex2 = myIndex;
				
				if(OurUtil.g_useMahalanobis)
				{
					//calculate the mean of the mahalanobis distance using both covariance matrices. P.200 in Criminisi 2011
					double mahalanobis = CalculateMahalanobisDistance(currPointArray, pointArray, p_currPoint.m_covarianceIndex);
					mahalanobis +=  CalculateMahalanobisDistance(currPointArray, pointArray, m_Points.elementAt(i).m_covarianceIndex);
					mahalanobis /= 2;
					
					if(mahalanobis > 10000)
						Debugger.DebugPrint("Really high distance for edge detected: " + mahalanobis, Debugger.g_debug_MEDIUM, Debugger.DebugType.CONSOLE);
					
					edge.m_weight = mahalanobis;
					edge2.m_weight =  mahalanobis;
				}
				else //Eucledian distance
				{
					double eucledian = CalculateEucledianDistance(currPointArray, pointArray);
					edge.m_weight = 
					edge2.m_weight =  eucledian;
				}
				p_currPoint.m_edges.add(edge);
				m_Points.elementAt(i).m_edges.add(edge2);
			}
		}
		
		
		private double CalculateEucledianDistance(double[] p_point1, double[] p_point2) 
		{
			double retVal = 0;
			double[] distVec = new double[p_point1.length];
			for(int i = 0; i < distVec.length; i++)
				retVal += Math.pow(p_point1[i] - p_point2[i], 2);
			return Math.sqrt(retVal);
		}
		
		public Instance CalculateHighestUncertaintyAndPropagateLabels(double[] p_outVal)
		{
			Debugger.DebugPrint("Starting path finding with " + (m_Points.size() - m_labeledIndices.size()) +" unlabeled and " + m_labeledIndices.size() + " lableled instance(s)", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			if(m_labeledIndices.size() == 0)
				return null;
			Instance retVal = null;;
			double out = 0;
			if(OurUtil.g_useMahalanobis)
			{
				double[] totalDist = new double[m_Points.size()];
				double[][] pointToLabeled = new double[m_Points.size()][m_labeledIndices.size()];
				for(int i = 0; i < m_labeledIndices.size(); i++)
				{
					int index = m_labeledIndices.elementAt(i);
					//Calculate from label to each point
					double[] distances = Dijkstra(index);
					for(int j = 0; j < distances.length; j++)
					{
						if(m_labeledIndices.contains(j))
						{
							pointToLabeled[j][i] =  -1;
							totalDist[j] = -1;
						}
						else
						{
							pointToLabeled[j][i] =  distances[j];
							totalDist[j] += distances[j];
						}
					}
				}
				double[] shortest = new double[pointToLabeled.length];
				for(int i = 0; i < pointToLabeled.length; i++)
				{
					if(m_labeledIndices.contains(i))
						continue;
					double localShortest = Double.MAX_VALUE;					
					for(int j = 0; j < pointToLabeled[i].length; j++)
						localShortest = (pointToLabeled[i][j] < localShortest) ? pointToLabeled[i][j] : localShortest;						
					shortest[i] = localShortest;	
				}
				
				
				for(int i = 0; i < pointToLabeled.length; i++)
				{
					if(m_labeledIndices.contains(i))
						continue;
					//double label = 0;
					if(shortest[i] > out)
					{
						retVal = m_Points.elementAt(i).m_instance;
						out = shortest[i];
						//TODO Working with java is like working with a lava flow, with your hand.					
					}
					double label = 0;
					if(OurUtil.g_useWeightedTransduction)
					{
						for(int j = 0; j < m_labeledIndices.size(); j++)
						{
						
							double percentage =  Math.abs((pointToLabeled[i][j] / totalDist[i]) - 1)/(pointToLabeled[i].length-1);	
						// if 0 we only have 1 labeled and then we will simply apply it directly
							percentage = (percentage == 0  || (Double.isNaN(percentage)) || Double.isInfinite(percentage)) ? 1 : percentage; 
							label +=  percentage * m_Points.elementAt(m_labeledIndices.elementAt(j)).m_instance.classValue();
						}
					
							int labelIndex = m_Points.elementAt(i).m_instance.numAttributes()-1;
							double error = Math.abs( label - m_Points.elementAt(i).m_instance.value(labelIndex))/ (m_Points.elementAt(i).m_instance.value(labelIndex) + label);
							m_Points.elementAt(i).m_errorPercentage = (m_Points.elementAt(i).m_errorPercentage ==Double.MAX_VALUE ) ? error : m_Points.elementAt(i).m_errorPercentage;
							m_Points.elementAt(i).m_instance.setValue(m_Points.elementAt(i).m_instance.numAttributes()-1, label); 
					}
					else
					{
						int shortestLabelIndex = Integer.MAX_VALUE;
						double dist = Double.MAX_VALUE;
						for(int j = 0; j < pointToLabeled[i].length; j++)
						{
							if(pointToLabeled[i][j] < dist)
							{
								dist = pointToLabeled[i][j];
								shortestLabelIndex = j;
							}
						}
						label = m_Points.elementAt(m_labeledIndices.elementAt(shortestLabelIndex)).m_instance.classValue();
						int labelIndex = m_Points.elementAt(i).m_instance.numAttributes()-1;
						double error = Math.abs( label - m_Points.elementAt(i).m_instance.value(labelIndex))/ (m_Points.elementAt(i).m_instance.value(labelIndex) + label);
						m_Points.elementAt(i).m_errorPercentage = (m_Points.elementAt(i).m_errorPercentage ==Double.MAX_VALUE ) ? error : m_Points.elementAt(i).m_errorPercentage;
						m_Points.elementAt(i).m_instance.setValue(m_Points.elementAt(i).m_instance.numAttributes()-1, label); 	
					}
				}
			}
			else //Eucledian stuff
			{
				double[] temp1 = new double[m_Points.elementAt(0).m_instance.numAttributes()-1];
				double[] temp2 = new double[m_Points.elementAt(0).m_instance.numAttributes()-1];
				//TODO: ADD unweighted label
				
				double[] shortest = new double[m_Points.size()];
				for(int i = 0; i < m_Points.size(); i++)
				{
					if(m_Points.elementAt(i).m_labeled)
					{
						shortest[i] = -Double.MAX_VALUE;
						continue;
					}
					double[] distance = new double[m_labeledIndices.size()];
					double totalDist = 0;
					double localshort = Double.MAX_VALUE;
					for(int j = 0; j < temp1.length; j++)
						temp1[j] = m_Points.elementAt(i).m_instance.toDoubleArray()[j];
					for(int j = 0 ; j < m_labeledIndices.size(); j++)
					{
						for(int k = 0; k < temp2.length; k++)
							temp2[k] = m_Points.elementAt(m_labeledIndices.elementAt(j)).m_instance.toDoubleArray()[k];
						double dist = CalculateEucledianDistance(temp1, temp2);
						distance[j] = dist;
						totalDist += dist;
						if(dist < localshort)
							localshort = dist;				
					}
					shortest[i] = localshort;
					double label = 0;
					if(OurUtil.g_useWeightedTransduction)
					{
						for(int j = 0; j < m_labeledIndices.size(); j++)
						{
							double percentage = Math.abs((distance[j] / totalDist) -1 ) /(distance.length -1);
							percentage = (percentage == 0  || (Double.isNaN(percentage)) || Double.isInfinite(percentage)) ? 1 : percentage;
							label += percentage* m_Points.elementAt(m_labeledIndices.elementAt(j)).m_instance.classValue();
						}
						int labelIndex = m_Points.elementAt(i).m_instance.numAttributes()-1;
						double error = Math.abs((m_Points.elementAt(i).m_instance.value(labelIndex) - label)/ (m_Points.elementAt(i).m_instance.value(labelIndex)));
						m_Points.elementAt(i).m_errorPercentage = (m_Points.elementAt(i).m_errorPercentage ==Double.MAX_VALUE ) ? error : m_Points.elementAt(i).m_errorPercentage;
						m_Points.elementAt(i).m_instance.setValue(m_Points.elementAt(i).m_instance.numAttributes()-1, label);
					}
					else
					{
						double dist = Double.MAX_VALUE;
						int shortestLabelIndex = -3123;
						for(int j = 0; j < shortest.length; j++)
						{
							if(shortest[j] < dist)
							{
								dist = shortest[j];
								shortestLabelIndex = j;
							}
						}
						label = m_Points.elementAt(m_labeledIndices.elementAt(shortestLabelIndex)).m_instance.classValue();
						int labelIndex = m_Points.elementAt(i).m_instance.numAttributes()-1;
						double error = Math.abs( label - m_Points.elementAt(i).m_instance.value(labelIndex))/ (m_Points.elementAt(i).m_instance.value(labelIndex) + label);
						m_Points.elementAt(i).m_errorPercentage = (m_Points.elementAt(i).m_errorPercentage ==Double.MAX_VALUE ) ? error : m_Points.elementAt(i).m_errorPercentage;
						m_Points.elementAt(i).m_instance.setValue(m_Points.elementAt(i).m_instance.numAttributes()-1, label); 
					}
					
				}
				for(int i = 0; i < shortest.length; i++)
					if(shortest[i] > out)
					{
						out = shortest[i];
						retVal = m_Points.elementAt(i).m_instance;
					}
			} 
			p_outVal[0] = out;
			Debugger.DebugPrint("Path Finding Complete", Debugger.g_debug_LOW, Debugger.DebugType.CONSOLE);
			return retVal;
		}
		private double[] Dijkstra(int p_start)
		{
			//Shortest distance to each point from origin
			double[] minDist = new double[m_Points.size()];
			Arrays.fill(minDist, Double.MAX_VALUE);
			//minDist[p_start] = 0; //Origin dist
			Set<OurUtil.Pair<Point, Double>> queue = new LinkedHashSet<OurUtil.Pair<Point, Double>>();
			queue.add(new OurUtil.Pair<Point, Double>(m_Points.elementAt(p_start),0.0));
			
			while(!queue.isEmpty())
			{
				// Java.... why are you not c++
				OurUtil.Pair<Point, Double> itr = queue.iterator().next();
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
						queue.remove(new OurUtil.Pair<Point, Double>(target, minDist[temp.m_pointIndex2]));
						
						minDist[temp.m_pointIndex2] = totalDist;
						queue.add(new OurUtil.Pair<Point, Double>(target, totalDist));
					}
				}
			}

			return minDist;
		}
		
		private double CalculateMahalanobisDistance(double[] p_first, double[] p_second, int p_covMatIndex)
		{
			double retVal = 0;
			
			//retval = D^T * M^-1 * D
			double[] distanceVec = new double[p_first.length];
			OurUtil.Subtract(p_first, p_second, distanceVec);
			
	    	double[] tempVec = new double[distanceVec.length];
	    	double[][] inverseMat = m_covarianeMatrices.elementAt(p_covMatIndex);
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
			if(retVal < 0 || Double.isInfinite(retVal) || Double.isNaN(retVal))
			{
				System.out.println("Edge distance is infinite, negative or NaN, most likely caused by zero covariance matrix");
				retVal = Double.MAX_VALUE;
			}
			
			distanceVec = null;
			tempVec = null;
			inverseMat = null;
			
			return retVal;
		}
		//==================INTERNAL STRUCTS=======================================
		private class Point implements Cloneable, Serializable
		{
			Point()
			{
				m_edges = new Vector<Edge>();
			}
			public void Cleanup() 
			{
				if(m_edges != null){
					m_edges.clear();
					m_edges = null;
				}
			}
			Point(Instance p_instance, boolean p_labeled, int p_covarianceIndex)
			{
				m_instance = p_instance;
				m_labeled = p_labeled;
				m_covarianceIndex = p_covarianceIndex;
				m_errorPercentage = Double.MAX_VALUE;
				m_edges = new Vector<Edge>();
			}
			Point Clone()
			{
				Point retPoint = new Point();
				retPoint.m_instance = (Instance) m_instance.copy();
				retPoint.m_labeled = m_labeled;
				retPoint.m_covarianceIndex = m_covarianceIndex;
				retPoint.m_errorPercentage = Double.MAX_VALUE; //This should be calculated by each graph and should therefore not be copiede
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