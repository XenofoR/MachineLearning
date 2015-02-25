import java.awt.Color;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.Window;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.FastScatterPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.AbstractRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.PaintUtilities;
import org.jfree.util.ShapeUtilities;
import org.jfree.data.xy.XYSeriesCollection;

import weka.core.Instances;


public class Plotter
{
	//FastScatterPlot m_scatter;
	JFreeChart m_chart;
	 //float m_data[][];
	XYSeriesCollection m_data;
	ApplicationFrame m_frame;
	int m_counter = 0;
	boolean m_plot;
	Plotter()
	{
		
	}
	public void Init(String p_name)
	{
		
		m_frame = new ApplicationFrame(p_name);
		m_data = new XYSeriesCollection();
	}
	public void SetPlot(boolean p_plot)
	{
		m_plot = p_plot;
	}
	public void Set2dPlotValues(double[] x, double[] y) 
	{
		if(m_plot == false)
			return;
		XYSeries series = new XYSeries("Cluster" + m_counter);
		
	//	double temp[][] = {x,y};
		//int xl = x.length, yl = y.length;
		for(int i = 0; i < x.length; i++)
		{
			//for(int j = 0; j < y.length; j++)
			{
				series.add(x[i], y[i]);
			}
		}
		m_data.addSeries(series);	
		m_counter++;
	}
	public void Set2dPlotValues(Instances p_Unlabeled, Instances p_labeled) 
	{
		if(m_plot == false)
			return;
		double x[] = new double[p_Unlabeled.numInstances()];
		double y[] = new double[p_Unlabeled.numInstances()];
		XYSeries series = new XYSeries("unlabeled" + m_counter);
		for(int i = 0; i < p_Unlabeled.numInstances(); i++)
		{
			x[i] = p_Unlabeled.instance(i).toDoubleArray()[0];
			y[i] = p_Unlabeled.instance(i).toDoubleArray()[1];
		}
		double xL[] = new double[p_labeled.numInstances()];		
		double yL[] = new double[p_labeled.numInstances()];	
		XYSeries lSeries = new XYSeries("Labels" + m_counter);
		for(int i = 0; i < p_labeled.numInstances(); i++)
		{
			xL[i] = p_labeled.instance(i).toDoubleArray()[0];
			yL[i] = p_labeled.instance(i).toDoubleArray()[1];
		}
	//	double temp[][] = {x,y};
		//int xl = x.length, yl = y.length;
		for(int i = 0; i < p_Unlabeled.numInstances(); i++)	
			series.add(x[i], y[i]);
		
		for(int i = 0; i < p_labeled.numInstances(); i++)	
			lSeries.add(xL[i], yL[i]);
		
		m_data.addSeries(series);
		m_data.addSeries(lSeries);
		m_counter++;
	}
	//http://www.java2s.com/Code/Java/Chart/JFreeChartFastScatterPlotDemo.htm
	public void Display2dPlot() 
	{
		if(m_plot == false)
			return;
		//Something is missing....
		//m_scatter = new FastScatterPlot(m_data, m_xAxis, m_yAxis);
		m_chart = ChartFactory.createScatterPlot("Clusters", "X", "Y", m_data);
		// HEJ DU KAN FIXA SHAPES OM DU VILL
		Shape shape = null;
		XYPlot tempPlot = (XYPlot) m_chart.getPlot();
		XYItemRenderer renderer = tempPlot.getRenderer();
		for(int i = 0; i < m_data.getSeriesCount(); i++)
		{
			
			if(i % 2 == 0) //Unlabeled
			{
				shape = ShapeUtilities.createDiagonalCross(2, 2);
			}
			else //Labeled
			{
				
				shape = ShapeUtilities.createUpTriangle(2);
				Paint p = ((AbstractRenderer)renderer).lookupSeriesPaint(i-1);
				renderer.setSeriesPaint(i, p );
			}
			
			
			renderer.setSeriesShape(i, shape);
			
		}
		
		ChartPanel panel = new ChartPanel(m_chart,true);
		panel.setPreferredSize(new java.awt.Dimension(500, 270));
		panel.setMinimumDrawHeight(10);
        panel.setMaximumDrawHeight(2000);
        panel.setMinimumDrawWidth(20);
        panel.setMaximumDrawWidth(2000);
        m_frame.setContentPane(panel);
        m_frame.pack();

        RefineryUtilities.centerFrameOnScreen(m_frame);
        m_frame.setVisible(true);
	}
}