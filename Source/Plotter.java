import java.awt.Shape;
import java.awt.Window;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.FastScatterPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.ShapeUtilities;
import org.jfree.data.xy.XYSeriesCollection;

import weka.core.Instances;


public class Plotter
{
	//FastScatterPlot m_scatter;
	static JFreeChart m_chart;
	 //float m_data[][];
	static XYSeriesCollection m_data;
	static ApplicationFrame m_frame;
	static int m_counter = 0;
	static boolean m_plot;
	Plotter()
	{
		
	}
	static public void Init()
	{
		
		m_frame = new ApplicationFrame("Test");
		m_data = new XYSeriesCollection();
	}
	static public void SetPlot(boolean p_plot)
	{
		m_plot = p_plot;
	}
	static public void Set2dPlotValues(double[] x, double[] y) 
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
	static public void Set2dPlotValues(Instances p_instances) 
	{
		if(m_plot == false)
			return;
		double x[] = new double[p_instances.numInstances()];
		double y[] = new double[p_instances.numInstances()];
		XYSeries series = new XYSeries("Cluster" + m_counter);
		for(int i = 0; i < p_instances.numInstances(); i++)
		{
			x[i] = p_instances.instance(i).toDoubleArray()[0];
			y[i] = p_instances.instance(i).toDoubleArray()[1];
		}
				
		
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
	//http://www.java2s.com/Code/Java/Chart/JFreeChartFastScatterPlotDemo.htm
	static public void Display2dPlot() 
	{
		if(m_plot == false)
			return;
		//Something is missing....
		//m_scatter = new FastScatterPlot(m_data, m_xAxis, m_yAxis);
		m_chart = ChartFactory.createScatterPlot("Clusters", "X", "Y", m_data);
		
		Shape shape = ShapeUtilities.createDiamond(3);
		XYPlot tempPlot = (XYPlot) m_chart.getPlot();
		XYItemRenderer renderer = tempPlot.getRenderer();
		renderer.setSeriesShape(0, shape);
		
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