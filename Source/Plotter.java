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


public class Plotter
{
	//FastScatterPlot m_scatter;
	JFreeChart m_chart;
	NumberAxis m_xAxis, m_yAxis;
	//float m_data[][];
	XYSeriesCollection m_data;
	ApplicationFrame m_frame;
	Plotter()
	{
		
	}
	public void Init()
	{
		m_xAxis = new NumberAxis("X");
		m_xAxis.setAutoRangeIncludesZero(false);
		m_yAxis = new NumberAxis("Y");
		m_yAxis.setAutoRangeIncludesZero(false);
		m_frame = new ApplicationFrame("Test");
		m_data = new XYSeriesCollection();
	}
	public void Set2dPlotValues(double[] x, double[] y) 
	{
		
		float minX = Float.MAX_VALUE;
		float maxX = -Float.MAX_VALUE;
		float minY = Float.MAX_VALUE;
		float maxY = -Float.MAX_VALUE;
		//Find values to use as plot range
		for(int i = 0; i < x.length; i++)
		{	
			minX = (float) (x[i] < minX ? x[i] : minX);
			maxX = (float) (x[i] > maxX ? x[i] : maxX);
			minY = (float) (y[i] < minY ? x[i] : minY);
			maxY = (float) (y[i] > maxY ? x[i] : maxY);
		}
		XYSeries series = new XYSeries("Stuff");
		m_xAxis.setRange(minX, maxX);
		m_yAxis.setRange(minY, maxY);
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
		
	}
	//http://www.java2s.com/Code/Java/Chart/JFreeChartFastScatterPlotDemo.htm
	public void Display2dPlot() 
	{
		//Something is missing....
		//m_scatter = new FastScatterPlot(m_data, m_xAxis, m_yAxis);
		m_chart = ChartFactory.createScatterPlot("test", "The fuck", "Useless shit", m_data);
		
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