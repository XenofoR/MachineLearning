import java.awt.Window;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.FastScatterPlot;
import org.jfree.ui.RefineryUtilities;



public class Plotter
{
	FastScatterPlot m_scatter;
	JFreeChart m_chart;
	NumberAxis m_xAxis, m_yAxis;
	float m_data[][];
	Plotter()
	{
		
	}
	public void Init()
	{
		m_xAxis = new NumberAxis("X");
		m_yAxis = new NumberAxis("Y");
	}
	public void Set2dPlotValues(double[] x, double[] y) 
	{
		m_data = new float[2][y.length];
		//...fucking java
		double temp[][] = {x,y};
		int xl = x.length, yl = y.length;
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < y.length; j++)
			{
				m_data[i][j] = (float) temp[i][j];
			}
		}
		
		
	}
	//http://www.java2s.com/Code/Java/Chart/JFreeChartFastScatterPlotDemo.htm
	public void Display2dPlot() 
	{
		//Something is missing....
		m_scatter = new FastScatterPlot(m_data, m_xAxis, m_yAxis);
		m_chart = new JFreeChart("Test", m_scatter);
		ChartPanel panel = new ChartPanel(m_chart,true);
		panel.setMinimumDrawHeight(10);
        panel.setMaximumDrawHeight(2000);
        panel.setMinimumDrawWidth(20);
        panel.setMaximumDrawWidth(2000);
		panel.setVisible(true);
	}
}