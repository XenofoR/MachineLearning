import java.io.File;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffLoader;

public class Loader
{
	File m_file;
	Instances m_structure;
	ArffLoader m_arffLoader;
	CSVLoader m_csvLoader;
	public Loader()
	{
		m_arffLoader = new ArffLoader();
		m_csvLoader = new CSVLoader();
	}
	
	public void setFile(File p_file) throws Exception
	{
		m_file = p_file;
		
		String filename = p_file.getName();
		
		String[] parts = filename.split("\\.");
		
		switch(parts[1])
		{
			case("arff"):
				m_arffLoader.setFile(p_file);
				m_structure = m_arffLoader.getDataSet();
				break;
			case("csv"):
				m_csvLoader.setSource(p_file);
				m_structure = m_csvLoader.getDataSet();
				break;
			default:
				break;
		}
	}
	
	public Instances getStructure()
	{
		return m_structure;
	}
}