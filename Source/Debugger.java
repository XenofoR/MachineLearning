import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;



public  class Debugger
{
	public final static int g_debug_NONE = 0;
	public final static int g_debug_LOW = 1;
	public final static int g_debug_MEDIUM = 2;
	public final static int g_debug_HIGH = 3;
	static boolean g_plot = false;;
	private static int m_globalDebugLevel = g_debug_NONE;
	private static String m_debugFilePath;
	public enum DebugType
	{
		FILE,
		CONSOLE,
		BOTH,
		NONE
	}
	Debugger()
	{
		
	}
	/**
	 * @param p_debugLevel 
	 * @param p_debugFilePath
	 *  If no debugFilePath is specified debug to file will be disabled
	 */
	public static void Init(int p_debugLevel, String p_debugFilePath)
	{
		m_globalDebugLevel = p_debugLevel;
		m_debugFilePath = p_debugFilePath;
	}
	/**
	 * 
	 * @param p_debugData
	 * @param p_minDebugLevel
	 * @param p_debugType
	 * CONSOLE, FILE or BOTH. Be aware that writing to file is expensive and therefore performance may be low
	 * when using large amount of FILE/BOTH debugging.
	 * @return
	 * FALSE if unsuccesful. This can happen if Global debugLevel is to lower than the minimum debuglevel
	 *  required for this print or if attempting to debug to file and no filepath is specified at init
	 * TRUE otherwise
	 * @throws FileNotFoundException 
	 * @throws UnsupportedEncodingException 
	 */
	public static boolean DebugPrint(String p_debugData, int p_minDebugLevel, DebugType p_debugType)
	{
		
		if((p_debugType == DebugType.FILE && m_debugFilePath == null) ||  p_minDebugLevel == g_debug_NONE)
			return false;
		if(p_debugType == DebugType.CONSOLE || p_debugType == DebugType.BOTH)
		{
			if(p_minDebugLevel <= m_globalDebugLevel)
			{
				System.out.println(p_debugData);
				return true;
			}
		}
		if(p_debugType == DebugType.FILE || p_debugType == DebugType.BOTH)
		{
			if(p_minDebugLevel <= m_globalDebugLevel)
			{
				try{
				Writer w = new BufferedWriter(new FileWriter(m_debugFilePath, true));
				w.write(p_debugData);
				w.close();
				}
				catch(Exception E)
				{
					System.out.println("Exception found when attempting to write to file: " + E.toString());
					return false;
				}
				return true;
			}
		}
		return false;
	}
	
}