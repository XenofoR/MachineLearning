import java.util.Vector;



class Timer
{
	Vector<Long> m_timers;
	Vector<Integer> m_freeIndex;
	Timer() 
	{
		m_timers = new Vector<Long>();
		m_freeIndex = new Vector<Integer>();
	}
	int StartTimer()
	{
		if(m_freeIndex.isEmpty())
		{
			m_timers.add(System.currentTimeMillis());
			return m_timers.size()-1;
		}
		else
		{
			int freeIndex = m_freeIndex.firstElement();
			m_timers.set(freeIndex, System.currentTimeMillis());
			m_freeIndex.remove(0);
			return freeIndex;
		}
	}
	void StopTimer(int p_index)
	{
		m_freeIndex.add(p_index);
		m_timers.set(p_index, Long.MIN_VALUE);
	}
	Long GetRawTime(int p_index)
	{
		return (System.currentTimeMillis()) - m_timers.elementAt(p_index);
	}
	
	String GetFormatedTime(int p_index)
	{
		return ConvertRawToFormated((System.currentTimeMillis()) - m_timers.elementAt(p_index));
	}
	String ConvertRawToFormated(Long p_rawTime)
	{
		Long seconds = (p_rawTime / 1000) % 60;
		Long minutes = (p_rawTime / (1000 * 60)) % 60;
		Long hours=	(p_rawTime / (1000 * 60 * 60)) % 24;
		String retString = "" + hours + ":" + minutes + ":" + seconds + ":" + (p_rawTime - (seconds * 1000) -( minutes * 60000) - (hours *1000 * 60 * 60));
		return retString;
	}
}