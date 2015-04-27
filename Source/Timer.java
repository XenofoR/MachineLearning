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
	Long GetTime(int p_index)
	{
		return (System.currentTimeMillis()) - m_timers.elementAt(p_index);
	}
}