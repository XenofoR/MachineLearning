import weka.core.InstanceComparator;
import weka.core.Instances;


public class Philadelphiaost
{
	Instances m_instances;
	public Philadelphiaost() {
		// TODO Auto-generated constructor stub
	}
	void Init(Instances p_unlabeled)
	{
		m_instances = new Instances(p_unlabeled);// might need to change if new doesn't do what i want
	}
	Instances ConsultOracle(Instances p_instances)
	{
		Instances retInsts = new Instances(m_instances, 0); // might need to change if new doesn't do what i want
		InstanceComparator comp = new InstanceComparator();
		int breakCounter = 0;
		for(int i =0; i < p_instances.size(); i++)
		{
			for(int j = 0; j < m_instances.size(); j++)
				if(comp.compare(p_instances.instance(i), m_instances.instance(j)) == 0)
				{
					retInsts.add(m_instances.instance(j));
					breakCounter++;
					/*PRISON*/break;
				}
			if(breakCounter == p_instances.size())
				break;
		}
		retInsts.setClassIndex(retInsts.numAttributes() -1);
		return retInsts;
	}
}