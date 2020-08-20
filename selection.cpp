#include "selection.h"

namespace selection
{
	tournament_result k_tournament(population& pop, int k)
	{
		tournament_result res;
		for(int i = 0; i < k; ++i)
		{
			auto * solution = &pop.random_solution();
			while(solution == res.mom || solution == res.dad)
			{
				solution = &pop.random_solution();
			}
			
			if(nullptr == res.mom || res.mom->penalty > solution->penalty)
			{
				
				if(!res.dead && res.dad)
				{
					res.dead = res.dad;
				}
				
				res.dad = res.mom;
				res.mom = solution;
				
			} else if(nullptr == res.dad || res.dad->penalty > solution->penalty)
			{
				if(!res.dead && res.dad)
				{
					res.dead = res.dad;
				}
				
				res.dad = solution;
				
			} else if(nullptr == res.dead || res.dead->penalty < solution->penalty)
			{
				res.dead = solution;
			}
		}
		
		return res;
	}
};
