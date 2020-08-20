#ifndef SELECTION_H
#define SELECTION_H

#include "genetic_solution.h"
#include "population.h"

namespace selection
{
	struct tournament_result
	{
		union
		{
			struct
			{
				genetic_solution * mom{nullptr};
				genetic_solution * dad{nullptr};
				genetic_solution * dead{nullptr};
			};
			genetic_solution * selected_solutions[3];
		};
	};
	
	tournament_result k_tournament(population&, int);
};

#endif
