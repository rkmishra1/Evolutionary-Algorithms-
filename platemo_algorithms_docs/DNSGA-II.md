# DNSGA-II

**Tags**: <2007> <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>

## Description
Dynamic NSGA-II

## Reference
K. Deb, U. Bhaskara Rao N., and S. Karthik. Dynamic multi-objective optimization and decision-making using modified NSGA-II: A case study on hydro-thermal power scheduling. Proceedings of the International Conference on Evolutionary Multi-Criterion Optimization, 2007, 803-817.

## Source Code

### `Changed.m`
```matlab
function changed = Changed(Problem,Population)
% Detect whether the problem changes

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    RePop1  = Population(randperm(end,ceil(end/10)));
    RePop2  = Problem.Evaluation(RePop1.decs);
    changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);
end
```

### `DNSGAII.m`
```matlab
classdef DNSGAII < ALGORITHM
% <2007> <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>
% Dynamic NSGA-II
% type ---   1 --- 1. Mutation based reinitialization 2. Random reinitialization
% zeta --- 0.2 --- Ratio of reinitialized solutions
 
%------------------------------- Reference --------------------------------
% K. Deb, U. Bhaskara Rao N., and S. Karthik. Dynamic multi-objective
% optimization and decision-making using modified NSGA-II: A case study on
% hydro-thermal power scheduling. Proceedings of the International
% Conference on Evolutionary Multi-Criterion Optimization, 2007, 803-817.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [type,zeta] = Algorithm.ParameterSet(1,0.2);
            % Reset the number of saved populations (only for dynamic optimization)
            Algorithm.save = sign(Algorithm.save)*inf;
            
            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            % Archive for storing all populations before each change
            AllPop = [];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Changed(Problem,Population)
                    % Save the population before the change
                    AllPop = [AllPop,Population];
                    % React to the change
                    [Population,FrontNo,CrowdDis] = Reinitialization(Problem,Population,type,zeta);
                end
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                if Problem.FE >= Problem.maxFE
                    % Return all populations
                    Population = [AllPop,Population];
                    [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    Population = Population(rank);
                end
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `Reinitialization.m`
```matlab
function [Population,FrontNo,CrowdDis] = Reinitialization(Problem,Population,type,zeta)
% Re-initialize solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = floor(length(Population)*zeta/2)*2;
    Selected = randperm(length(Population),N);
    if type == 1
        Population(Selected) = OperatorGA(Problem,Population(Selected),{0,0,inf,10});
    else
        Population(Selected) = Problem.Initialization(N);
    end
    unSelected = setdiff(1:length(Population),Selected);
    Population(unSelected) = Problem.Evaluation(Population(unSelected).decs);
    [~,FrontNo,CrowdDis]   = EnvironmentalSelection(Population,length(Population));
end
```
