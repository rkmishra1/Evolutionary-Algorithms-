# g-NSGA-II

**Tags**: <2009> <multi> <real/integer/label/binary/permutation>

## Description
g-dominance based NSGA-II

## Reference
J. Molina, L. V. Santana, A .G. Hernandez-Diaz, C. A. Coello Coello, and multiobjective metaheuristics. European Journal of Operational Research, 2009, 197(2): 685-692.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N,Point)
% The environmental selection of g-NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    PopObj = Evaluate(Population.objs,Point);
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    
    %% Select the solutions in the last front by their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `Evaluate.m`
```matlab
function PopObj = Evaluate(PopObj,Point)
% g-dominance based function evaluation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Point = repmat(Point,size(PopObj,1),1);
    Flag  = all(PopObj<=Point,2) | all(PopObj>=Point,2);
    Flag  = repmat(Flag,1,size(PopObj,2));
    PopObj(~Flag) = PopObj(~Flag) + 1e10;
end
```

### `gNSGAII.m`
```matlab
classdef gNSGAII < ALGORITHM
% <2009> <multi> <real/integer/label/binary/permutation>
% g-dominance based NSGA-II
% Point --- --- Preferred point

%------------------------------- Reference --------------------------------
% J. Molina, L. V. Santana, A .G. Hernandez-Diaz, C. A. Coello Coello, and
% R. Caballero. g-dominance: Reference point based dominance for
% multiobjective metaheuristics. European Journal of Operational Research,
% 2009, 197(2): 685-692.
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
            Point = Algorithm.ParameterSet(zeros(1,Problem.M)+0.5);

            %% Generate random population
            Population = Problem.Initialization();
            FrontNo    = NDSort(Evaluate(Population.objs,Point),inf);
            CrowdDis   = CrowdingDistance(Population.objs,FrontNo);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N,Point);
            end
        end
    end
end
```
