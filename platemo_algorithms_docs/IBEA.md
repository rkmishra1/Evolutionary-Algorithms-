# IBEA

**Tags**: <2004> <multi/many> <real/integer/label/binary/permutation>

## Description
Indicator-based evolutionary algorithm

## Reference
E. Zitzler and S. Kunzli. Indicator-based selection in multiobjective search. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2004, 832-842.

## Source Code

### `CalFitness.m`
```matlab
function [Fitness,I,C] = CalFitness(PopObj,kappa)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N      = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C       = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,kappa)
% The environmental selection of IBEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Next          = 1 : length(Population);
    [Fitness,I,C] = CalFitness(Population.objs,kappa);
    while length(Next) > N
        [~,x]   = min(Fitness(Next));
        Fitness = Fitness + exp(-I(Next(x),:)/C(Next(x))/kappa);
        Next(x) = [];
    end
    Population = Population(Next);
end
```

### `IBEA.m`
```matlab
classdef IBEA < ALGORITHM
% <2004> <multi/many> <real/integer/label/binary/permutation>
% Indicator-based evolutionary algorithm
% kappa --- 0.05 --- Fitness scaling factor

%------------------------------- Reference --------------------------------
% E. Zitzler and S. Kunzli. Indicator-based selection in multiobjective
% search. Proceedings of the International Conference on Parallel Problem
% Solving from Nature, 2004, 832-842.
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
            kappa = Algorithm.ParameterSet(0.05);

            %% Generate random population
            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,-CalFitness(Population.objs,kappa));
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,kappa);
            end
        end
    end
end
```
