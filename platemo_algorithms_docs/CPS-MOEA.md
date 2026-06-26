# CPS-MOEA

**Tags**: <2015> <multi> <real/integer> <expensive>

## Description
Classification and Pareto domination based multi-objective evolutionary

## Reference
J. Zhang, A. Zhou, and G. Zhang. A classification and Pareto domination based multiobjective evolutionary algorithm. Proceedings of the IEEE Congress on Evolutionary Computation, 2015, 2883-2890.

## Source Code

### `CPSMOEA.m`
```matlab
classdef CPSMOEA < ALGORITHM
% <2015> <multi> <real/integer> <expensive>
% Classification and Pareto domination based multi-objective evolutionary
% algorithm
% M --- 3 --- Number of generated offsprings for each solution

%------------------------------- Reference --------------------------------
% J. Zhang, A. Zhou, and G. Zhang. A classification and Pareto domination
% based multiobjective evolutionary algorithm. Proceedings of the IEEE
% Congress on Evolutionary Computation, 2015, 2883-2890.
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
            M = Algorithm.ParameterSet(3);

            %% Generate random population
            Population   = Problem.Initialization();
            [Pgood,Pbad] = NDS(Population,floor(Problem.N/2));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                KNN(Pgood.decs,Pbad.decs);
                Offspring  = Operator(Problem,Population,M);
                Population = NDS([Population,Offspring],Problem.N);
                FrontNo    = NDSort(Offspring.objs,1);
                Pgood      = NDS([Pgood,Offspring(FrontNo==1)],floor(Problem.N/2));
                Pbad       = NDS([Pbad,Offspring(FrontNo~=1)],floor(Problem.N/2));
            end
        end
    end
end
```

### `KNN.m`
```matlab
function Labels = KNN(varargin)
% K-nearest neighbor classification

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

persistent model

    if nargin == 2
        %% Train
        model.data  = [varargin{1};varargin{2}];
        model.label = [true(1,size(varargin{1},1)),false(1,size(varargin{2},1))];
    else
        %% Predict
        Distance = pdist2(varargin{1},model.data);
        [~,rank] = sort(Distance,2);
        Labels   = sum(model.label(rank(:,1:5))==1,2) > 2;
    end
end
```

### `NDS.m`
```matlab
function [Pgood,Pbad] = NDS(Population,K)
% Sort the population based on non-dominated sorting and crowding distance

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    FrontNo  = NDSort(Population.objs,inf);
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    [~,rank] = sortrows([FrontNo;-CrowdDis]');
    Pgood    = Population(rank(1:K));
    Pbad     = Population(rank(end-K+1:end));
end
```

### `Operator.m`
```matlab
function Offspring  = Operator(Problem,Population,M)
% Generate offsprings by DE and KNN based surrogate model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = length(Population);
    
    %% Generate all candidate offsprings
    CandidateDec = OperatorDE(Problem,Population(repmat(1:N,1,M)).decs,Population(randi(N,1,N*M)).decs,Population(randi(N,1,N*M)).decs);
    
    %% Classification based preselection (CPS)
    Labels       = reshape(KNN(CandidateDec),N,M) + rand(N,M);
    [~,best]     = max(Labels,[],2);
    OffspringDec = CandidateDec((best-1)*N+(1:N)',:);
    Offspring    = Problem.Evaluation(OffspringDec);
end
```
