# MaOEA-CSS

**Tags**: <2017> <multi/many> <real/integer/label/binary/permutation>

## Description
Many-objective evolutionary algorithms based on coordinated selection

## Reference
Z. He and G. G. Yen. Many-objective evolutionary algorithms based on coordinated selection strategy. IEEE Transactions on Evolutionary Computation, 2017, 21(2): 220-233.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,Zmin,t,K)
% The environmental selection of MaOEA-CSS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the distance between each solution to the ideal point
    PopObj = Population.objs - repmat(Zmin,length(Population),1);
    Con    = sqrt(sum(PopObj.^2,2));

	%% Calculate the angle between each two solutions
    Angle = acos(1-pdist2(PopObj,PopObj,'cosine'));
    Angle(logical(eye(length(Population)))) = inf;
    
    %% Eliminate solutions one by one
    Remain = 1 : length(Population);
    while length(Remain) > K
        % Identify the two solutions A and B with the minimum angle
        [sortA,rank1] = sort(Angle(Remain,Remain),2);
        [~,rank2]     = sortrows(sortA);
        A = rank2(1);
        B = rank1(A,1);
        % Eliminate one of A and B
        if Con(Remain(A)) - Con(Remain(B)) > t
            Remain(A) = [];
        elseif Con(Remain(B)) - Con(Remain(A)) > t
            Remain(B) = [];
        else
            Remain(A) = [];
        end
    end
    % Population for next generation
    Population = Population(Remain);
end
```

### `MaOEACSS.m`
```matlab
classdef MaOEACSS < ALGORITHM
% <2017> <multi/many> <real/integer/label/binary/permutation>
% Many-objective evolutionary algorithms based on coordinated selection
% strategy
% t --- 0 --- Threshold value in environmental selection

%------------------------------- Reference --------------------------------
% Z. He and G. G. Yen. Many-objective evolutionary algorithms based on
% coordinated selection strategy. IEEE Transactions on Evolutionary
% Computation, 2017, 21(2): 220-233.
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
            t = Algorithm.ParameterSet(0);

            %% Generate random population
            Population = Problem.Initialization();
            Zmin       = min(Population.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = MatingSelection(Population.objs,Zmin);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Zmin       = min([Zmin;Offspring.objs],[],1);
                Population = EnvironmentalSelection([Population,Offspring],Zmin,t,Problem.N);
            end
        end
    end
end
```

### `MatingSelection.m`
```matlab
function MatingPool = MatingSelection(PopObj,Zmin)
% The mating selection of MaOEA-CSS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    
    %% Calculate the ASF value of each solution
    W      = max(1e-6,PopObj./repmat(sum(PopObj,2),1,M));
    PopObj = PopObj - repmat(Zmin,N,1);
    ASF    = max(PopObj./W,[],2);
    % Obtain the rank value of each solution's ASF value
    [~,rank]    = sort(ASF);
    [~,ASFRank] = sort(rank);
    
    %% Calculate the minimum angle of each solution to others
    Angle = acos(1-pdist2(PopObj,PopObj,'cosine'));
    Angle(logical(eye(N))) = inf;
    Amin  = min(Angle,[],2);
    
    %% Binary tournament selection
    MatingPool = zeros(1,N);
    for i = 1 : N
        p = randperm(N,2);
        if ASF(p(1)) < ASF(p(2)) && Amin(p(1)) > Amin(p(2))
            p = p(1);
        else
            p = p(2);
        end
        if rand < 1.0002-ASFRank(p)/N
            MatingPool(i) = p;
        else
            MatingPool(i) = randi(N);
        end
    end
end
```
