# NSGA-II-conflict

**Tags**: <2014> <many> <real/integer/label/binary/permutation>

## Description
NSGA-II with conflict-based partitioning strategy

## Reference
A. L. Jaimes, C. A. Coello Coello, H. Aguirre, and K. Tanaka. Objective space partitioning using conflict information for solving many-objective problems. Information Sciences, 2014, 268: 305-327.

## Source Code

### `ConflictPartition.m`
```matlab
function Psi = ConflictPartition(PopObj,NS)
% Partitioning of objectives using conflict information

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    M = size(PopObj,2);
    
    %% Calculate the modified correlation matrix of objectives
    cMatrix = 1 - corr(PopObj);
    cMatrix = repmat(max(cMatrix,[],2)',M,1).*(1-eye(M));
    
    %% Partition the objectives
    k      = ceil(M/NS);
    Psi    = cell(1,NS);
    remain = 1 : M;
    for s = 1 : NS-1
        [~,L]  = sort(cMatrix(remain,remain),2);
        [~,i]  = min(L(:,k));
        Psi{s} = remain(L(i,1:k));
        remain(L(i,1:k)) = [];
    end
    Psi{NS} = remain;
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N,Psi)
% The environmental selection of NSGA-II-conflict

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Selected = zeros(1,N);
    FrontNo  = zeros(1,N);
    CrowdDis = zeros(1,N);
    PopObj   = Population.objs;
    for i = 1 : length(Psi)
        index = (i-1)*ceil(N/length(Psi))+1 : min(N,i*ceil(N/length(Psi)));
        [Selected(index),FrontNo(index),CrowdDis(index)] = SubSelection(PopObj(:,Psi{i}),length(index));
    end
    Population = Population(Selected);
end

function [Next,FrontNo,CrowdDis] = SubSelection(PopObj,N)
% Environmental selection based on only several objectives

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Next     = find(Next);
    FrontNo  = FrontNo(Next);
    CrowdDis = CrowdDis(Next);
end
```

### `NSGAIIconflict.m`
```matlab
classdef NSGAIIconflict < ALGORITHM
% <2014> <many> <real/integer/label/binary/permutation>
% NSGA-II with conflict-based partitioning strategy
% NS     ---  2 --- Number of subspaces
% cycles --- 10 --- Number of cycles

%------------------------------- Reference --------------------------------
% A. L. Jaimes, C. A. Coello Coello, H. Aguirre, and K. Tanaka. Objective
% space partitioning using conflict information for solving many-objective
% problems. Information Sciences, 2014, 268: 305-327.
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
            [NS,cycles] = Algorithm.ParameterSet(2,10);
            Gc          = ceil(Problem.maxFE/Problem.N/cycles);

            %% Generate random population
            Psi        = {1:Problem.M};
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N,Psi);

            %% Optimization
            phase = true;
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N,Psi);
                if ~phase && mod(ceil(Problem.FE/Problem.N),Gc)/Gc < 0.3
                    % Change to the approximation phase
                    Psi   = {1:Problem.M};
                    phase = true;
                elseif phase && mod(ceil(Problem.FE/Problem.N),Gc)/Gc >= 0.3
                    % Change to the partitioning phase
                    Psi   = ConflictPartition(Population.objs,NS);
                    phase = false;
                end
            end
        end
    end
end
```
