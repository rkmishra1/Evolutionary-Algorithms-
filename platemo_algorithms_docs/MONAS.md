# MONAS

**Tags**: <2025> <multi> <real/integer/label/binary/permutation> <multimodal>

## Description
Multi-objective neural architecture search

## Reference
F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. An evolutionary framework for multi-objective neural architecture search. IEEE Transactions on Evolutionary Computation, 2025.

## Source Code

### `CalFitness.m`
```matlab
function [D,Fitness] = CalFitness(PopObj,PopDec)
% Calculate the fitness of each solution based on original objective space

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D_Pop    = 1./(Distance(:,floor(sqrt(N)))+2);
    D        = D_Pop;
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFitnessDecAux.m`
```matlab
function [Fitness,D] = CalFitnessDecAux(PopObj,PopDec)
% Calculate the fitness of each solution in auxialiry population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    N = size(PopObj,1);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopDec,PopDec);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D_Dec    = 1./(Distance(:,floor(sqrt(N)))+2);
    D        = D_Dec;
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness,D_Dec] = EnvironmentalSelection(Population,N)
% The environmental selection of SPEA2 based on objective space

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    [D_Dec,Fitness] = CalFitness(Population.objs,Population.decs);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,Population(Next).decs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
    D_Dec      = D_Dec(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population     = Population(rank);
    D_Dec          = D_Dec(rank);
end

function Del = Truncation(PopObj,PopDec,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance_Pop = pdist2(PopObj,PopObj);
    Distance_Pop(logical(eye(length(Distance_Pop)))) = inf;
    
    D = Distance_Pop;
    
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(D(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `EnvironmentalSelectionAux.m`
```matlab
function [Population,Fitness,D] = EnvironmentalSelectionAux(Population,N)
% The auxiliary environmental selection for decision space

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    %% Calculate the fitness of each solution
    [Fitness,D] = CalFitnessDecAux(Population.objs,Population.decs);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).decs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
    D = D(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population     = Population(rank);
    D              = D(rank);
end

function Del = Truncation(PopDec,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopDec,PopDec);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopDec,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `MONAS.m`
```matlab
classdef MONAS < ALGORITHM
% <2025> <multi> <real/integer/label/binary/permutation> <multimodal>
% Multi-objective neural architecture search

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. An evolutionary framework
% for multi-objective neural architecture search. IEEE Transactions on
% Evolutionary Computation, 2025.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            
            %% calculate fitness of populations
            [Fitness1,D_Dec] = CalFitness(Population1.objs,Population1.decs);
            [Fitness2,D]     = CalFitnessDecAux(Population2.objs,Population2.decs);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                if Problem.FE <= Problem.maxFE/2
                    MatingPool1 = TournamentSelection(2,Problem.N,D_Dec,Fitness1);
                    MatingPool2 = TournamentSelection(2,Problem.N,D,Fitness2);
                    Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                    Offspring2  = OperatorGA(Problem,Population2(MatingPool2));
                else
                    MatingPool1 = TournamentSelection(2,2*Problem.N,D_Dec,Fitness1);
                    MatingPool2 = TournamentSelection(2,2*Problem.N,D,Fitness2);
                    Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                    Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                end
                Offspring = [Offspring1,Offspring2];
                [Population1,Fitness1,D_Dec] = EnvironmentalSelection([Population1,Offspring],Problem.N);
                [Population2,Fitness2,D]     = EnvironmentalSelectionAux([Population2,Offspring],Problem.N);
            end
        end
    end
end
```
