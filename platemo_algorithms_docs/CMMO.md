# CMMO

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <multimodal>

## Description
Coevolutionary multi-modal multi-objective optimization framework

## Reference
F. Ming, W. Gong, L. Wang, and L. Gao. Balancing convergence and diversity in objective and decision spaces for multimodal multi-objective optimization. IEEE Transactions on Emerging Topics in Computational Intelligence, 2023, 7(2): 474-486.

## Source Code

### `CMMO.m`
```matlab
classdef CMMO < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <multimodal>
% Coevolutionary multi-modal multi-objective optimization framework

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, L. Wang, and L. Gao. Balancing convergence and
% diversity in objective and decision spaces for multimodal multi-objective
% optimization. IEEE Transactions on Emerging Topics in Computational
% Intelligence, 2023, 7(2): 474-486.
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
            
            %% parameters for epsilon and initialization of epsilon
            G = ceil(Problem.maxFE/(2 * Problem.N));
            gen = 1;
            Tc = 0.8 * G;
            tao = 0.1;
            epsilon_0 = sum(max(Population2.objs,[],1),2);
            threshold = 0.1;
            epsilon_k = epsilon_0;
            
            %% calculate fitness of populations
            [D_Dec,D_Pop,Fitness1] = CalFitness(Population1.objs,Population1.decs);
            [Fitness2,D] = CalFitnessDecEpsilon(Population2.objs,Population2.decs,epsilon_k);

            %% Optimization
            while Algorithm.NotTerminated(Population1)         
                % The value of e(k) and the search strategy are set.
                if gen < 0.2 * G
                    epsilon_k = 0;
                elseif gen < Tc
                    epsilon_k = (1 - tao) * epsilon_k;
                    if epsilon_k <= threshold
                        epsilon_k = (gen/G) * epsilon_0;
                    end
                end
                
                MatingPool1 = TournamentSelection(2,Problem.N,D_Dec,D_Pop,Fitness1);
                MatingPool2 = TournamentSelection(2,Problem.N,D,Fitness2);
                Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                [Population1,Fitness1,D_Dec,D_Pop] = EnvironmentalSelection([Population1,Offspring1,Offspring2],Problem.N);
                [Population2,Fitness2,D] = EnvironmentalSelectionDec([Population2,Offspring1,Offspring2],Problem.N,epsilon_k);
                
                gen = ceil(Problem.FE/(2 * Problem.N));
                objs = sum(Population2.objs,2);
                max_obj = max(objs);
                epsilon_0 = min(max_obj,epsilon_0);
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function [D_Dec,D_Pop,Fitness] = CalFitness(PopObj,PopDec)
% Calculate the fitness of each solution based on original objective space

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
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D_Pop = 1./(Distance(:,floor(sqrt(N)))+2);
    
    Distance = pdist2(PopDec,PopDec);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D_Dec = 1./(Distance(:,floor(sqrt(N)))+2);

    D = D_Pop + D_Dec;
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFitnessDecEpsilon.m`
```matlab
function [Fitness,D] = CalFitnessDecEpsilon(PopObj,PopDec,epsilon)
% Calculate the fitness of each solution based on epsilon objective space

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
            k = any(PopObj(j,:)-PopObj(i,:) > epsilon) - any(PopObj(j,:)-PopObj(i,:) < epsilon);
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
    D_Dec = 1./(Distance(:,floor(sqrt(N)))+2);
    
    D = D_Dec;
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness,D_Dec,D_Pop] = EnvironmentalSelection(Population,N)
% The environmental selection of SPEA2 based on objective space

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
    [D_Dec,D_Pop,Fitness] = CalFitness(Population.objs,Population.decs);

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
    D_Pop      = D_Pop(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population = Population(rank);
    D_Dec = D_Dec(rank);
    D_Pop = D_Pop(rank);
end

function Del = Truncation(PopObj,PopDec,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance_Pop = pdist2(PopObj,PopObj);
    Distance_Pop(logical(eye(length(Distance_Pop)))) = inf;
    
    Distance_Dec = pdist2(PopDec,PopDec);
    Distance_Dec(logical(eye(length(Distance_Dec)))) = inf;
    
    D = Distance_Pop + Distance_Dec;
    
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(D(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `EnvironmentalSelectionDec.m`
```matlab
function [Population,Fitness,D] = EnvironmentalSelectionDec(Population,N,epsilon)
% The environmental selection based on decision space

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
    [Fitness,D] = CalFitnessDecEpsilon(Population.objs,Population.decs,epsilon);

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
    Population = Population(rank);
    D = D(rank);
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
