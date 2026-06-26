# TPCMaO

**Tags**: <2023> <many> <real/integer/label/binary/permutation> <constrained>

## Description
Three-population based constrained many-objective co-evolutionary algorithm

## Reference
Y. Tian, Z. Shi, Y. Zhang, L. Zhang, H. Zhang, and X. Zhang. Solving optimal power flow problems via a constrained many-objective co-evolutionary algorithm. Frontiers in Energy Research, 2023, 11: 1293193.

## Source Code

### `Archive.m`
```matlab
function Population = Archive(Population,N)
% Select feasible and non-dominated solutions by using SPEA2-CDP

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    fIndex     = all(Population.cons <= 0,2);
    Population = Population(fIndex);
    if isempty(Population)
        return;
    elseif size(Population,2) > N
        Fitness = CalFitness(Population.objs,Population.cons);
        Next    = Fitness < 1;
        if sum(Next) < N
            [~,Rank] = sort(Fitness);
            Next(Rank(1:N)) = true;
        elseif sum(Next) > N
            Del  = Truncation(Population(Next).objs,sum(Next)-N);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
        % Population for next generation
        Population = Population(Next);
    end
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    N = size(PopObj,1);
    
    %% Calculate the shifted distance between each two solutions
    Distance = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Distance(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    
    %% Truncation
    Del = false(1,N);
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
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
    
    %% Calculate the shifted distance between each two solutions
    Distance = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Distance(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    
    %% Calculate D(i)
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness,feasible_rate] = EnvironmentalSelection(Population,N,type,epsilon)
% Three environmental selection strategies

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    switch type
        case 1
            Fitness = CalFitness(Population.objs,Population.cons);
        case 2
            Fitness = CalFitness(Population.objs);
        case 3
            PopCon = Population.cons;
            CV     = sum(max(0,PopCon),2);
            index  = find(CV<=epsilon);
            PopCon(index(:,1),:) = 0;
            Fitness = CalFitness(Population.objs,PopCon);
    end
    [~,extreme] = min(Population.objs,[],1);
    Fitness(unique(extreme)) = 0;
    
    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
    CV = sum(max(0,Population.cons),2);
    feasible_rate = sum(CV==0)/size(Population,2);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    N = size(PopObj,1);
    
    %% Calculate the shifted distance between each two solutions
    Distance = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Distance(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    
    %% Truncation
    Del = false(1,N);
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `TPCMaO.m`
```matlab
classdef TPCMaO < ALGORITHM
% <2023> <many> <real/integer/label/binary/permutation> <constrained>
% Three-population based constrained many-objective co-evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, Z. Shi, Y. Zhang, L. Zhang, H. Zhang, and X. Zhang. Solving
% optimal power flow problems via a constrained many-objective
% co-evolutionary algorithm. Frontiers in Energy Research, 2023, 11:
% 1293193.
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
            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Population3 = Problem.Initialization();
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            Fitness2    = CalFitness(Population2.objs);
            Fitness3    = CalFitness(Population3.objs);

            %% Evaluate the Population
            cons = Population3.cons;
            cons(cons<0) = 0;
            epsilon0     = max(sum(cons,2));
            if epsilon0 == 0
                epsilon0 = 1;
            end
            epsilon = epsilon0;
            arch    = Archive(Population3,Problem.N);
            
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                % Evolve Population1 and Population2
                MatingPool = TournamentSelection(2,Problem.N,Fitness1);
                Offspring1 = OperatorGAhalf(Problem,Population1(MatingPool));
                MatingPool = TournamentSelection(2,Problem.N,Fitness2);
                Offspring2 = OperatorGAhalf(Problem,Population2(MatingPool));
                [tr2_1,~,feasible_rate] = EnvironmentalSelection([Population2,Offspring2],Problem.N,1);
                if feasible_rate > 0.5
                    rand_number1 = randperm(Problem.N);
                    [Population1,Fitness1] = EnvironmentalSelection([Population1,Offspring1,tr2_1(rand_number1(1:Problem.N/2))],Problem.N,1);
                    [Population2,Fitness2] = EnvironmentalSelection([Population2,Offspring2,Offspring1],Problem.N,2);
                else
                    [Population1,Fitness1] = EnvironmentalSelection([Population1,Offspring1,Offspring2],Problem.N,1);
                    [Population2,Fitness2] = EnvironmentalSelection([Population2,Offspring2,Offspring1],Problem.N,2);
                end
                % Evolve Population3
                cons = Population3.cons;
                cons(cons<0) = 0;
                if Problem.FE < 0.9*Problem.maxFE
                    if mean(sum(cons,2)<=0.02) < 0.9
                        epsilon = 0.9*epsilon;
                    else
                        epsilon = epsilon0*(1-Problem.FE/0.9/Problem.maxFE)^2;
                    end
                else
                    epsilon = 0;
                end
                MatingPool = TournamentSelection(2,Problem.N,Fitness3);
                Offspring  = OperatorGA(Problem,Population3(MatingPool));
                [Population3,Fitness3] = EnvironmentalSelection([Population3,Offspring],Problem.N,3,epsilon);
                % Output the non-dominated and feasible solutions
                arch = Archive([arch,Population3],Problem.N);
                if Problem.FE >= Problem.maxFE
                    Population1 = Archive([arch,Population1],Problem.N);
                end
            end
        end
    end
end
```
