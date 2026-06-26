# EMCMO

**Tags**: <2022> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Evolutionary multitasking-based constrained multiobjective optimization

## Reference
K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, and C. Yue. An evolutionary multitasking optimization framework for constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2022, 26(2): 263-277.

## Source Code

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

% This function is written by Kangjia Qiao

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
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EMCMO.m`
```matlab
classdef EMCMO < ALGORITHM
% <2022> <multi> <real/integer/label/binary/permutation> <constrained>
% Evolutionary multitasking-based constrained multiobjective optimization

%------------------------------- Reference --------------------------------
% K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, and C. Yue. An evolutionary
% multitasking optimization framework for constrained multi-objective
% optimization problems. IEEE Transactions on Evolutionary Computation,
% 2022, 26(2): 263-277.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    methods
        function main(Algorithm,Problem)
            Population{1} = Problem.Initialization();
            Population{2} = Problem.Initialization();
            Fitness{1}    = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{2}    = CalFitness(Population{2}.objs);
            transfer_state=0;
            
            cnt=0;
            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                cnt =cnt+1;
                if   transfer_state == 0
                    for i = 1: 2
                        valOffspring{i} = OperatorGAhalf(Problem,Population{i}(randi(Problem.N,1,Problem.N)));
                    end
                    
                    for i = 1:2
                        if i == 1
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{1},valOffspring{1},valOffspring{2}],Problem.N,i);
                        else
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{2},valOffspring{2},valOffspring{1}],Problem.N,i);
                        end
                    end
                    
                    if Problem.FE/Problem.maxFE >=0.2
                        transfer_state = 1;
                    end
                    
                else
                    
                    for i = 1: 2
                        MatingPool = TournamentSelection(2,Problem.N,Fitness{i});
                        valOffspring{i} = OperatorGAhalf(Problem,Population{i}(MatingPool));
                    end
                    [~,~,Next] = EnvironmentalSelection( [Population{2},valOffspring{2}],Problem.N,1);
                    succ_rate(1,cnt) =  (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
                    
                    [~,~,Next] = EnvironmentalSelection( [Population{1},valOffspring{1}],Problem.N,2);
                    succ_rate(2,cnt) =  (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
                    
                    for i = 1:2
                        if   succ_rate(i,cnt) >0
                            rand_number = randperm(Problem.N);
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{i},valOffspring{i},Population{2/i}(rand_number(1:Problem.N/2))],Problem.N,i);
                        else
                            [Population{i},Fitness{i},~] = EnvironmentalSelection( [Population{i},valOffspring{i},valOffspring{2/i}],Problem.N,i);
                        end
                    end
                end
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness,Next] = EnvironmentalSelection(Population,N,isOrigin)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    %% Calculate the fitness of each solution
    if isOrigin==1
        Fitness = CalFitness(Population.objs,Population.cons);
    else
        Fitness = CalFitness(Population.objs);
    end

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
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population = Population(rank);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```
