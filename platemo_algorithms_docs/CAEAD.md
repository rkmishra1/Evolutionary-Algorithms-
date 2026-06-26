# CAEAD

**Tags**: <2021> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Dual-population evolutionary algorithm based on alternative evolution and degeneration

## Reference
J. Zou, R. Sun, S. Yang, and J. Zheng. A dual-population algorithm based on alternative evolution and degeneration for solving constrained multi- objective optimization problems. Informaction Scinece, 2021, 239: 89-102.

## Source Code

### `CAEAD.m`
```matlab
classdef CAEAD < ALGORITHM 
% <2021> <multi> <real/integer/label/binary/permutation> <constrained>
% Dual-population evolutionary algorithm based on alternative evolution and degeneration
% type --- 1 --- Type of operator (1. DE 2. GA)

%------------------------------- Reference --------------------------------
% J. Zou, R. Sun, S. Yang, and J. Zheng. A dual-population algorithm based
% on alternative evolution and degeneration for solving constrained multi-
% objective optimization problems. Informaction Scinece, 2021, 239: 89-102.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

   methods
       function main(Algorithm,Problem)
           type = Algorithm.ParameterSet(1);
           
           %% Generate random population
           Population1 = Problem.Initialization();
           Population2 = Problem.Initialization();
           Fitness1    = CalFitness(Population1.objs,Population1.cons,0);
           Fitness2    = CalFitness(Population2.objs,Population2.cons,1e6);
           
           min_epsilon      = 1e-4;
           change_threshold = 1e-2;
           max_change       = 1;
           epsilon_k        = 1e8;
           tao              = 0.05;
           max_ep           = 0;
           gen              = 1;
           stage            = false;
           
           %% Optimization
           while Algorithm.NotTerminated(Population1)
               pop_cons2      = Population2.cons;
               cv2            = overall_cv(pop_cons2);
               population     = [Population2.decs,Population2.objs,cv2];
               Objvalues(gen) = sum(sum(Population2.objs,1));
               ep(gen)        =  epsilon_k;
               if type == 1
                   MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                   MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);
                   Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                   Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
               elseif type == 2
                   MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                   MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                   Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                   Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
               end
               [FrontNo2,~] = NDSort(Population2.objs,size(Population2.objs,1));
               NC2 = size(find(FrontNo2==1),2);
               if gen ~= 1
                   max_change = abs(Objvalues(gen)-Objvalues(gen-1));
               end               
               if max_change <= change_threshold &&NC2 == Problem.N && stage == false
                   epsilon_k = max(population(:,end),[],1);
                   stage = true;
               end
               Offspring3 = [];
               if stage == true
                   if type == 1
                       Offspring3 = OperatorDE(Problem,Population1,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                   elseif type == 2
                       for i=1:Problem.N/2
                           Offtemp = OperatorGAhalf(Problem,[Population1(MatingPool1(i)),Population2(MatingPool2(i))]);
                           Offspring3 = [Offspring3,Offtemp];
                       end
                   end
               end
               if stage == true
                   [stage,epsilon_k] =  update_epsilon(stage,tao,epsilon_k,max_ep,min_epsilon);
               end
               if epsilon_k < 9e5
                   max_ep = max(max_ep,epsilon_k);
               end
               [Population1,Fitness1] = EnvironmentalSelection([Population1,Offspring1,Offspring2,Offspring3],Problem.N,true,0);
               [Population2,Fitness2] = EnvironmentalSelection([Population2,Offspring2],Problem.N,false,epsilon_k);
               gen = gen+1;
           end
       end
   end
end

function result = overall_cv(cv)
    cv(cv <= 0) = 0;cv = abs(cv);
    result = sum(cv,2);
end

function [stage,result] = update_epsilon(stage,tao,epsilon_k,epsilon_0,min_epsilon)
    if epsilon_k > min_epsilon
        result = (1 - tao) * epsilon_k;
        stage = true;
    else
        result = epsilon_0;
        stage = false;
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon,epsilon)
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
    CV(find(CV<=epsilon)) = 0;
    
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

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection(Population,N,isOrigin,epsilon)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    if isOrigin
        Fitness = CalFitness(Population.objs,Population.cons,0);
    else
        Fitness = CalFitness(Population.objs,Population.cons,epsilon);
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
