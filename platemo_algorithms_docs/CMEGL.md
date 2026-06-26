# CMEGL

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Constrained evolutionary multitasking with global and local auxiliary tasks

## Reference
K. Qiao, J. Liang, Z. Liu, K. Yu, C. Yue, and B. Qu. Evolutionary multitasking with global and local auxiliary tasks for constrained multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2023, 10(10): 1951-1964.

## Source Code

### `CMEGL.m`
```matlab
classdef CMEGL < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <constrained>
% Constrained evolutionary multitasking with global and local auxiliary tasks

%------------------------------- Reference --------------------------------
% K. Qiao, J. Liang, Z. Liu, K. Yu, C. Yue, and B. Qu. Evolutionary
% multitasking with global and local auxiliary tasks for constrained
% multi-objective optimization. IEEE/CAA Journal of Automatica Sinica,
% 2023, 10(10): 1951-1964.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence MaOperatorGAzine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population1 = Problem.Initialization(); % Main task
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            Population2 = Problem.Initialization(); % Global auxiliary task
            Fitness2   = CalFitness(Population2.objs);
            Population3 = Problem.Initialization(); % Local auxiliary task
            Fitness3   = CalFitness(Population3.objs,Population3.cons);
            
            % Calculate the constraint boundary of local auxiliary task
            cons = Population1.cons;
            cons(cons<0) = 0;
            cons =sum(cons,2);
            index =find(cons>0);
            if isempty(index)
                VAR0 = 0;
            else
                VAR0 =  mean(cons(index));
            end
            cnt  = 0;	% index of generation
            flag = 0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                cnt =cnt +1;
                if flag == 0
                    std_obj(cnt,:) = std(Population2.objs,[],1);
                    if cnt>100
                        if  sum(std(std_obj(cnt-100:cnt,:),[],1)<0.5) == Problem.M
                            flag = 1;
                        end
                    end
                end
                %% Offspring generation
                MatingPool = TournamentSelection(2,Problem.N,Fitness1);
                Offspring1 = OperatorGAhalf(Problem,[Population1(MatingPool)]);

                if flag == 0
                    MatingPool = TournamentSelection(2,Problem.N,Fitness2);
                    Offspring2 = OperatorGAhalf(Problem,[Population2(MatingPool)]);
                else
                    Offspring2 = [];
                end

                if length(Population3) <=1
                    Offspring3 = [];
                else
                    MatingPool = TournamentSelection(2,min(length(Population3),Problem.N/2),Fitness3);
                    Offspring3 = OperatorGA(Problem,[Population3(MatingPool)]);
                end

                %% Environmental selection
                [Population1,Fitness1] = EnvironmentalSelection([Population1,Offspring2,Offspring3],Problem.N,true);  
                [Population1,Fitness1] = EnvironmentalSelection([Population1,Offspring1],Problem.N,true);  

                if flag == 0
                    [Population2,Fitness2] = EnvironmentalSelection([Population2,Offspring1,Offspring2,Offspring3],Problem.N,false);
                end

                [Population3,Fitness3] = EnvironmentalSelection_LAT([Population3,Offspring1,Offspring2,Offspring3],Problem.N,VAR0);

                % Calculate the constraint boundary of local auxiliary task
                cons = Offspring1.cons;
                cons(cons<0) = 0;
                cons = sum(cons,2);
                index = find(cons>0);
                if isempty(index)
                    VAR0 = 0;
                else
                    VAR0 = mean(cons(index));
                end
            end
        end
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
function [Population,Fitness] = EnvironmentalSelection(Population,N,isOrigin)
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

### `EnvironmentalSelection_LAT.m`
```matlab
function [return_pop,return_Fitness] = EnvironmentalSelection_LAT(Population,N,VAR)
% Multi-objective-based CHT is used to sort the Population of local auxiliary task

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    input_cons = Population.cons;
    input_cons(input_cons<0) = 0;
    input_cons = sum(input_cons,2);

    findex = find(input_cons<=VAR);
    fPopulation = Population(findex);

    if isempty(fPopulation)
        fPopulation = [];
        fFitness = [];
    elseif length(fPopulation) <= N
        cons = fPopulation.cons;
        cons(cons<0)=0;
        cons = sum(cons,2);
        fFitness = CalFitness([fPopulation.objs,cons]);

        % Sort the population
        [fFitness,rank] = sort(fFitness);
        fPopulation = fPopulation(rank);
        fFitness = fFitness(rank);
    elseif length(fPopulation) > N
        cons = fPopulation.cons;
        cons(cons<0)=0;
        cons = sum(cons,2);
        fFitness = CalFitness([fPopulation.objs,cons]);
        Next = fFitness < 1;
        if sum(Next) <= N
            [~,Rank] = sort(fFitness);
            Next(Rank(1:N )) = true;
        elseif sum(Next) > N
            Del  = Truncation(fPopulation(Next).objs, sum(Next)-N );
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end

        fPopulation = fPopulation(Next);
        fFitness   = fFitness(Next);
        % Sort the population
        [fFitness,rank] = sort(fFitness);
        fPopulation = fPopulation(rank);

    end

    return_pop = [fPopulation];
    return_Fitness = [fFitness];
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        if isempty(Remain)
            keyboard
        end
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```
