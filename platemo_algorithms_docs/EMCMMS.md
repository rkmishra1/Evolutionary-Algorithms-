# EMCMMS

**Tags**: <2024> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Evolutionary multitasking with a cooperative multistep mutation strategy

## Reference
K. Qiao, K. Yu, C. Yue, B. Qu, M. Liu, and J. Liang. A cooperative multistep mutation strategy for multiobjective optimization problems with deceptive constraints. IEEE Transactions on Systems, Man, and Cybernetics, 2024, 54(11): 6670-6682.

## Source Code

### `Auxiliray_task_EnvironmentalSelection.m`
```matlab
function [return_pop,return_Fitness] = Auxiliray_task_EnvironmentalSelection(Population,N,VAR)

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
    
    findex  = input_cons<=VAR;
    ifindex = input_cons>VAR;
    
    fPopulation  = Population(findex);
    ifPopulation = Population(ifindex);
    
    if isempty(fPopulation)
        ifFitness = CalFitness(ifPopulation.objs,ifPopulation.cons); % ifPopulation.cons
        Next2     = ifFitness < 1;
        if sum(Next2) <= N
            [~,Rank] = sort(ifFitness);
            Next2(Rank(1:N )) = true;
        elseif sum(Next2) > N
            Del  = Truncation(ifPopulation(Next2).objs, sum(Next2)-N );
            Temp = find(Next2);
            Next2(Temp(Del)) = false;
        end
    
        ifPopulation = ifPopulation(Next2);
        ifFitness    = ifFitness(Next2);
    
        [ifFitness,rank] = sort(ifFitness);
        ifPopulation     = ifPopulation(rank);
    
        fPopulation = [];
        fFitness    = [];
    elseif length(fPopulation) < N
        cons     = fPopulation.cons;
        cons(cons<0) = 0;
        cons     = sum(cons,2);
        fFitness = CalFitness([fPopulation.objs,cons]);
        Next     = fFitness < 1;
    
        [~,Rank] = sort(fFitness);
        Next(Rank(1:length(fPopulation) )) = true;
    
        fPopulation = fPopulation(Next);
        fFitness    = fFitness(Next);
    
        [fFitness,rank] = sort(fFitness);
        fPopulation     = fPopulation(rank);

        ifFitness = CalFitness(ifPopulation.objs,ifPopulation.cons); % ,
        Next2     = ifFitness < 1;
        if sum(Next2) <= N - length(fPopulation)
            [~,Rank] = sort(ifFitness);
            Next2(Rank(1:N - length(fPopulation) )) = true;
        elseif sum(Next2) > N - length(fPopulation)
            Del  = Truncation(ifPopulation(Next2).objs, sum(Next2)-(N - length(fPopulation)) );
            Temp = find(Next2);
            Next2(Temp(Del)) = false;
        end
    
        ifPopulation = ifPopulation(Next2);
        ifFitness    = ifFitness(Next2) + max(fFitness);
    
        [ifFitness,rank] = sort(ifFitness);
        ifPopulation     = ifPopulation(rank);
    
    elseif length(fPopulation) >= N
        cons     = fPopulation.cons;
        cons(cons<0) = 0;
        cons     = sum(cons,2);
        fFitness = CalFitness([fPopulation.objs,cons]);
        Next     = fFitness < 1;
        if sum(Next) <= N
            [~,Rank] = sort(fFitness);
            Next(Rank(1:N )) = true;
        elseif sum(Next) > N
            Del  = Truncation(fPopulation(Next).objs, sum(Next)-N );
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
    
        fPopulation = fPopulation(Next);
        fFitness    = fFitness(Next);
    
        [fFitness,rank] = sort(fFitness);
        fPopulation     = fPopulation(rank);
    
        ifPopulation = [];
        ifFitness    = [];
    end
    return_pop     = [fPopulation,ifPopulation];
    return_Fitness = [fFitness,ifFitness];
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

### `CMMS.m`
```matlab
function [ Offspring ] = CMMS(Population, Problem, p,Population2, flag_index,Direc_index)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    Nw          = min(Problem.D,Problem.N);
    permutation = randperm(Problem.N);
    Population  = Population(permutation);
    
    N = ceil(Problem.N*p);
    D = Problem.D;
    
    for i = 1 : N
        r1(i) = randi([1,Problem.N],1);
        while r1(i) == i
            r1(i) = randi([1,Problem.N],1);
        end
    
        r2(i) = randi([1,Problem.N],1);
        while r2(i)==i || r2(i)==r1(i)
            r2(i) = randi([1,Problem.N],1);
        end
    
        r3(i) = randi([1,Problem.N],1);
        while r3(i)==r2(i) || r3(i)==r1(i) || r3(i)==i
            r3(i) = randi([1,Problem.N],1);
        end
    end
    Offspring = [];
    for j = Direc_index
        start_X = Population(1:N).decs;
        if flag_index == 2
            end_X = Population2(r1).decs;
            if j == 2
                end_X = Problem.lower+Problem.upper - end_X;
            end
        elseif flag_index == 1
            end_X = Population(r2).decs;
            if j == 2
                end_X = Problem.lower+Problem.upper - end_X;
            end
        end
        Intervalmax = sqrt(sum((Problem.upper-Problem.lower).^2,2));
        Intervalmin = 0;
        RandSample  = Intervalmin + rand(N,Nw)*(Intervalmax-Intervalmin);
    
        for i = 1 : Nw
            result_pop = end_X + repmat(RandSample(:,i),1,D).* (start_X - end_X);
            result_pop = boundConstraint (result_pop, Population(1:N).decs, [Problem.lower;Problem.upper]);
            Offspring  = [Offspring;result_pop];
        end
    end
    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Lower       = repmat(Problem.lower,N,1);
    Upper       = repmat(Problem.upper,N,1);
    Site        = rand(N,D) < proM/D;
    mu          = rand(N,D);
    temp        = Site & mu<=0.5;
    
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    
    Offspring = Problem.Evaluation(Offspring);
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

### `EMCMMS.m`
```matlab
classdef EMCMMS < ALGORITHM
% <2024> <multi> <real/integer/label/binary/permutation> <constrained>
% Evolutionary multitasking with a cooperative multistep mutation strategy

%------------------------------- Reference --------------------------------
% K. Qiao, K. Yu, C. Yue, B. Qu, M. Liu, and J. Liang. A cooperative
% multistep mutation strategy for multiobjective optimization problems with
% deceptive constraints. IEEE Transactions on Systems, Man, and
% Cybernetics, 2024, 54(11): 6670-6682.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            run_rate    = 0.5;
            select_rate = 0.1;

            %% Initialization
            Population1 = Problem.Initialization();
            Fitness1    = CalFitness(Population1.objs,Population1.cons);

            Population2 = Problem.Initialization();
            Fitness2    = CalFitness(Population2.objs,Population2.cons);

            X    = 0;
            cons = [Population1.cons;Population2.cons];
            cons(cons<0) = 0;
            VAR0 = max(sum(cons,2));
            if VAR0 == 0
                VAR0 = 1;
            end
            cnt = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population2)
                cp  = (-log(VAR0)-6)/log(1-0.5);
                VAR = VAR0*(1-X)^cp;
                cnt = cnt + 1;
                if Problem.FE/Problem.maxFE < run_rate
                    flag_index  = 2;
                    Direc_index = 2;
                    Offspring3  = CMMS( Population1, Problem, select_rate,Population2,flag_index,Direc_index);

                    [Population1,Fitness1] = Main_task_EnvironmentalSelection([Population1,Offspring3],Problem.N,true);
                    [Population2,Fitness2] = Auxiliray_task_EnvironmentalSelection( [Population2,Offspring3], Problem.N,VAR);
                end

                MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                Offspring1  = OperatorGAhalf(Problem,[Population1(MatingPool1)]);

                MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                Offspring2  = OperatorGAhalf(Problem,[Population2(MatingPool2)]);

                [Population1,Fitness1] = Main_task_EnvironmentalSelection([Population1,Offspring1,Offspring2],Problem.N,true);
                [Population2,Fitness2] = Auxiliray_task_EnvironmentalSelection( [Population2,Offspring2,Offspring1], Problem.N,VAR);
                X = X + 1/(Problem.maxFE/Problem.N);
            end
        end
    end
end
```

### `Main_task_EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = Main_task_EnvironmentalSelection(Population,N,isOrigin)
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
    Population     = Population(rank);
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
