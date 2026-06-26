# IMTCMO

**Tags**: <2024> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Improved evolutionary multitasking-based CMOEA

## Reference
K. Qiao, J. Liang, K. Yu, C. Yue, H. Lin, D. Zhang, and B. Qu. Evolutionary constrained multiobjective optimization: scalable high-dimensional constraint benchmarks and algorithm. IEEE Transactions on Evolutionary Computation, 2024, 28(4): 965-979.

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

    findex = find(input_cons<=VAR);
    ifindex = find(input_cons>VAR);

    fPopulation = Population(findex);
    ifPopulation = Population(ifindex);

    if isempty(fPopulation)
        ifFitness = CalFitness(ifPopulation.objs,ifPopulation.cons);
        Next2 = ifFitness < 1;
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
        % Sort the population
        [ifFitness,rank] = sort(ifFitness);
        ifPopulation = ifPopulation(rank);

        fPopulation = [];
        fFitness = [];

    elseif length(fPopulation) <= N
        cons = fPopulation.cons;
        cons(cons<0)=0;
        cons = sum(cons,2);
        fFitness = CalFitness([fPopulation.objs,cons]);
        Next = fFitness < 1;

        [~,Rank] = sort(fFitness);
        Next(Rank(1:length(fPopulation) )) = true;

        fPopulation = fPopulation(Next);
        fFitness    = fFitness(Next);
        % Sort the population
        [fFitness,rank] = sort(fFitness);
        fPopulation = fPopulation(rank);
        %
        ifFitness = CalFitness(ifPopulation.objs,ifPopulation.cons); % ,
        Next2 = ifFitness < 1;
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
        % Sort the population
        [ifFitness,rank] = sort(ifFitness);
        ifPopulation = ifPopulation(rank);

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

        ifPopulation = [];
        ifFitness = [];
    end

    return_pop = [fPopulation,ifPopulation];
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

### `IMTCMO.m`
```matlab
classdef IMTCMO < ALGORITHM
% <2024> <multi> <real/integer/label/binary/permutation> <constrained>
% Improved evolutionary multitasking-based CMOEA

%------------------------------- Reference --------------------------------
% K. Qiao, J. Liang, K. Yu, C. Yue, H. Lin, D. Zhang, and B. Qu.
% Evolutionary constrained multiobjective optimization: scalable
% high-dimensional constraint benchmarks and algorithm. IEEE Transactions
% on Evolutionary Computation, 2024, 28(4): 965-979.
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
            %% Generate random population
            Population1 = Problem.Initialization();
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            Zmin1       = min(Population1.objs,[],1);

            Population2 = Problem.Initialization();
            Fitness2    = CalFitness(Population2.objs,Population2.cons);
            Zmin2       = min(Population2.objs,[],1);

            cons = [Population1.cons;Population2.cons];
            cons(cons<0) = 0;
            VAR0 = max(sum(cons,2));
            if VAR0 == 0
                VAR0 = 1;
            end
            X=0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                %% Udate the epsilon value
                cp  = (-log(VAR0)-6)/log(1-0.5);
                VAR = VAR0*(1-X)^cp;

                %% Offspring generation
                MatingPool = [Population1(randsample(Problem.N,Problem.N))];
                [Mate1,Mate2,Mate3]  = Neighbor_Pairing_Strategy(MatingPool,Zmin1);
                Offspring1(1:Problem.N/2) = OperatorDE_rand_1(Problem,Mate1(1:Problem.N/2), Mate2(1:Problem.N/2), Mate3(1:Problem.N/2));
                Offspring1(1+Problem.N/2:Problem.N) = OperatorDE_pbest_1_main(Population1, Problem.N/2, Problem, Fitness1, 0.1);

                MatingPool = [Population2(randsample(Problem.N,Problem.N))];
                [Mate1,Mate2,Mate3]  = Neighbor_Pairing_Strategy(MatingPool,Zmin2);
                Offspring2(1:Problem.N/2) = OperatorDE_rand_1(Problem,Mate1(1:Problem.N/2),Mate2(1:Problem.N/2),Mate3(1:Problem.N/2));
                Offspring2(1+Problem.N/2:Problem.N) = OperatorDE_pbest_1_main(Population2, Problem.N/2, Problem, Fitness2, 0.1);

                Zmin1 = min([Zmin1;Offspring1.objs],[],1);
                Zmin2 = min([Zmin2;Offspring2.objs],[],1);

                %% Environmental selection
                [Population1,Fitness1] = Main_task_EnvironmentalSelection([Population1,Offspring1,Offspring2], Problem.N, true);
                [Population2,Fitness2] = Auxiliray_task_EnvironmentalSelection([Population2,Offspring2,Offspring1], Problem.N, VAR);

                X=X+1/(Problem.maxFE/Problem.N);
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

### `Neighbor_Pairing_Strategy.m`
```matlab
function [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPop,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    Objs    = MatingPop.objs;
    [Num,M] = size(Objs);
    Objs    = (Objs - repmat(Zmin,Num,1));
    Objs    = Objs./repmat(sqrt(sum(Objs.^2,2)),1,M);

    CosV = Objs * Objs';
    CosV = CosV - 3*eye(Num,Num);

    [~,SInd] = sort(-CosV,2);

    Nr       = 10;
    Neighbor = SInd(:,1:Nr);

    Mate1 = MatingPop;

    P = ones(Num,2);
    for i = 1 : Num
        P(i,1:2) = Neighbor(i,randsample(Nr,2));
    end

    Mate2 = MatingPop(P(:,1));
    Mate3 = MatingPop(P(:,2));
end
```

### `OperatorDE_pbest_1.m`
```matlab
function Offspring = OperatorDE_pbest_1(Problem,Parent1,Parent2,Parent3,Parent4)
% The operator of differential evolution.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    %% Parameter setting
    [proM,disM] = deal(1,20);
    if isa(Parent1(1),'SOLUTION')
        evaluated = true;
        Parent1   = Parent1.decs;
        Parent2   = Parent2.decs;
        Parent3   = Parent3.decs;
        Parent4   = Parent4.decs;
    else
        evaluated = false;
    end
    [N,D] = size(Parent1);

    Fm    = [0.6,0.8,1.0];
    CRm   = [0.1,0.2,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));
    index = randi([1,length(CRm)],N,1);
    CR    = CRm(index);
    CR    = CR';

    %% Differental evolution
    Site = rand(N,D) < repmat(CR,1,D);

    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F(Site).*(Parent2(Site)-Offspring(Site) + Parent3(Site)-Parent4(Site));

    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `OperatorDE_pbest_1_main.m`
```matlab
function [ Offspring ] = OperatorDE_pbest_1_main(Population, popsize, Problem, Fitness, p)
% The operator of DE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    permutation = randperm(Problem.N);
    r0          = permutation;
    [r1,r2,r3]  = gnR1R2R3(Problem.N, r0);

    array = permutation(1:popsize);

    [~, indBest] = sort(Fitness, 'ascend');
    pNP          = max(round(p * Problem.N), 2);    % choose at least two best solutions  
    randindex    = ceil(rand(1, popsize) * pNP);	% select from [1, 2, 3, ..., pNP]
    randindex    = max(1, randindex);               % to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest        = Population(indBest(randindex));  % randomly choose one of the top 100p% solutions

    Offspring = OperatorDE_pbest_1(Problem,Population(array),pbest,Population(r1(1:popsize)),Population(r2(1:popsize)));
end
```

### `OperatorDE_rand_1.m`
```matlab
function Offspring = OperatorDE_rand_1(Problem,Parent1,Parent2,Parent3)
% The operator of DE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    %% Parameter setting
    [proM,disM] = deal(1,20);
    if isa(Parent1(1),'SOLUTION')
        evaluated  = true;
        Parent1 = Parent1.decs;
        Parent2 = Parent2.decs;
        Parent3 = Parent3.decs;
    else
        evaluated = false;
    end
    [N,D] = size(Parent1);

    Fm    = [0.6,0.8,1.0];
    CRm   = [0.1,0.2,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));
    index = randi([1,length(CRm)],N,1);
    CR    = CRm(index);
    CR    = CR';

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F(Site).*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `gnR1R2R3.m`
```matlab
function [r1,r2,r3] = gnR1R2R3(NP1, r0)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    NP0 = length(r0);

    r1  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else            % regenerate r1 if it is equal to r0
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
    end

    r2  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else            % regenerate r2 if it is equal to r0 or r1
            r2(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
    end

    r3  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 999
        pos = ((r3 == r1) | (r3 == r0) | (r3 == r2));
        if sum(pos) == 0
            break;
        else            % regenerate r2 if it is equal to r0 or r1
            r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
    end
end
```
