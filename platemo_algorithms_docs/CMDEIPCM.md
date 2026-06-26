# CMDEIPCM

**Tags**: <2022> <multi> <real/integer> <large/none> <constrained>

## Description
Constrained multiobjective differential evolution algorithm with an infeasible proportion control mechanism

## Reference
J. Liang, X. Ban, K. Yu, K. Qiao, and B. Qu. Constrained multiobjective differential evolution algorithm with infeasible-proportion control mechanism. Knowledge-Based Systems, 2022, 250: 109105.

## Source Code

### `CMDEIPCM.m`
```matlab
classdef CMDEIPCM < ALGORITHM
% <2022> <multi> <real/integer> <large/none> <constrained>
% Constrained multiobjective differential evolution algorithm with an infeasible proportion control mechanism

%------------------------------- Reference --------------------------------
% J. Liang, X. Ban, K. Yu, K. Qiao, and B. Qu. Constrained multiobjective
% differential evolution algorithm with infeasible-proportion control
% mechanism. Knowledge-Based Systems, 2022, 250: 109105.
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
            Population2 = Problem.Initialization();    
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            Fitness2    = CalFitness(Population2.objs);
    
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                Offspring1 = DEgenerator(Population1,Population2,Problem,Fitness1);
                Offspring2 = DEgenerator(Population2,Population1,Problem,Fitness2);
                [Population1,Fitness1] = EnvironmentalSelection1([Population1,Offspring1,Offspring2],Problem.N,true);
                pinfea     = 0.5*(1-cos((1-Problem.FE./Problem.maxFE)*pi));
                Population = [Population2,Offspring1,Offspring2];
                Obj        = Population.objs;
                MaxObj     = max(Obj);
                cons  = Population.cons;
                cons(cons<=0) = 0;
                CV    = sum(cons,2);
                Infea = find(CV > 0);
                p_in  = size(Infea,1)/size(Population.decs,1);
                if p_in > pinfea
                    Infea_Sec = randperm(size(Infea,1),floor(size(Population.decs,1)*(p_in - pinfea)));
                    Obj(Infea(Infea_Sec),:) = Obj(Infea(Infea_Sec),:) + MaxObj;
                end
                [Population2,Fitness2] = EnvironmentalSelection2(Population,Obj,Problem.N,false);
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

### `DEgenerator.m`
```matlab
function Offspring1=DEgenerator(Population1,Population2,Problem,Fitness1)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    p1 = Population1.decs;
    lu = [Problem.lower;Problem.upper];
    [popsize1,n] = size(p1);
    trial = zeros(popsize1,n);
    p2    = Population2.decs;
    
    for i = 1 : popsize1
        l = rand;
        if l <= 1/3
            F = 0.6;
        elseif l <= 2/3
            F = 0.8;
        else
            F = 1.0;
        end
    
        l = rand;
        if l <= 1/3
            CR = 0.1;
        elseif l <= 2/3
            CR = 0.2;
        else
            CR = 1.0;
        end
        indexset    = 1 : popsize1;
        indexset(i) = [];
        r1  = floor(rand*(popsize1-1))+1;
        xr1 = indexset(r1);
        indexset(r1) = [];
        r2  = floor(rand*(popsize1-2))+1;
        xr2 = indexset(r2);
        indexset(r2) = [];
        r3  = floor(rand*(popsize1-3))+1;
        xr3 = indexset(r3);
    
        Fr    = find(Fitness1 < 1);
        rr    = floor(rand*length(Fr))+1;
        best1 = Fr(rr);
    
        o = rand;
    
        if o < 1/2
            v = p1(i,:) + F*(p2(xr1,:)-p1(i,:)) + F*(p1(xr2,:)-p1(xr3,:));
        else
            v = p1(i,:) + F*(p1(best1,:)-p1(i,:)) + F*(p1(xr2,:)-p1(xr3,:));
        end
        % Handle the elements of the mutant vector which violate the boundary
        w = find(v < lu(1,:));
        if ~isempty(w)
            v(1, w) = 2 * lu(1, w) -  v(1, w);
            w1 = find( v(1, w) > lu(2, w));
            if ~isempty(w1)
                v(1, w(w1)) = lu(2, w(w1));
            end
        end
        y = find(v > lu(2, :));
        if ~isempty(y)
            v(1, y) =  2 * lu(2, y) - v(1, y);
            y1 = find(v(1, y) < lu(1, y));
            if ~isempty(y1)
                v(1, y(y1)) = lu(1, y(y1));
            end
        end
        % Binomial crossover
        t = rand(1, n) < CR;
        j_rand = floor(rand * n) + 1;
        t(1, j_rand) = 1;
        t_ = 1 - t;
        trial(i, :) = t .* v + t_ .* p1(i,:);
    end
    Offspring1 = Problem.Evaluation(trial);
end
```

### `EnvironmentalSelection1.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection1(Population,N,isOrigin)
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

### `EnvironmentalSelection2.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection2(Population,Obj,N,isOrigin)
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
        Fitness = CalFitness(Obj,Population.cons);
    else
        Fitness = CalFitness(Obj);
    end

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Obj(Next,:),sum(Next)-N);
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
