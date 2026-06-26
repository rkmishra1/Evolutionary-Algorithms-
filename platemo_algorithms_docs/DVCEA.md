# DVCEA

**Tags**: <2025> <multi/many> <real/integer> <large/none> <constrained>

## Description
Decision variables classification-based evolutionary algorithm

## Reference
X. Ban, J. Liang, K. Qiao, K. Yu, Y. Wang, J. Zhu, B. Qu. A decision variables classification-based evolutionary algorithm for constrained multi-objective optimization problems. IEEE/CAA Journal of Automatica Sinica, 2025, 12(9): 1830-1849.

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
    Distance = sort(Distance,2); %每一行从小到大排列
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFitness_E.m`
```matlab
function Fitness = CalFitness_E(PopObj,PopCon,epsilon)
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
        CV(CV<=epsilon)=0;
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

### `DEgenerator_better.m`
```matlab
function Offspring1=DEgenerator_better(Population1,Problem,FEA,epsilon)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    p1           = Population1.decs;
    [popsize1,n] = size(p1);
    Offspring    = [];
    Fitness1     = CalFitness_E(Population1.objs,Population1.cons,epsilon);
    
    while length(Population1) > 1
        l = rand;
        if l <= 1/3
            F = .6;
        elseif l <= 2/3
            F = 0.8;
        else
            F = 1.0;
        end
        l = rand;
        if l <= 1/3
            CR = .1;
        elseif l <= 2/3
            CR = 0.2;
        else
            CR = 1.0;
        end
        indexset     = 1 : popsize1;
        r1           = floor(rand*(popsize1-1))+1;
        xr1          = indexset(r1);
        indexset(r1) = [];
        r2           = floor(rand*(popsize1-2))+1;
        xr2          = indexset(r2);
    
        if Fitness1(xr1) < Fitness1(xr2)
            best1  = xr1;
            worst1 = xr2;
        else
            best1  = xr2;
            worst1 = xr1;
        end
    
        v = p1(best1,:) + F*(p1(best1,:)-p1(worst1,:));
    
        o1    = rand(1,Problem.D);
        index = find(o1 <= 0.5);
    
        p1(worst1,index) = p1(best1,index);
        v2 = p1(worst1,:);
    
        %% Binomial crossover
        t      = rand(1, n) < CR;
        j_rand = floor(rand * n) + 1;
        t(1, j_rand) = 1;
        t_     = 1 - t;
    
        v = t .* v + t_ .* p1(best1,:);
    
        vv1      = p1(best1,:);
        vv1(FEA) = v(FEA);
        vv2      = p1(worst1,:);
        vv2(FEA) = v2(FEA);
    
        Offspring = [Offspring;vv1;vv2];
        Population1(:,[xr1,xr2]) = [];
        p1 = Population1.decs;
        [popsize1,n] = size(p1);
    end

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Lower       = repmat(Problem.lower,Problem.N,1);
    Upper       = repmat(Problem.upper,Problem.N,1);
    Site        = rand(Problem.N,Problem.D) < proM/Problem.D;
    mu          = rand(Problem.N,Problem.D);
    temp        = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    
    Offspring1 = Problem.Evaluation(Offspring);
end
```

### `DVCEA.m`
```matlab
classdef DVCEA < ALGORITHM
% <2025> <multi/many> <real/integer> <large/none> <constrained>
% Decision variables classification-based evolutionary algorithm

%------------------------------- Reference --------------------------------
% X. Ban, J. Liang, K. Qiao, K. Yu, Y. Wang, J. Zhu, B. Qu. A decision 
% variables classification-based evolutionary algorithm for constrained 
% multi-objective optimization problems. IEEE/CAA Journal of Automatica 
% Sinica, 2025, 12(9): 1830-1849.
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
            Population  = Problem.Initialization();
            [~, C]      = kmeans(Population.decs,5);
            [FEA,INFEA] = Variable_classification(Problem,Population,C);

            cons  = Population.cons;
            cons(cons <= 0) = 0;
            conss = sum(cons,2);
            epsilon0 = max(conss);
            if epsilon0 == 0
                epsilon0 = 1;
            end
            Fitness = CalFitness_E(Population.objs,Population.cons,epsilon0);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                cp        = (-log(epsilon0)-6)/log(1-0.5);
                epsilon   = epsilon0*(1-Problem.FE/Problem.maxFE)^cp;
                Offspring = OperatorDE_pbest_1_main(Population, Problem.N, Problem, Fitness, FEA, 0.1);
                [Population,~] = Improve_E_EnvironmentalSelection([Population,Offspring],Problem.N,epsilon);
                Offspring = DEgenerator_better(Population,Problem,INFEA,epsilon);
                [Population,Fitness] = Improve_E_EnvironmentalSelection([Population,Offspring],Problem.N,epsilon);
            end
        end
    end
end
```

### `Improve_E_EnvironmentalSelection.m`
```matlab
function [return_pop,return_Fitness] = Improve_E_EnvironmentalSelection(Population,N,VAR)

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
        ifPopulation     = ifPopulation(rank);
        fPopulation      = [];
        fFitness         = [];
    elseif length(fPopulation) <= N
        cons = fPopulation.cons;
        cons(cons<0) = 0;
        cons = sum(cons,2);
        fFitness = CalFitness([fPopulation.objs,cons]);
        Next = fFitness < 1;
        
        [~,Rank] = sort(fFitness);
        Next(Rank(1:length(fPopulation) )) = true;
        
        fPopulation = fPopulation(Next);
        fFitness    = fFitness(Next);
        % Sort the population
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
        % Sort the population
        [ifFitness,rank] = sort(ifFitness);
        ifPopulation = ifPopulation(rank);
    elseif length(fPopulation) > N
        cons         = fPopulation.cons;
        cons(cons<0) = 0;
        cons         = sum(cons,2);
        fFitness     = CalFitness([fPopulation.objs,cons]);
        Next         = fFitness < 1;
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
        % Sort the population
        [fFitness,rank] = sort(fFitness);
        fPopulation     = fPopulation(rank);
        ifPopulation    = [];
        ifFitness       = [];
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

### `OperatorDE_pbest_1.m`
```matlab
function Offspring = OperatorDE_pbest_1(Problem,Parent1,Parent2,Parent3,Parent4)

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
        Parent4 = Parent4.decs;
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
function [ Offspring ] = OperatorDE_pbest_1_main(Population, popsize, Problem, Fitness, FEA, p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    OffDec      = Population.decs;
    permutation = randperm(Problem.N);
    r0          = permutation;
    [r1,r2,~]   = gnR1R2R3(Problem.N, r0);
    
    array = permutation(1:popsize);
    
    [~, indBest] = sort(Fitness, 'ascend');
    pNP          = max(round(p * Problem.N), 2);    % choose at least two best solutions  
    randindex    = ceil(rand(1, popsize) * pNP);    % select from [1, 2, 3, ..., pNP]
    randindex    = max(1, randindex);               % to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest        = Population(indBest(randindex));  % randomly choose one of the top 100p% solutions
    
    NewDec        = OperatorDE_pbest_1(Problem,Population(array).decs,pbest.decs,Population(r1(1:popsize)).decs,Population(r2(1:popsize)).decs);
    OffDec(:,FEA) = NewDec(:,FEA);
    Offspring     = OffDec;
    Offspring     = Problem.Evaluation(Offspring);
end
```

### `Variable_classification.m`
```matlab
function [FEA,INFEA] = Variable_classification(Problem,Population,C)
% Detect the kind of each decision variable

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    pop_dec = Population.decs;
    D       = size(pop_dec,2);
    PN      = 4;
    SN      = 5;

    theta = 0.00001;
    for d = 1 : D
        Per = (Problem.lower(d):(Problem.upper(d)-Problem.lower(d))/(PN-1):Problem.upper(d))';
        for j = 1 : SN
            pop_dec_SN      = repmat(C(j,:),PN,1);
            pop_dec_SN(:,d) = Per;
            pop_fenxi       = Problem.Evaluation(pop_dec_SN);
            pop_fenxi_con   = pop_fenxi.cons;
            pop_fenxi_con(pop_fenxi_con < 0) = 0;
            pop_fenxi_con   = sum(pop_fenxi_con,2);
            pop_dec_con     = pop_fenxi_con;
            VarCon(j,d)     = std(pop_dec_con);
        end
    end
    meanCon = mean(VarCon);
    FEA     = find(meanCon>theta);
    INFEA   = find(meanCon<=theta);
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
    
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 1001
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else   % regenerate r1 if it is equal to r0
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000 % this has never happened so far
            error('Can not genrate r1 in 1000 iterations');
        end
    end
    
    r2  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 1001
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else   % regenerate r2 if it is equal to r0 or r1
            r2(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000 % this has never happened so far
            error('Can not genrate r2 in 1000 iterations');
        end
    end
    
    r3  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 1001
        pos = ((r3 == r1) | (r3 == r0) | (r3 == r2));
        if sum(pos) == 0
            break;
        else   % regenerate r2 if it is equal to r0 or r1
            r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000 % this has never happened so far
            error('Can not genrate r3 in 1000 iterations');
        end
    end
end
```
