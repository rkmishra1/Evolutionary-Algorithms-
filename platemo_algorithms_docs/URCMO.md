# URCMO

**Tags**: <2023> <multi> <real/integer> <constrained>

## Description
Utilizing the relationship between constrained and unconstrained Pareto fronts for constrained multi-objective optimization

## Reference
J. Liang, K. Qiao, K. Yu, B. Qu, C. Yue, W. Guo, and L. Wang. Utilizing the relationship between unconstrained and constrained Pareto fronts for constrained multi-objective optimization. IEEE Transactions on Cybernetics, 2023, 53(6): 3873-3886.

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

### `Classification.m`
```matlab
function [ flag,ll] = Classification( Population1,Population2, beita)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    ll = -1;

    pop1  = Population1.decs;
    conv1 = Population1.cons;
    conv1(conv1<=0) = 0;
    conv1 = sum(conv1,2);
    obj1  = Population1.objs;
    conv2 = Population2.cons;
    conv2(conv2<=0) = 0;
    conv2 = sum(conv2,2);
    obj2  = Population2.objs;

    [FrontNo,MaxFNo] = NDSort(obj1,conv1,inf);
    x1 = find(FrontNo==1);
    [FrontNo,MaxFNo] = NDSort(obj2,inf);
    x2 = find(FrontNo==1);


    if length(find(conv2(x2)<=0))==0 % When all solutions of population2 are infeasible, type-IV. Herein, 3 indicates S3
        flag = 3;
    elseif length(find(conv2(x2)>0))==0 % When all solutions of population2 are feasible, type-I.
        flag = 1;
    elseif  length(find(conv2>0))>0 && length(find(conv2>0))< size(pop1,1)  % When population2 has both feasible and infeasible solutions
        obj1 = obj1(x1,:);
        obj2 = obj2(x2,:);
        [FrontNo,MaxFNo] = NDSort([obj1;obj2],inf); 

        ll = length((find(FrontNo(1:length(x1))==1)))/length(FrontNo(1:length(x1)));  
        %  ll indicates the ratio that pop1(x1) belong to the first level in
        %  the combination of pop1(x1) and pop2(x2)

        if ll > beita
            flag = 1;
        elseif ll < 1 - beita
            flag = 3;
        else
            flag = 2;
        end
    end
end
```

### `DE_current_to_other_pbest_1.m`
```matlab
function [ offspring ] = DE_current_to_other_pbest_1(Problem, Population, popsize, other_Fitness, Population2, p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    Fm  = [0.6,0.8,1.0];
    CRm = [0.1,0.2,1.0];
    lu  = [Problem.lower;Problem.upper];

    index = randi([1,length(Fm)],popsize,1);
    F     = Fm(index);
    F     = F';

    permutation = randperm(Problem.N);
    r0          = permutation;
    [r1, r2,r3] = gnR1R2R3(Problem.N,  r0);

    array     = permutation(1:popsize);
    pop       = Population.decs;
    pop1      = pop(array,:);
    other_pop = Population2.decs;

    [~, indBest] = sort(other_Fitness, 'ascend');
    pNP          = max(round(p * Problem.N), 2); % choose at least two best solutions  %ŽһֱѡǰŽȽ
    randindex    = ceil(rand(1, popsize) * pNP); % select from [1, 2, 3, ..., pNP]
    randindex    = max(1, randindex); % to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest        = other_pop(indBest(randindex), :); % randomly choose one of the top 100p% solutions

    vi = pop1 + F(:, ones(1, Problem.D)) .*(   pbest - pop1  + pop(r2(array),:) - pop(r3(array),:));
    vi = boundConstraint(vi, pop1, lu);
    u  = vi;
    
    offspring = Problem.Evaluation(u);
end
```

### `DE_current_to_rand_1.m`
```matlab
function [ offspring ] = DE_current_to_rand_1(Problem, Population, popsize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    Fm  = [0.6,0.8,1.0];
    CRm = [0.1,0.2,1.0];
    lu  = [Problem.lower;Problem.upper];

    index = randi([1,length(Fm)],popsize,1);
    F     = Fm(index);
    F     = F';
    
    permutation = randperm(Problem.N);
    r0          = permutation;
    [r1, r2,r3] = gnR1R2R3(Problem.N,  r0);

    array = permutation(1:popsize);
    pop   = Population.decs;

    pop1 = pop(1:popsize,:);

    vi = pop1 +  repmat(rand(popsize,1),1,Problem.D) .*(pop(r1(array),:) - pop1 ) + F(:, ones(1, Problem.D)) .*(pop(r2(array),:) - pop(r3(array),:));
    vi = boundConstraint(vi, pop1, lu);
    u  = vi;
    
    offspring = Problem.Evaluation(u);
end
```

### `DE_transfer.m`
```matlab
function [ offspring ] = DE_transfer(Problem, Population1, Population2, popsize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    Fm  = [0.6,0.8,1.0];
    CRm = [0.1,0.2,1.0];

    index = randi([1,length(Fm)],popsize,1);
    F     = Fm(index);
    F     = F';
    index = randi([1,length(CRm)],popsize,1);
    CR    = CRm(index);
    CR    = CR';

    index       = randi(Problem.N, popsize,1);
    permutation = randperm(Problem.N);
    array       = permutation(1:popsize);

    pop1 = Population1(array).decs;
    pop2 = Population2.decs;

    vi = pop2(index,:);

    mask  = rand(popsize, Problem.D) > CR(:, ones(1, Problem.D)); % mask is used to indicate which elements of ui comes from the parent
    rows  = (1 : popsize)'; cols = floor(rand(popsize, 1) * Problem.D)+1; % choose one position where the element of ui doesn't come from the parent
    jrand = sub2ind([popsize Problem.D], rows, cols); mask(jrand) = false;
    u     = vi; u(mask) = pop1(mask);

    offspring = Problem.Evaluation(u);
end
```

### `First_Stage_EnvironmentalSelection.m`
```matlab
function [Population,succ1,succ2,Fitness] = First_Stage_EnvironmentalSelection(Population,N,isOrigin)
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
    if isOrigin == 1
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

    %% calculate success rate1
    off_index = Next(1+N:2*N);
    succ1 = zeros(1,length(off_index));
    for j = 1 : length(off_index)
        if off_index(j) == 1
            succ1(j)=1;
        end
    end

    %% calculate success rate2
    off_index = Next(1+2*N:end);
    succ2     = zeros(1,length(off_index));
    for j = 1 : length(off_index)
        if off_index(j) == 1
            succ2(j) = 1;
        end
    end
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

### `GA_TournamentSelection.m`
```matlab
function [ offspring ] = GA_TournamentSelection(Problem, Population, Fitness, popsize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    MatingPool1 = TournamentSelection(2,2*popsize,Fitness);
    offspring   = OperatorGAhalf(Problem,Population(MatingPool1));
end
```

### `Second_Stage_EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = Second_Stage_EnvironmentalSelection(Population,N,isOrigin)
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
    if isOrigin == 1
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

### `URCMO.m`
```matlab
classdef URCMO < ALGORITHM
% <2023> <multi> <real/integer> <constrained>
% Utilizing the relationship between constrained and unconstrained Pareto fronts for constrained multi-objective optimization

%------------------------------- Reference --------------------------------
% J. Liang, K. Qiao, K. Yu, B. Qu, C. Yue, W. Guo, and L. Wang. Utilizing
% the relationship between unconstrained and constrained Pareto fronts for
% constrained multi-objective optimization. IEEE Transactions on
% Cybernetics, 2023, 53(6): 3873-3886.
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
            %% Generate random population
            warning off
            Population{1} = Problem.Initialization();
            Population{2} = Problem.Initialization();
            
            Fitness{1} = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{2} = CalFitness(Population{2}.objs);
            
            %% Parameter settings
            cnt       = 0;
            p         = 0.1;
            first_FES = 10000;
            beita     = 0.9;

            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                cnt = cnt + 1;
                %% learning phase
                if Problem.FE < first_FES
                    for i = 1 : 2
                        valOffspring{i}(1:Problem.N/2)             = GA_TournamentSelection(Problem, Population{i}, Fitness{i}, Problem.N/2);
                        valOffspring{i}(1+Problem.N/2 : Problem.N) = DE_current_to_rand_1(Problem, Population{i}, Problem.N/2);
                    end
                    
                    % Environmental selection
                    [Population{1},succ1,succ2,Fitness{1}] = First_Stage_EnvironmentalSelection([Population{1},valOffspring{1},valOffspring{2}],Problem.N,1);
                    [Population{2},~,~,Fitness{2}]         = First_Stage_EnvironmentalSelection([Population{2},valOffspring{2},valOffspring{1}],Problem.N,2);
                    
                    succ_jilu(cnt,1) = sum(succ1(1:Problem.N/2));
                    succ_jilu(cnt,2) = sum(succ1(1 + Problem.N/2 : Problem.N));
                    
                    succ_jilu(cnt,3) = sum(succ2(1:Problem.N/2));
                    succ_jilu(cnt,4) = sum(succ2(1 + Problem.N/2 : Problem.N)); % only succ2 is used in the paper
                    
                    if Problem.FE >= first_FES
                        for num = 1 : 4
                            if std(succ_jilu(:,num)) ~= 0
                                a(num) = mean(succ_jilu(:,num)) / std(succ_jilu(:,num));
                            else
                                a(num) = 0;
                            end
                        end
                        
                        [flag,ll] = Classification(Population{1},Population{2},beita);  % flag correspondings to the three kinds of evolving strategies
                        
                        if flag == 1
                            if a(3) < a(4)
                                flag = 3;
                            end
                        end
                        
                    end
                    
                % Evolving phase
                else
                    valOffspring{1}(1:Problem.N/2)             = GA_TournamentSelection(Problem, Population{1}, Fitness{1}, Problem.N/2);
                    valOffspring{1}(1+Problem.N/2 : Problem.N) = DE_current_to_rand_1(Problem, Population{1}, Problem.N/2);
                    
                    if flag == 1
                        valOffspring{2}(1:Problem.N/2)              = GA_TournamentSelection(Problem, Population{2}, Fitness{2}, Problem.N/2);
                        valOffspring{2}( 1+Problem.N/2 : Problem.N) = DE_transfer(Problem, Population{2}, Population{1}, Problem.N/2);
                    elseif flag == 2
                        num      = ceil(Problem.N/3);
                        rand_num = rand;
                        if rand_num <= 1/3
                            valOffspring{2}(1:num)               = GA_TournamentSelection(Problem, Population{2}, Fitness{2}, num);
                            valOffspring{2}(1+num : 2*num)       = DE_transfer(Problem, Population{2}, Population{1}, num);
                            valOffspring{2}(1+2*num : Problem.N) = DE_current_to_other_pbest_1(Problem,Population{2},Problem.N-2*num,Fitness{1},Population{1},p);
                        elseif rand_num <=2/3
                            valOffspring{2}(1:num)               = GA_TournamentSelection(Problem, Population{2}, Fitness{2}, num);
                            valOffspring{2}(1+num : 2*num)       = DE_current_to_other_pbest_1(Problem,Population{2},num,Fitness{1},Population{1},p);
                            valOffspring{2}(1+2*num : Problem.N) = DE_transfer(Problem, Population{2}, Population{1}, Problem.N-2*num);
                        elseif rand_num <= 1
                            valOffspring{2}(1:num)               = DE_current_to_other_pbest_1(Problem,Population{2},num,Fitness{1},Population{1},p);
                            valOffspring{2}(1+num : 2*num)       = DE_transfer(Problem, Population{2}, Population{1}, num);
                            valOffspring{2}(1+2*num : Problem.N) = GA_TournamentSelection(Problem, Population{2}, Fitness{2}, Problem.N-2*num);
                        end
                    elseif flag == 3
                        valOffspring{2} = DE_current_to_other_pbest_1(Problem,Population{2},Problem.N,Fitness{1},Population{1},p);
                    end
                    
                    % Environmental selection
                    [Population{1},Fitness{1}] = Second_Stage_EnvironmentalSelection([Population{1},valOffspring{1},valOffspring{2}],Problem.N,1);
                    [Population{2},Fitness{2}] = Second_Stage_EnvironmentalSelection([Population{2},valOffspring{2},valOffspring{1}],Problem.N,2);
                end
            end
        end
    end
end
```

### `boundConstraint.m`
```matlab
function vi = boundConstraint (vi, pop, lu)
% Set the value to be the middle if the boundary constraint is violated

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    [NP, D] = size(pop);  % the population size and the problem's dimension

    %% check the lower bound
    xl      = repmat(lu(1, :), NP, 1);
    pos     = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    %% check the upper bound
    xu      = repmat(lu(2, :), NP, 1);
    pos     = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
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

% This function is written by Kangjia Qiao

    NP0 = length(r0);

    r1  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : inf
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else   % regenerate r1 if it is equal to r0
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000     % this has never happened so far
            error('Can not genrate r1 in 1000 iterations');
        end
    end

    r2 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : inf
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else   % regenerate r2 if it is equal to r0 or r1
            r2(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000     % this has never happened so far
            error('Can not genrate r2 in 1000 iterations');
        end
    end

    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : inf
        pos = ((r3 == r1) | (r3 == r0) | (r3 == r2));
        if sum(pos) == 0
            break;
        else   % regenerate r2 if it is equal to r0 or r1
            r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000     % this has never happened so far
            error('Can not genrate r3 in 1000 iterations');
        end
    end
end
```
