# DBEMTO

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Double-balanced evolutionary multi-task optimization

## Reference
K. Qiao, J. Liang, K. Yu, M. Wang, B. Qu, C. Yue, and Y. Gou. A self-adaptive evolutionary multi-task based constrained multi-objective evolutionary algorithm. IEEE Transactions on Emerging Topics in Computational Intelligence, 2023, 7(4): 1098-1112.

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
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `DBEMTO.m`
```matlab
classdef DBEMTO < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <constrained>
% Double-balanced evolutionary multi-task optimization

%------------------------------- Reference --------------------------------
% K. Qiao, J. Liang, K. Yu, M. Wang, B. Qu, C. Yue, and Y. Gou. A
% self-adaptive evolutionary multi-task based constrained multi-objective
% evolutionary algorithm. IEEE Transactions on Emerging Topics in
% Computational Intelligence, 2023, 7(4): 1098-1112.
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
            Population{1} = Problem.Initialization();
            Population{2} = Problem.Initialization();

            Fm  = [0.6,0.8,1.0];
            CRm = [0.1,0.2,1.0];
            lu  = [Problem.lower;Problem.upper];
            strategy_num         = 3;
            arrayGbestChange     = ones(2,strategy_num);
            arrayGbestChangeRate = zeros(2,strategy_num);
            indexBestLN(1:2)     = 2;
            cnt = 0;
            leastSelectionPro = 0.05;
            mixPopSize        = Problem.N;
            M = Problem.M;
            numViaLN    = zeros(2,strategy_num);
            consumedFES = zeros(2,strategy_num);

            Fitness{1} = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{2} = CalFitness(Population{2}.objs);

            k = 2;

            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                cnt = cnt +1;
                for m = 1 :2
                    pop{m,1} = Population{1,m}.decs;
                    pop{m,2}(:,1:M) = Population{1,m}.objs;
                end
                for i = 1 : k
                    vi = 0;
                    ui = 0;
                    arrayGbestChangeRate(i,1) = arrayGbestChange(i,1);
                    arrayGbestChangeRate(i,2) = arrayGbestChange(i,2);
                    arrayGbestChangeRate(i,3) = arrayGbestChange(i,3);

                    [~,indexBestLN(i)] = max(arrayGbestChangeRate(i,:));
                    if sum(arrayGbestChangeRate(i,:) == arrayGbestChangeRate(i,1)) == strategy_num
                        indexBestLN(i) = 1;
                    end
                    arrayGbestChange(i,:)     = 0.1 * ones(1,strategy_num);
                    arrayGbestChangeRate(i,:) = zeros(1,strategy_num);
                    consumedFES(i,:)          = ones(1,strategy_num);

                    permutation = randperm(mixPopSize);
                    if indexBestLN(i) == 1
                        arrayThird{i}  = permutation(1:ceil(leastSelectionPro*mixPopSize));
                        arraySecond{i} = permutation(ceil(leastSelectionPro*mixPopSize+1): ceil(2*leastSelectionPro*mixPopSize));
                        arrayFirst{i}  = permutation(ceil(2*leastSelectionPro*mixPopSize+1):end);
                        numViaLN(i,1)  = numViaLN(i,1) + 1;
                    elseif indexBestLN(i) == 2
                        arrayThird{i}  = permutation(1:ceil(leastSelectionPro*mixPopSize));
                        arrayFirst{i}  = permutation(ceil(leastSelectionPro*mixPopSize+1): ceil(2*leastSelectionPro*mixPopSize));
                        arraySecond{i} = permutation(ceil(2*leastSelectionPro*mixPopSize+1):end);
                        numViaLN(i,2)  = numViaLN(i,2) + 1;
                    elseif indexBestLN(i) == 3
                        arrayFirst{i}  = permutation(1:ceil(leastSelectionPro*mixPopSize));
                        arraySecond{i} = permutation(ceil(leastSelectionPro*mixPopSize+1): ceil(2*leastSelectionPro*mixPopSize));
                        arrayThird{i}  = permutation(ceil(2*leastSelectionPro*mixPopSize+1):end);
                        numViaLN(i,3)  = numViaLN(i,3) + 1;
                    end

                    rateViaLN{i}(cnt,:) = numViaLN(i,:)/sum(numViaLN(i,:));
                    consumedFES(i,:)    = consumedFES(i,:) + [length(arrayFirst{i}),length(arraySecond{i}),length(arrayThird{i})];
                    r0 = permutation;

                    %% =================Intra-task convergence strategy=================================
                    if ~isempty(arrayFirst{i})
                        if i == 1
                            MatingPool1 = TournamentSelection(2,2*length(arrayFirst{i}),Fitness{i});
                        else
                            MatingPool1 = TournamentSelection(2,2*length(arrayFirst{i}),Fitness{i});
                        end
                        valOffspring{i}(1,arrayFirst{i}) = OperatorGAhalf(Problem,Population{i}(MatingPool1));
                    end
                    %%  ====================Intra-task diversity strategy===========================
                    if ~isempty(arraySecond{i})
                        [r1,r2,r3] = gnR1R2R3(mixPopSize,  r0);
                        pop2       = pop{i,1}(arraySecond{i},:);
                        popsize2   = length(arraySecond{i});

                        index = randi([1,length(Fm)],popsize2,1);
                        F2    = Fm(index);
                        F2    = F2';
                        index = randi([1,length(CRm)],popsize2,1);
                        CR2   = CRm(index);
                        CR2   = CR2';

                        vi = pop{i,1}(r1(arraySecond{i}),:)  + F2(:, ones(1, Problem.D)) .* (pop{i,1}(r2(arraySecond{i}),:) - pop{i,1}(r3(arraySecond{i}),:));
                        vi = boundConstraint(vi, pop2, lu);

                        mask        = rand(popsize2, Problem.D) > CR2(:, ones(1, Problem.D));
                        rows        = (1 : popsize2)';
                        cols        = floor(rand(popsize2, 1) * Problem.D)+1;
                        jrand       = sub2ind([popsize2 Problem.D], rows, cols);
                        mask(jrand) = false;
                        u2          = vi;
                        u2(mask)    = pop2(mask);
                        valOffspring{i}(1,arraySecond{i}) = Problem.Evaluation(u2);
                    end
                    %% ====================Inter-task transfer strategy====================================
                    if ~isempty(arrayThird{i})
                        rand_perm   = randperm(mixPopSize);
                        MatingPool1 = rand_perm(1:length(arrayThird{i}));
                        valOffspring{i}(1,arrayThird{i}) = Population{2/i}(MatingPool1);
                    end
                end
                %% ====================Environmental selection====================================
                for i = 1 : k
                    if indexBestLN(i) ~= 3
                        if i == 1
                            [Population{i},~,Fitness{1}]    = EnvironmentalSelection([Population{i},valOffspring{2/i}],mixPopSize,i);
                            [Population{i},succ,Fitness{1}] = EnvironmentalSelection([Population{i},valOffspring{i}],mixPopSize,i);
                        elseif i == 2
                            [Population{i},~,Fitness{2}]    = EnvironmentalSelection([Population{i},valOffspring{2/i}],mixPopSize,i);
                            [Population{i},succ,Fitness{2}] = EnvironmentalSelection([Population{i},valOffspring{i}],mixPopSize,i);
                        end
                    else
                        if i == 1
                            [Population{i},succ,Fitness{1}] = EnvironmentalSelection([Population{i},valOffspring{i}],mixPopSize,i);
                        elseif i == 2
                            [Population{i},succ,Fitness{2}] = EnvironmentalSelection([Population{i},valOffspring{i}],mixPopSize,i);
                        end
                    end
                    arrayGbestChange(i,1) = arrayGbestChange(i,1) + sum(succ(arrayFirst{i}))/length(arrayFirst{i});
                    arrayGbestChange(i,2) = arrayGbestChange(i,2) + sum(succ(arraySecond{i}))/length(arraySecond{i});
                    arrayGbestChange(i,3) = arrayGbestChange(i,3) + sum(succ(arrayThird{i}))/length(arrayThird{i});
                end
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,succ,Fitness] = EnvironmentalSelection(Population,N,isOrigin)
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
    if isOrigin==1
        con = Population.cons;
        Fitness = CalFitness(Population.objs,con);
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
    %% calculate success rate
    parent_index = Next(1:N);
    off_index = Next(1+N:2*N);
    succ = zeros(1,length(off_index));
    for j = 1:length(off_index)
        if off_index(j) == 1
            succ(j)=1;
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

### `boundConstraint.m`
```matlab
function vi = boundConstraint (vi, pop, lu)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    NP  = size(pop,1);  
    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
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
