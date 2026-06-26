# PRCEA

**Tags**: <2026> <multi/many> <real/integer> <large/none> <constrained/none>

## Description
Promising region-guided large-scale constrained MOEA

## Reference
X. Zhong, X. Yao, K. Qiao, D. Gong, and Y. Jin. A large-scale constrained multi-objective evolutionary algorithm with promising region detection and diversity enhancement. IEEE Transactions on Evolutionary Computation, 2026.

## Source Code

### `AES.m`
```matlab
function Offspring = AES(Problem, Population, Fitness, Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
   
    N = length(Population);
    
    %% Mating selection
    MatingPool = TournamentSelection(2,N,Fitness);
    Population = Population(MatingPool);
    Fitness    = Fitness(MatingPool);
    
    %% Clustering
    [~,indBest] = sort(Fitness, 'ascend');
    Se          = Population(indBest(1:ceil(N/2)));     % Winner
    Sp          = Population(indBest(1+ceil(N/2):end)); % Loser

    % Perform bidirectional sampling on each solution in Se to generate new solutions.
    Off1 = DirectedSampling(Problem,Se);

    % For each solution in Sp, find the paired solution in Se that forms the minimum angle with it.
    [Mate1,Mate2] = Pair(Se,Sp,Zmin);
    Off2          = DE_best_1(Problem,Mate1.decs,Mate2.decs);

    Offspring = [Off1,Off2];
end

function [Mate1,Mate2] = Pair(Se,Sp,Zmin)

    %% Normalization
    SeObj   = Se.objs;
    [Num,M] = size(SeObj);
    SeObj   = (SeObj - repmat(Zmin,Num,1))./repmat(sqrt(sum(SeObj.^2,2)),1,M);
    SpObj   = Sp.objs;
    SpObj   = (SpObj - repmat(Zmin,Num,1))./repmat(sqrt(sum(SpObj.^2,2)),1,M);
    
    %% Association
    Cosine        = 1-pdist2(SpObj,SeObj,'cosine');
    [~,associate] = max(Cosine,[],2);
    Mate1         = Sp;
    Mate2         = Se(associate);
end

function Offspring = DE_best_1(Problem,Parent1,Parent2)
    [N,D] = size(Parent1);

    %% Differental evolution
    Intervalmin = sqrt(sum((Parent2-Parent1).^2,2));
    Intervalmax = sqrt(sum((Problem.upper-Problem.lower).^2,2));
    sigma       = Intervalmin + (Intervalmax-Intervalmin)*(1-Problem.FE/Problem.maxFE)^2;
    F           = rand(N,1).*sigma; F = F(:, ones(1,D));
    Offspring   = Parent1;
    Offspring   = Offspring + F.*(Parent2-Offspring)./Intervalmin;

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
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
    Offspring = Problem.Evaluation(Offspring);
end

function SampleSolution= DirectedSampling(Problem,Population)
    Nw     = length(Population);
    PopDec = Population.decs;
    Upper  = Problem.upper;
    Lower  = Problem.lower;

    Directnorm = [sqrt(sum((PopDec - repmat(Lower,Nw,1)).^2,2));sqrt(sum((PopDec - repmat(Upper,Nw,1)).^2,2))];
    Direction  = [PopDec - repmat(Lower,Nw,1);PopDec - repmat(Upper,Nw,1)]./repmat(Directnorm,1,Problem.D);

    %% Generate guiding solutions
    Intervalmax    = sqrt(sum((Upper-Lower).^2,2))*(1-Problem.FE/Problem.maxFE).^2;
    Intervalmin    = 0;
    RandSample     = Intervalmin + rand(1,Nw*4)*(Intervalmax-Intervalmin);
    SampleSolution = GenerateSampleSolution(Problem,PopDec,RandSample,Direction);

end

function SampleSolution = GenerateSampleSolution(Problem,X,RandSample,Direct)
% Generate some sample solutions along with the guiding directions

    Nw   = length(RandSample)/4;
    PopX = [X + repmat(RandSample(1:Nw)',1,Problem.D).* Direct(1:Nw,:);...
        X - repmat(RandSample(1+Nw:Nw*2)',1,Problem.D).* Direct(1:Nw,:);...
        X + repmat(RandSample(Nw*2+1:Nw*3)',1,Problem.D).* Direct(Nw+1:end,:);...
        X - repmat(RandSample(Nw*3+1:end)',1,Problem.D).* Direct(Nw+1:end,:)];
    PopX = max(min(repmat(Problem.upper,size(PopX,1),1),PopX),repmat(Problem.lower,size(PopX,1),1));
    SampleSolution = Problem.Evaluation(PopX);
end
```

### `CDP_EPD_Update.m`
```matlab
function [Population,Fitness] = CDP_EPD_Update(Population,N,isOrigin)

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
        % CDP-based
        Fitness = CalFitness(Population.objs,Population.cons);
    else
        % EPD-based
        cons = Population.cons;
        cons(cons<0) = 0;
        cons    = sum(cons,2);
        Fitness = CalFitness([Population.objs,cons]);
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
    elseif nargin == 3
        CV = sum(max(0,PopCon),2);
        CV(CV < epsilon) = 0;
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

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

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
    if class(Problem)=="my_t_problem2_1"
         Offspring = feixianxing_repair_infsolution(Offspring);
    end
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `OperatorDE_pbest_1_main.m`
```matlab
function [ Offspring ] = OperatorDE_pbest_1_main(Problem, Population, popsize, Fitness, p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N           = length(Population);
    permutation = randperm(N);
    r0          = permutation;
    [r1,r2,r3]  = gnR1R2R3(N, r0);

    array = permutation(1:popsize);

    [~, indBest] = sort(Fitness, 'ascend');
    pNP          = max(round(p * N), 2);            % choose at least two best solutions
    randindex    = ceil(rand(1, popsize) * pNP);    % select from [1, 2, 3, ..., pNP]
    randindex    = max(1, randindex);               % to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest        = Population(indBest(randindex));  % randomly choose one of the top 100p% solutions

    Offspring = OperatorDE_pbest_1(Problem,Population(array),pbest,Population(r1(1:popsize)),Population(r2(1:popsize)));
end
```

### `OperatorDE_rand_1.m`
```matlab
function Offspring = OperatorDE_rand_1(Problem,Parent1,Parent2,Parent3)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [proM,disM] = deal(1,20);
    if isa(Parent1(1),'SOLUTION')
        evaluated = true;
        Parent1   = Parent1.decs;
        Parent2   = Parent2.decs;
        Parent3   = Parent3.decs;
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
    if class(Problem)=="my_t_problem2_1"
         Offspring = feixianxing_repair_infsolution(Offspring);
    end
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `PRCEA.m`
```matlab
classdef PRCEA < ALGORITHM
% <2026> <multi/many> <real/integer> <large/none> <constrained/none>
% Promising region-guided large-scale constrained MOEA

%------------------------------- Reference --------------------------------
% X. Zhong, X. Yao, K. Qiao, D. Gong, and Y. Jin. A large-scale constrained
% multi-objective evolutionary algorithm with promising region detection
% and diversity enhancement. IEEE Transactions on Evolutionary Computation,
% 2026.
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
            Pop{1} = Problem.Initialization();
            Pop{2} = Problem.Initialization();
            Off    = {[],[]};
            stage            = 1;
            lgap             = 20;
            change_threshold = 1e-3;
            NP               = Problem.N;
            cnt              = 0;
    
            %% Optimization
            while Algorithm.NotTerminated(Pop{1})
                cnt = cnt + 1;
                % Population Update
                [Pop{1},Fit{1}] = CDP_EPD_Update([Pop{1}, Off{1},Off{2}],NP, true);
                if stage == 1
                    [Pop{2},Fit{2}] = CDP_EPD_Update([Pop{2}, Off{1},Off{2}],NP,false);
                else
                    [Pop{2},Fit{2}] = PRDD_Update(Pop{1},[Pop{2}, Off{1},Off{2}],NP);
                end
                % Udate ideal point
                for i = 1 : 2
                    zmin{i} = min([Pop{i}.objs], [], 1) - 1e-6;
                end
                % Update stage
                if stage == 1
                    Objs = Pop{1}.objs;
                    ideal_points(cnt,:) = min(Objs, [], 1);
                    nadir_points(cnt,:) = max(Objs, [], 1);
                    mean_points(cnt,:)  = mean(Objs, 1);
                    if cnt >= lgap
                        max_change = calc_maxchange(ideal_points, nadir_points, mean_points, cnt, lgap);
                        if max_change <= change_threshold
                            stage = 2;
                        end
                    end
                end
                % Offspring generation
                Off = cell(1, 2);
                for i = 1 : 2
                    if stage == 1
                        Off{i} = AES(Problem, Pop{i}, Fit{i}, zmin{i});
                    else
                        MatingPool          = [Pop{i}(randsample(NP,NP))];
                        [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPool,zmin{i});
                        Off{i}(1:NP/2)      = OperatorDE_rand_1(Problem,Mate1(1:NP/2), Mate2(1:NP/2), Mate3(1:NP/2));
                        Off{i}(1+NP/2:NP)   = OperatorDE_pbest_1_main(Problem, Pop{i}, NP/2, Fit{i}, 0.1);
                    end
                end
                if Problem.FE >= Problem.maxFE
                    [Pop{1},~] = CDP_EPD_Update([Pop{1}, Off{1},Off{2}],Problem.N, true);
                end
            end
        end
    end
end
```

### `PRDD_Update.m`
```matlab
function [return_pop,return_Fitness] = PRDD_Update(mainPop,Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    return_pop     = [];
    return_Fitness = [];
    maxfit         = 0;
    input_cons     = mainPop.cons;
    input_cons(input_cons<0) = 0;
    input_cons = sum(input_cons,2);

    S1 = [];
    S2 = [];
    S3 = [];
    if sum(input_cons<=0) == 0
        S1 = [S1,Population];
    else
        FP = mainPop(input_cons<=0);
        [FrontNo,~] = NDSort(FP.objs,1);
        FP = FP(FrontNo ==1);
        for i = 1 : length(Population)
            Diff = sign(Population(i).objs-FP.objs);
            if sum(max(Diff,[],2)-min(Diff,[],2)==2)==length(FP)
                S1 =[S1,Population(i)];
            elseif any(sum(Diff<=0,2)==size(FP.objs,2))
                S2 = [S2,Population(i)];
            elseif any(sum(Diff>=0,2)==size(FP.objs,2))
                S3 = [S3,Population(i)];
            end
        end
    end

    if length(S1)>=N
        %EPD for S1
        [return_pop,return_Fitness] = CDP_EPD_Update(S1,N,false);
    else
        if ~isempty(S1)
            return_pop     = S1;
            return_Fitness = CalFitness([S1.objs,sum(max(S1.cons,0),2)]);
            maxfit         = max(return_Fitness);
        end
        %ERPD for S2
        if length(S2) >= N-length(S1)
            Fitness = CalFitness([-S2.objs,sum(max(S2.cons,0),2)]);
            Next    = Fitness < 1;
            if sum(Next) <= N - length(S1)
                [~,Rank] = sort(Fitness);
                Next(Rank(1:N - length(S1) )) = true;
            elseif sum(Next) > N - length(S1)
                Del  = Truncation(S2(Next).objs, sum(Next)-(N-length(S1)));
                Temp = find(Next);
                Next(Temp(Del)) = false;
            end
            S2             = S2(Next);
            Fitness        = Fitness(Next) + maxfit;      
            return_pop     = [return_pop,S2];
            return_Fitness = [return_Fitness,Fitness];
        %PD for S3
        else
            if ~isempty(S2)
                return_pop     = [return_pop,S2];
                Fitness        = CalFitness([-S2.objs,sum(max(S2.cons,0),2)])+maxfit;
                return_Fitness = [return_Fitness,Fitness];
                maxfit         = max(return_Fitness);
            end

            Fitness = CalFitness(S3.objs); 
            Next    = Fitness < 1;
            if sum(Next) <= N - length(return_pop)
                [~,Rank] = sort(Fitness);
                Next(Rank(1:N - length(return_pop) )) = true;
            elseif sum(Next) > N - length(return_pop)
                Del  = Truncation(S3(Next).objs, sum(Next)-(N-length(return_pop)));
                Temp = find(Next);
                Next(Temp(Del)) = false;
            end
            S3             = S3(Next);
            Fitness        = Fitness(Next) + maxfit;      
            return_pop     = [return_pop,S3];
            return_Fitness = [return_Fitness,Fitness];
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

### `calc_maxchange.m`
```matlab
function max_change = calc_maxchange(ideal_points,nadir_points,mean_points,gen,last_gen)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    delta_value = 1e-6 * ones(1,size(ideal_points,2));
    idx         = gen - last_gen + 1;
    rz          = abs((ideal_points(gen,:) - ideal_points(idx,:)) ./ max(abs(ideal_points(idx,:)),delta_value));
    nrz         = abs((nadir_points(gen,:) - nadir_points(idx,:)) ./ max(abs(nadir_points(idx,:)),delta_value));
    mrz         = abs((mean_points(gen,:)  - mean_points(idx,:))  ./ max(abs(mean_points(idx,:)),delta_value));
    max_change  = max([rz, nrz, mrz]);
end
```
