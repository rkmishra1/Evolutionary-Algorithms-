# IMTCMO_BS

**Tags**: <2024> <multi/many> <real/integer/label/binary/permutation> <large> <constrained/none>

## Description
Improved evolutionary multitasking-based CMOEA with bidirectional sampling

## Reference
K. Qiao, J. Liang, K. Yu, W. Guo, C. Yue, B. Qu, and P. N. Suganthan. Benchmark problems for large-scale constrained multi-objective optimization with baseline results. Swarm and Evolutionary Computation, 2024, 86: 101504.

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

    input_cons = Population.cons;
    input_cons(input_cons<0) = 0;
    input_cons = sum(input_cons,2);

    findex  = find(input_cons<=VAR);
    ifindex = find(input_cons>VAR);

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
        % Sort the population
        [ifFitness,rank] = sort(ifFitness);
        ifPopulation     = ifPopulation(rank);

        fPopulation = [];
        fFitness    = [];

    elseif length(fPopulation) <= N
        cons         = fPopulation.cons;
        cons(cons<0) = 0;
        cons         = sum(cons,2);
        fFitness     = CalFitness([fPopulation.objs,cons]);
        Next         = fFitness < 1;

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
        ifPopulation     = ifPopulation(rank);

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

### `Convergence_DirectedSampling.m`
```matlab
function [GuidingSolution,SampleSolution ]= Convergence_DirectedSampling(Global,Population,Ns,Nw,RefV,VAR)
% Acquiring Guiding Solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Classter the reference vectors
    BoundRefV               = eye(Global.M,Global.M);
    BoundRefV(BoundRefV==0) = 10e-7;
    [~,CenterRefV,~,~]      = kmeans(RefV,Nw);
    DirectRefV              = [BoundRefV;CenterRefV];
    Nw                      = size(DirectRefV,1);

    %% Identify guiding directions
    Best       = GenerateRepresetativeSolution(Population.objs,DirectRefV);
    PopDec     = Population.decs;
    BestX      = PopDec(Best,:);
    Upper      = Global.upper;
    Lower      = Global.lower;
    Directnorm = [sqrt(sum((BestX - repmat(Lower,Nw,1)).^2,2));sqrt(sum((BestX - repmat(Upper,Nw,1)).^2,2))];
    Direction  = [BestX - repmat(Lower,Nw,1);BestX - repmat(Upper,Nw,1)]./repmat(Directnorm,1,Global.D);

    %% Generate guiding solutions
    Intervalmax    = sqrt(sum((Upper-Lower).^2,2));
    Intervalmin    = 0;
    Nw             = 2*Nw;
    RandSample     = Intervalmin + rand(Ns,Nw)*(Intervalmax-Intervalmin);
    SampleSolution = GenerateSampleSolution(Global,RandSample,Direction);

    cons = sum(max(SampleSolution.cons,0),2);
    cons(cons<VAR) = 0;

    GuidingSolution = SampleSolution((NDSort(SampleSolution.objs,cons,1)==1));
    end

    function Best = GenerateRepresetativeSolution(Obj,RefV)
    % Find out respective solutions

    %% Normalization
    np   = size(Obj,1);
    Obj  = (Obj-repmat(min(Obj),np,1))./(repmat(max(Obj),np,1)-repmat(min(Obj),np,1));
    Nr   = size(RefV,1);
    Best = zeros(Nr,1);

    %% Assign individuals for each reference vector
    Cosine        = 1-pdist2(Obj,RefV,'cosine');
    [~,associate] = max(Cosine,[],2);

    Indflag = zeros(np,1);
    current = cell(Nr,1);
    for i = 1 : Nr
        current{i,1} = find(associate == i);
    end
    for i = 1 : Nr
        if length(current{i,1}) > 1
            rand_index = ceil(rand*length(current{i,1}));
            Best(i,1) = current{i,1}(rand_index);
            Indflag(current{i,1}(rand_index),1) = 1;
        elseif length(current{i,1}) == 1
            Best(i,1) = current{i,1}(1);
            Indflag(current{i,1}(1),1) = 1;
        end
    end

    for i = 1 : Nr
        if isempty(current{i,1})
            [~,indCon] = sort(Cosine(:,i),'descend');
            k = 1;
            if length(indCon) > Nr
                while Indflag(indCon(k),1) == 1
                    k = k + 1;
                end
                Best(i,1) = indCon(k);
                Indflag(indCon(k),1) = 1;
            else
                Best(i,1) = indCon(1);
            end
        end
    end
end

function SampleSolution = GenerateSampleSolution(Global,RandSample,Direct)
% Generate some sample solutions along with the guiding directions

    [Ns,Nw]        = size(RandSample);
    Nw             = Nw/2;
    SampleSolution = [];
    for i = 1 : Ns
        PopX = [repmat(Global.lower,Nw,1) + repmat(RandSample(i,1:Nw)',1,Global.D).* Direct(1:Nw,:);...
            repmat(Global.upper,Nw,1) + repmat(RandSample(i,Nw+1:end)',1,Global.D).* Direct(Nw+1:end,:)];

        PopX = max(min(repmat(Global.upper,size(PopX,1),1),PopX),repmat(Global.lower,size(PopX,1),1));
        SampleSolutiontemp = Global.Evaluation(PopX);
        SampleSolution     = [SampleSolution,SampleSolutiontemp];
    end
end
```

### `Diversity_DirectedSampling.m`
```matlab
function [GuidingSolution,SampleSolution ]= Diversity_DirectedSampling(Global,Population,Ns,Nw,RefV,VAR)
% Acquiring Guiding Solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Classter the reference vectors
    BoundRefV               = eye(Global.M,Global.M);
    BoundRefV(BoundRefV==0) = 10e-7;
    [~,CenterRefV,~,~]      = kmeans(RefV,Nw);
    DirectRefV              = [BoundRefV;CenterRefV];
    Nw                      = size(DirectRefV,1);

    %% Identify guiding directions
    Best   = GenerateRepresetativeSolution(Population.objs,DirectRefV);
    PopDec = Population.decs;
    Upper  = Global.upper;
    Lower  = Global.lower;
    if size(Best,1)==1
        start_X = repmat(PopDec(Best,:),2,1);
        end_X   = [Lower;Upper];
    elseif mod(size(Best,1),2)==1
        BestX   = PopDec(Best(randperm(size(Best,1)-1)),:);
        start_X = BestX(1:size(BestX,1)/2,:);
        end_X   =  BestX(1+size(BestX,1)/2:end,:);
    else
        BestX   = PopDec(Best,:);
        start_X = BestX(1:size(BestX,1)/2,:);
        end_X   =  BestX(1+size(BestX,1)/2:end,:);
    end
    Nw = size(start_X,1);

    Directnorm = [sqrt(sum((start_X - end_X).^2,2))];
    Direction  = [start_X - end_X]./repmat(Directnorm,1,Global.D);

    %% Generate guiding solutions
    Intervalmax    = sqrt(sum((Upper-Lower).^2,2));
    Intervalmin    = 0;
    RandSample     = Intervalmin + rand(Ns,Nw)*(Intervalmax-Intervalmin);
    SampleSolution = GenerateSampleSolution(Global,RandSample,Direction,end_X);

    cons = sum(max(SampleSolution.cons,0),2);
    cons(cons<VAR) = 0;

    GuidingSolution = SampleSolution((NDSort(SampleSolution.objs,cons,1)==1));
end

function Best = GenerateRepresetativeSolution(Obj,RefV)
% Find out respective solutions

    %% Normalization
    np   = size(Obj,1);
    Obj  = (Obj-repmat(min(Obj),np,1))./(repmat(max(Obj),np,1)-repmat(min(Obj),np,1));
    Nr   = size(RefV,1);
    Best = zeros(Nr,1);

    %% Assign individuals for each reference vector
    Cosine        = 1-pdist2(Obj,RefV,'cosine'); % pdist()返回的是cos(sita)的数字值
    [~,associate] = max(Cosine,[],2);

    Indflag = zeros(np,1);
    current = cell(Nr,1);
    for i = 1 : Nr
        current{i,1} = find(associate == i);
    end
    for i = 1 : Nr
        if length(current{i,1})>1
            rand_index = ceil(rand*length(current{i,1}));
            Best(i,1)  = current{i,1}(rand_index);
            Indflag(current{i,1}(rand_index),1) = 1;
        elseif length(current{i,1})==1
            Best(i,1) = current{i,1}(1);
            Indflag(current{i,1}(1),1) = 1;
        end
    end
    for i = 1 : Nr
        if isempty(current{i,1})
            [~,indCon] = sort(Cosine(:,i),'descend');
            k = 1;
            if length(indCon) > Nr
                while Indflag(indCon(k),1) == 1
                    k = k + 1;
                end
                Best(i,1) = indCon(k);
                Indflag(indCon(k),1) = 1;
            else
                Best(i,1) = indCon(1);
            end
        end
    end
end

function SampleSolution = GenerateSampleSolution(Global,RandSample,Direct,end_X)
% Generate some sample solutions along with the guiding directions

    [Ns,Nw] = size(RandSample);
    SampleSolution = [];
    for i = 1 : Ns
        PopX = end_X + repmat(RandSample(i,1:Nw)',1,Global.D).* Direct(1:Nw,:);
        PopX = max(min(repmat(Global.upper,size(PopX,1),1),PopX),repmat(Global.lower,size(PopX,1),1));
        SampleSolutiontemp = Global.Evaluation(PopX);
        SampleSolution     = [SampleSolution,SampleSolutiontemp];
    end
end
```

### `IMTCMO_BS.m`
```matlab
classdef  IMTCMO_BS < ALGORITHM
% <2024> <multi/many> <real/integer/label/binary/permutation> <large> <constrained/none>
% Improved evolutionary multitasking-based CMOEA with bidirectional sampling

%------------------------------- Reference --------------------------------
% K. Qiao, J. Liang, K. Yu, W. Guo, C. Yue, B. Qu, and P. N. Suganthan.
% Benchmark problems for large-scale constrained multi-objective
% optimization with baseline results. Swarm and Evolutionary Computation,
% 2024, 86: 101504.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao
% If you have any question, please email qiaokangjia@yeah.net

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            [Nw,Ns,g] = Algorithm.ParameterSet(10,30,50);
            
            %% Initialization
            Population1 = Problem.Initialization();
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            Zmin1       = min(Population1.objs,[],1);
            
            Population2 = Problem.Initialization();
            Fitness2    = CalFitness(Population2.objs,Population2.cons);
            Zmin2       = min(Population2.objs,[],1);
            
            
            [V0,~]    = UniformPoint(Problem.N,Problem.M);
            [Vs0,L]   = UniformPoint(floor(Problem.N/5),Problem.M);
            [RefV,Vs] = deal(V0,Vs0);
            
            X    = 0;
            cons = [Population1.cons;Population2.cons];
            cons(cons<0) = 0;
            aaa  = sum(cons,2);
            VAR0 = max(aaa(~isinf(aaa)));
            if VAR0 == 0
                VAR0 = 1;
            end
            cnt = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                % Udate the epsilon value
                cp  = (-log(VAR0)-6)/log(1-0.5);
                VAR = VAR0*(1-X)^cp;
                cnt = cnt + 1;
                if cnt==1 || mod(cnt,g)==0
                    temp_Population = Population1;
                    if rand > 0.5
                        [GuidingSolution,samplepop] = Diversity_DirectedSampling(Problem,temp_Population,Ns,Nw,RefV,VAR);
                    else
                        [GuidingSolution,samplepop] = Convergence_DirectedSampling(Problem,temp_Population,Ns,Nw,RefV,VAR);
                    end
                    [Population1,Fitness1] = Main_task_EnvironmentalSelection([Population1,samplepop],Problem.N,true);
                    [Population2,Fitness2] = Auxiliray_task_EnvironmentalSelection([Population2,samplepop], Problem.N,VAR);
                end
                % Offspring generation
                MatingPool = [Population1(randsample(Problem.N,Problem.N))];
                [Mate1,Mate2,Mate3]       = Neighbor_Pairing_Strategy(MatingPool,Zmin1);
                Offspring1(1:Problem.N/2) = OperatorDE_rand_1(Problem,Mate1(1:Problem.N/2), Mate2(1:Problem.N/2), Mate3(1:Problem.N/2));
                Offspring1(1+Problem.N/2:Problem.N) = OperatorDE_pbest_1_main(Population1, Problem.N/2, Problem, Fitness1, 0.1);
                
                MatingPool = [Population2(randsample(Problem.N,Problem.N))];
                [Mate1,Mate2,Mate3]       = Neighbor_Pairing_Strategy(MatingPool,Zmin2);
                Offspring2(1:Problem.N/2) = OperatorDE_rand_1(Problem,Mate1(1:Problem.N/2),Mate2(1:Problem.N/2),Mate3(1:Problem.N/2));
                Offspring2(1+Problem.N/2:Problem.N) = OperatorDE_pbest_1_main(Population2, Problem.N/2, Problem, Fitness2, 0.1);
                
                Zmin1 = min([Zmin1;Offspring1.objs],[],1);
                Zmin2 = min([Zmin2;Offspring2.objs],[],1);
                
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

### `OperatorDE_pbest_1_main.m`
```matlab
function [ Offspring ] = OperatorDE_pbest_1_main(Population, popsize, Problem, Fitness, p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    warning off
    permutation = randperm(Problem.N);
    r0          = permutation;
    [r1, r2,r3] = gnR1R2R3(Problem.N, r0);

    array = permutation(1:popsize);
    pop1  = Population(array);

    [~, indBest] = sort(Fitness, 'ascend');
    pNP          = max(round(p * Problem.N), 2); % choose at least two best solutions
    randindex    = ceil(rand(1, popsize) * pNP); % select from [1, 2, 3, ..., pNP]
    randindex    = max(1, randindex); % to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest        = Population(indBest(randindex)); % randomly choose one of the top 100p% solutions

    Offspring = OperatorDE_pbest_1(Problem,Population(array),pbest,Population(r1(1:popsize)),Population(r2(1:popsize)));
end
```

### `OperatorDE_rand_1.m`
```matlab
function Offspring = OperatorDE_rand_1(Problem,Parent1,Parent2,Parent3)
% OperatorDE_rand_1

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
