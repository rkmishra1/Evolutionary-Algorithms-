# C3M

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Constraint, multiobjective, multi-stage, multi-constraint evolutionary algorithm

## Reference
R. Sun, J. Zou, Y. Liu, S. Yang, and J. Zheng. A multi-stage algorithm for solving multi-objective optimization problems with multi-constraints. IEEE Transactions on Evolutionary Computation, 2023, 27(5): 1207-1219.

## Source Code

### `C3M.m`
```matlab
classdef C3M < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <constrained>
% Constraint, multiobjective, multi-stage, multi-constraint evolutionary algorithm
% type --- 1 --- Type of operator (1. DE 2. GA)

%------------------------------- Reference --------------------------------
% R. Sun, J. Zou, Y. Liu, S. Yang, and J. Zheng. A multi-stage algorithm
% for solving multi-objective optimization problems with multi-constraints.
% IEEE Transactions on Evolutionary Computation, 2023, 27(5): 1207-1219.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruiqing Sun

    methods
        function main(Algorithm,Problem)
            type = Algorithm.ParameterSet(1);
            processcon = 0;
            Population = Problem.Initialization();
            totalcon = size(Population(1,1).con,2);
            Fitness    = CalFitness(Population.objs);
            arch = Population;
            gen = 2;
            Pops = [];
            change_threshold = 1e-3;
            for i = 1:totalcon
                Pops{1,i} = Population;
                Pops{2,i} = CalFitness(Population.objs,Population.cons,i);
                Pops{3,i} = i;
                Pops{4,i} = 0;
            end
            Objvalues(1) = sum(sum(Population.objs,1));
            ns = 0;
            flag = 1;
            state = 0;
            seq = [];
            index = 1;
            processed = [];
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Reinitlize
                if processcon <= totalcon
                    if ns
                        ns = 0;
                        Population = Problem.Initialization();
                        Fitness    = CalFitness(Population.objs,Population.cons,processcon);
                    end
                end

                % Stage2 -> Stage3
                if  flag && processcon > totalcon
                    AllPops = [];
                    for j = 1:size(Pops,2)
                        AllPops = [AllPops,Pops{1,j}];
                    end
                    flag = 0;
                    [Population,Fitness] = EnvironmentalSelection([arch,AllPops,Population],Problem.N,processcon,totalcon);
                end

                Objvalues(gen) = sum(sum(abs(Population.objs),1));

                % Reproduction
                if type == 1
                    MatingPool = TournamentSelection(2,2*Problem.N,Fitness);
                    Offspring  = OperatorDE(Problem,Population,Population(MatingPool(1:end/2)),Population(MatingPool(end/2+1:end)));
                elseif type == 2
                    MatingPool = TournamentSelection(2,Problem.N,Fitness);
                    Offspring  = OperatorGA(Problem,Population(MatingPool));
                end

                % EnvironmentalSelection
                if flag
                    for i = 1:totalcon
                        [Pops{1,i},Pops{2,i}] = EnvironmentalSelection([Pops{1,i},Offspring],Problem.N,i,totalcon);
                    end
                end
                [Population,Fitness] = EnvironmentalSelection([Population,Offspring],Problem.N,processcon,totalcon);

                % Update state of Population
                if processcon <=totalcon
                    state = is_stable(Objvalues,gen,Population,Problem.N,change_threshold,Problem.M);
                end

                % Judge Type B relationship
                if state && processcon <=totalcon
                    ns = 1;
                    AllPops = [];
                    for j = 1:size(Pops,2)
                        AllPops = [AllPops,Pops{1,j}];
                    end
                    [FrontNo,~] = NDSort(AllPops.objs,inf);
                    if processcon == 0
                        ranks = [];
                        for j = 1:size(Pops,2)
                            minF = min(FrontNo((j-1)*Problem.N+1:j*Problem.N));
                            ranks = [ranks,minF];
                        end
                        [~,seq] = sort(ranks);
                        seq = fliplr(seq);
                    else
                        processed = [processed,processcon];
                        Pops{4,processcon} = 1;
                        Minindex = min(FrontNo((processcon-1)*Problem.N+1:processcon*Problem.N));
                        for i = 1:size(Pops,2)
                            if i ~= processcon
                                maxindex = max(FrontNo((i-1)*Problem.N+1:i*Problem.N));
                                if maxindex <= Minindex
                                    Pops{4,i} = 1;
                                end
                            end
                        end
                    end
                    unpro = 0;
                    for i = 1:size(Pops,2)
                        unpro = unpro+Pops{4,i};
                    end
                    if unpro < totalcon
                        while Pops{4,seq(index)} == 1
                            index = index + 1;
                        end
                        processcon = seq(index);
                    else
                        processcon = totalcon + 1;
                    end
                end

                % force to stage 3
                if Problem.FE ==  Problem.maxFE * 0.7
                    processcon = totalcon+1;
                end

                % Update archive
                if flag
                    arch = archive([arch,Population,Offspring],Problem.N);
                end
                gen = gen+1;
            end
        end
    end
end

function result = is_stable(Objvalues,gen,Population,N,change_threshold,M)
    result = 0;
    [FrontNo,~]=NDSort(Population.objs,size(Population.objs,1));
    NC=size(find(FrontNo==1),2);
    max_change = abs(Objvalues(gen)-Objvalues(gen-1));
    if NC == N
        change_threshold = change_threshold * abs(((Objvalues(gen) / N))/(M))*10^(M-2);
        if max_change <= change_threshold
            result = 1;
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon,processcon)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruiqing Sun

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    elseif nargin == 2
        CV = sum(max(0,PopCon),2);
    else
        PopCon = max(0,PopCon(:,processcon));
        CV = sum(PopCon,2);
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
function [Population,Fitness] = EnvironmentalSelection(Population,N,processcon,totalcon)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruiqing Sun

    %% Calculate the fitness of each solution
    if ~processcon
        Fitness = CalFitness(Population.objs);
    elseif processcon > totalcon
        Fitness = CalFitness(Population.objs,Population.cons);
    else
        Fitness = CalFitness(Population.objs,Population.cons,processcon);
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

### `archive.m`
```matlab
function Population = archive(Population,N)
% Select feasible and non-dominated solutions by using NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruiqing Sun

    %% Select feasible solutions
    fIndex     = all(Population.cons <= 0,2);
    Population = Population(fIndex);

    if isempty(Population)
        return
    else
        %% Non-dominated sorting
        [FrontNo,~] = NDSort(Population.objs,1);
        Next = (FrontNo == 1);
        Population = Population(Next);
        if sum(Next) > N
            %% Calculate the crowding distance of each solution
            CrowdDis = CrowdingDistance(Population.objs);
            [~,Rank] = sort(CrowdDis,'descend');
            Population = Population(Rank(1:N));
        end
    end
end

function CrowdDis = CrowdingDistance(PopObj)
% Calculate the crowding distance of each solution

    [N,M]    = size(PopObj);
    CrowdDis = zeros(1,N);
    Fmax     = max(PopObj,[],1);
    Fmin     = min(PopObj,[],1);

    for i = 1 : M
        [~,Rank] = sortrows(PopObj(:,i));
        CrowdDis(Rank(1))   = inf;
        CrowdDis(Rank(end)) = inf;

        for j = 2 : (N - 1)
            CrowdDis(Rank(j)) = CrowdDis(Rank(j))+(PopObj(Rank(j+1),i)-PopObj(Rank(j-1),i))/(Fmax(i)-Fmin(i));
        end
    end
end
```
