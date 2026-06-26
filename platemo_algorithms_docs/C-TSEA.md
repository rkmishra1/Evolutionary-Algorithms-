# C-TSEA

**Tags**: <2021> <multi/many> <real/integer/label/binary/permutation> <constrained>

## Description
Constrained two-stage evolutionary algorithm

## Reference
F. Ming, W. Gong, H. Zhen, S. Li, L. Wang, and Z. Liao. A simple two-stage evolutionary algorithm for constrained multi-objective optimization. Knowledge-Based Systems, 2021, 228: 107263.

## Source Code

### `CTSEA.m`
```matlab
classdef CTSEA < ALGORITHM
% <2021> <multi/many> <real/integer/label/binary/permutation> <constrained>
% Constrained two-stage evolutionary algorithm

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, H. Zhen, S. Li, L. Wang, and Z. Liao. A simple
% two-stage evolutionary algorithm for constrained multi-objective
% optimization. Knowledge-Based Systems, 2021, 228: 107263.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    methods
        function main(Algorithm,Problem)
            %% Generate the sampling points and random population
            Population = Problem.Initialization();
            W = UniformPoint(Problem.N,Problem.M);
            [ARMOEA_Archive,RefPoint,Range] = UpdateRefPoint(Population.objs,W,[]);
            CV = sum(max(0,Population.cons),2);
            Archive = Population(CV==0);
            stage_conver = 0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Problem.FE<0.5*Problem.maxFE
                    % evolve population to PF by ARMOEA
                    MatingPool = MatingSelection1(Population,RefPoint,Range);
                    Offspring = OperatorGA(Problem,Population(MatingPool));
                    [ARMOEA_Archive,RefPoint,Range] = UpdateRefPoint([ARMOEA_Archive;Offspring.objs],W,Range);
                    Archive = UpdateArchive(Archive,[Population,Offspring],Problem.N);
                    [Population,Range] = EnvironmentalSelection1([Population,Offspring],RefPoint,Range,Problem.N);
                else
                    if stage_conver==0
                        % exchange archive and population
                        temp = Population;
                        Population = Archive;
                        Archive = temp;
                        stage_conver = 1;
                    end
                    % evolve population to CPF by modified SPEA2
                    MatingPool = MatingSelection2(Population,Archive,Problem.N);
                    Offspring = OperatorGA(Problem,MatingPool);
                    Population = EnvironmentalSelection2([Population,Offspring],Problem.N);
                end
            end
        end
    end
end
```

### `CalDistance.m`
```matlab
function Distance = CalDistance(PopObj,RefPoint)
% Calculate the distance between each solution to each adjusted reference point

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    N  = size(PopObj,1);
    NR = size(RefPoint,1);
    PopObj   = max(PopObj,1e-6);
    RefPoint = max(RefPoint,1e-6);

    %% Adjust the location of each reference point
    Cosine = 1 - pdist2(PopObj,RefPoint,'cosine');
    NormR  = sqrt(sum(RefPoint.^2,2));
    NormP  = sqrt(sum(PopObj.^2,2));
    d1     = repmat(NormP,1,NR).*Cosine;
    d2     = repmat(NormP,1,NR).*sqrt(1-Cosine.^2);
    [~,nearest] = min(d2,[],1);
    RefPoint    = RefPoint.*repmat(d1(N.*(0:NR-1)+nearest)'./NormR,1,size(RefPoint,2));
    
    %% Calculate the distance between each solution to each point
    Distance = pdist2(PopObj,RefPoint);
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

% This function is written by Fei Ming

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

### `EnvironmentalSelection1.m`
```matlab
function [Population,Range] = EnvironmentalSelection1(Population,RefPoint,Range,N)
% The environmental selection of AR-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Selection among feasible solutions
    % Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;
    % Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Last).objs,RefPoint,Range,N-sum(Next));
    Next(Last(Choose)) = true;
    Population = Population(Next);
    % Update the range for normalization
    Range(2,:) = max(Population.objs,[],1);
    Range(2,Range(2,:)-Range(1,:)<1e-6) = 1;
end

function Remain = LastSelection(PopObj,RefPoint,Range,K)
% Select part of the solutions in the last front

    N  = size(PopObj,1);
    NR = size(RefPoint,1);

    %% Calculate the distance between each solution and point
    Distance    = CalDistance(PopObj-repmat(Range(1,:),N,1),RefPoint);
    Convergence = min(Distance,[],2);
    
    %% Delete the solution which has the smallest metric contribution one by one
    [dis,rank] = sort(Distance,1);
    Remain     = true(1,N);
    while sum(Remain) > K
        % Calculate the fitness of noncontributing solutions
        Noncontributing = Remain;
        Noncontributing(rank(1,:)) = false;
        METRIC = sum(dis(1,:)) + sum(Convergence(Noncontributing));
        Metric = inf(1,N);
        Metric(Noncontributing) = METRIC - Convergence(Noncontributing);
        % Calculate the fitness of contributing solutions
        for p = find(Remain & ~Noncontributing)
            temp = rank(1,:) == p;
            noncontributing = false(1,N);
            noncontributing(rank(2,temp)) = true;
            noncontributing = noncontributing & Noncontributing;
            Metric(p) = METRIC - sum(dis(1,temp)) + sum(dis(2,temp)) - sum(Convergence(noncontributing));
        end
        % Delete the worst solution and update the variables
        [~,del] = min(Metric);
        temp    = rank ~= del;
        dis     = reshape(dis(temp),sum(Remain)-1,NR);
        rank    = reshape(rank(temp),sum(Remain)-1,NR);
        Remain(del) = false;
    end
end
```

### `EnvironmentalSelection2.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection2(Population,N)
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
    Fitness = CalFitness(Population.objs,Population.cons);

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

### `MatingSelection1.m`
```matlab
function MatingPool = MatingSelection1(Population,RefPoint,Range)
% The mating selection of AR-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each feasible solution based on IGD-NS
    % Calculate the distance between each solution and point
    N = length(Population);
    Distance    = CalDistance(Population.objs-repmat(Range(1,:),N,1),RefPoint);
    Convergence = min(Distance,[],2);
    [dis,rank]  = sort(Distance,1);
    % Calculate the fitness of noncontributing solutions
    Noncontributing = true(1,N);
    Noncontributing(rank(1,:)) = false;
    METRIC   = sum(dis(1,:)) + sum(Convergence(Noncontributing));
    fitness  = inf(1,N);
    fitness(Noncontributing) = METRIC - Convergence(Noncontributing);
    % Calculate the fitness of contributing solutions
    for p = find(~Noncontributing)
        temp = rank(1,:) == p;
        noncontributing = false(1,N);
        noncontributing(rank(2,temp)) = true;
        noncontributing = noncontributing & Noncontributing;
        fitness(p) = METRIC - sum(dis(1,temp)) + sum(dis(2,temp)) - sum(Convergence(noncontributing));
    end

    %% Combine the fitness of feasible solutions with the fitness of infeasible solutions
    Fitness = -inf(1,length(Population));
    
    %% Binary tournament selection
    MatingPool = TournamentSelection(2,length(Population),-Fitness);
end
```

### `MatingSelection2.m`
```matlab
function MatingPool = MatingSelection2(Population,Archive,N)
% The mating selection of stage 2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    Fitness1    = CalFitness(Population.objs,Population.cons);
    MatingPool1 = TournamentSelection(2,N,Fitness1);
    MatingPool  = Population(MatingPool1);
end
```

### `UpdateArchive.m`
```matlab
function UpdatedArchive=UpdateArchive(Archive,Population,N)
% Update Archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

        CV = sum(max(0,Population.cons),2);
        Archive=[Archive,Population(CV==0)];
        if length(Archive)==N
            UpdatedArchive=Archive;
        elseif length(Archive)<N
            Population=setdiff(Population,Archive);
            CV = sum(max(0,Population.cons),2);
            [~,index]=sort(CV,'ascend');
            remain_size=N-length(Archive);
            Remain=Population(index(1:remain_size));
            UpdatedArchive=[Archive,Remain];
        else
            Fitness=CalFitness(Archive.objs,Archive.cons);
            Next = Fitness < 1;
            if sum(Next) < N
                [~,Rank] = sort(Fitness);
                Next(Rank(1:N)) = true;
            elseif sum(Next) > N
                Del  = Truncation(Archive(Next).objs,sum(Next)-N);
                Temp = find(Next);
                Next(Temp(Del)) = false;
            end
            % Archive for next generation
            UpdatedArchive = Archive(Next);
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

### `UpdateRefPoint.m`
```matlab
function [Archive,RefPoint,Range] = UpdateRefPoint(Archive,W,Range)
% Reference point adaption

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

	%% Delete duplicated and dominated solutions
    Archive = unique(Archive(NDSort(Archive,1)==1,:),'rows');
    NA      = size(Archive,1);
    NW      = size(W,1);
    
	%% Update the ideal point
    if ~isempty(Range)
        Range(1,:) = min([Range(1,:);Archive],[],1);
    elseif ~isempty(Archive)
        Range = [min(Archive,[],1);max(Archive,[],1)];
    end
    
    %% Update archive and reference points
    if size(Archive,1) <= 1
        RefPoint = W;
    else
        %% Find contributing solutions and valid weight vectors
        tArchive = Archive - repmat(Range(1,:),NA,1);
        W        = W.*repmat(Range(2,:)-Range(1,:),NW,1);
        Distance      = CalDistance(tArchive,W);
        [~,nearestP]  = min(Distance,[],1);
        ContributingS = unique(nearestP);
        [~,nearestW]  = min(Distance,[],2);
        ValidW        = unique(nearestW(ContributingS));

        %% Update archive
        Choose = ismember(1:NA,ContributingS);
        Cosine = 1 - pdist2(tArchive,tArchive,'cosine');
        Cosine(logical(eye(size(Cosine,1)))) = 0;
        while sum(Choose) < min(3*NW,size(tArchive,1))
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
        Archive  = Archive(Choose,:);
        tArchive = tArchive(Choose,:);

        %% Update reference points
        RefPoint = [W(ValidW,:);tArchive];
        Choose   = [true(1,length(ValidW)),false(1,size(tArchive,1))];
        Cosine   = 1 - pdist2(RefPoint,RefPoint,'cosine');
        Cosine(logical(eye(size(Cosine,1)))) = 0;
        while sum(Choose) < min(NW,size(RefPoint,1))
            Selected = find(~Choose);
            [~,x]    = min(max(Cosine(~Choose,Choose),[],2));
            Choose(Selected(x)) = true;
        end
        RefPoint = RefPoint(Choose,:);
    end 
end
```
