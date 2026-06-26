# PRDH

**Tags**: <2024> <multi> <binary>

## Description
Problem reformulation and duplication handling

## Reference
R. Jiao, B. Xue, and M. Zhang. Solving multiobjective feature selection problems in classification via problem reformulation and duplication handling. IEEE Transactions on Evolutionary Computation, 2024, 28(4): 846-860.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population, FrontNo, CrowdDis] = EnvironmentalSelection(Population, N)
% The environmental selection of PRDH
% Note: We assume the first objective is the selected feature ratio, and the
%       second objective is the classification error rate. 

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    % Remove duplicated solutions (search space)
    [~, index] = unique(Population.decs, 'rows');
    Population = Population(index);

    FrontNo = NDSort(Population.objs, inf);
    NextNo  = false(1, size(Population.objs, 1));
    
    % Directly save solutions in the first front
    NextNo(find(FrontNo==1)) = 1;
    NDpoints = Population(find(FrontNo==1));

    %% Duplication handling (objective space)
    for i = 2:max(FrontNo)
        FrontPop  = [];
        No        = find(FrontNo==i);
        PopObj    = Population(No).objs;
        [~, ~, c] = unique(PopObj, 'rows', 'stable');
        for j = 1 : max(c)
            index = find(c==j);
            if size(index, 1) > 1
                selectedNo = DuplicationSelection(No(index), Population, NDpoints);
                FrontPop   = [FrontPop, selectedNo];
            else
                FrontPop = [FrontPop, No(index)];
            end
        end
        NextNo(FrontPop) = 1;
    end
    Population = Population(NextNo);
    FrontNo    = FrontNo(NextNo);
    
    %% Problem reformulation and constraint handling
    % Find the solution with the minimum number of selected features
    index   = findminobj1(Population.objs);
    PopObj  = Population.objs;
    minobj2 = PopObj(index, 2);

    if sum(PopObj(:, 2)<=minobj2) >= N
        %% Selection among feasible solutions
        boolFeasible = PopObj(:, 2)<=minobj2;
        Population   = Population(1:end, boolFeasible);
        FrontNo      = FrontNo(boolFeasible);
        sumt         = 0;
        for MaxFNo = 1 : max(FrontNo)
            sumt = sumt + sum(FrontNo==MaxFNo);
            if sumt >= N
                break;
            end
        end
        Next = FrontNo < MaxFNo;
    
        %% Calculate the crowding distance of each solution
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    
        %% Select the solutions in the last front based on their crowding distances
        Last     = find(FrontNo==MaxFNo);
        [~,Rank] = sort(CrowdDis(Last), 'descend');
        Next(Last(Rank(1:N-sum(Next)))) = true;
    
        %% Population for next generation
        Population = Population(Next);
        FrontNo    = FrontNo(Next);
        CrowdDis   = CrowdDis(Next);
    else
        %% Selection including infeasible solutions
        [~, rank] = sort(PopObj(:, 1));
        if size(Population, 2) < N
            N = size(Population, 2);
        end
        Population = Population(rank(1:N));
        FrontNo    = 1 : N;
        CrowdDis   = zeros(1, N);
    end
end

function selectedNo = DuplicationSelection(index, Population, NDpoints)
    %% Choose promising duplicated solutions (objective space) to survive
    NDobj = NDpoints.objs;
    NDdec = NDpoints.decs;
    Obj   = Population(index).objs;
    Dec   = Population(index).decs;
    c1    = find(min(abs(Obj(1,1)-NDobj(:,1)))==abs(Obj(1,1)-NDobj(:,1)));
    c2    = find(min(abs(Obj(1,2)-NDobj(:,2)))==abs(Obj(1,2)-NDobj(:,2)));
    o1    = [];
    for i = 1 : size(c1,1)
       tmp = pdist2(Dec(:,:), NDdec(c1(i,:), :), "hamming");
       o1  = [o1, tmp];
    end
    o1 = mean(o1, 2);
    o2 = [];
    for i = 1 : size(c2,1)
       index2 = find(NDdec(c2(i,:),:)==1);
       tmp    = pdist2(Dec(:,index2), NDdec(c2(i,:), index2), "hamming");
       o2     = [o2, tmp];
    end
    o2 = mean(o2, 2);
    o  = [o1,o2];
    [FrontNO, ~] = NDSort(-o, inf);  
    No = find(FrontNO==1);
    selectedNo   = index(No);
end

function minindex = findminobj1(obj)
    minindex = 1;
    for i = 1 : size(obj, 1)
        if obj(i, 1) < obj(minindex, 1) || (obj(i, 1)==obj(minindex, 1)&obj(i, 2)<obj(minindex, 2))
            minindex = i;
        end
    end
end
```

### `OffspringReproduction.m`
```matlab
function Offspring = OffspringReproduction(Problem, Parent, Parameter)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    if nargin > 2
        [proC,disC,proM,disM] = deal(Parameter{:});
    else
        [proC,disC,proM,disM] = deal(1,20,1,20);
    end
    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        Parent    = Parent.decs;
    else
        evaluated = false;
    end
    Parent1   = Parent(1:floor(end/2),:);
    Parent2   = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N, D]    = size(Parent1);
    Offspring = zeros(2*size(Parent1,1),size(Parent1,2));
    Type      = arrayfun(@(i)find(Problem.encoding==i),1:5,'UniformOutput',false);
    if ~isempty([Type{1:2}])    % Real and integer variables
        Offspring(:,[Type{1:2}]) = GAreal(Parent1(:,[Type{1:2}]),Parent2(:,[Type{1:2}]),Problem.lower([Type{1:2}]),Problem.upper([Type{1:2}]),proC,disC,proM*length([Type{1:2}])/size(Parent1,2),disM);
    end
    if ~isempty(Type{3})        % Label variables
        Offspring(:,Type{3}) = GAlabel(Parent1(:,Type{3}),Parent2(:,Type{3}),Problem.lower(Type{3}),Problem.upper(Type{3}),proC,proM*length(Type{3})/size(Parent1,2));
    end
    if ~isempty(Type{4})        % Binary variables
        Offspring(:,Type{4}) = GAbinary(Parent1(:,Type{4}),Parent2(:,Type{4}),proC,proM*length(Type{4})/size(Parent1,2));
    end
    if ~isempty(Type{5})        % Permutation variables
        Offspring(:,Type{5}) = GApermutation(Parent1(:,Type{5}),Parent2(:,Type{5}),proC);
    end

    if evaluated
        %% Repair solutions that do not select any features
        flag = sum(Offspring, 2) == 0;
        if sum(flag, 1) > 0
            Offspring(flag, 1:end) = randi([0,1], sum(flag, 1), D);
        end

        %% get unique offspring and individuals (function evaluated)
        Offspring = unique(Offspring, 'rows');
        Offspring = Offspring(sum(Offspring,2)>0, 1:end);
        Offspring = Problem.Evaluation(Offspring);
    end
end

function Offspring = GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM)
% Genetic operators for real and integer variables

    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
             
    %% Polynomial mutation
    Lower = repmat(lower,2*N,1);
    Upper = repmat(upper,2*N,1);
    Site  = rand(2*N,D) < proM/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end

function Offspring = GAlabel(Parent1,Parent2,lower,upper,proC,proM)
% Genetic operators for label variables

    %% Uniform crossover
    [N,D] = size(Parent1);
    k     = rand(N,D) < 0.5;
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
    
    %% Bitwise mutation
    Site = rand(2*N,D) < proM/D;
    Rand = round(unifrnd(repmat(lower,2*N,1),repmat(upper,2*N,1)));
    Offspring(Site) = Rand(Site);
end

function Offspring = GAbinary(Parent1,Parent2,proC,proM)
% Genetic operators for binary variables

    %% Uniform crossover
    [N,D] = size(Parent1);
    k     = rand(N,D) < 0.5;
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
    
    %% Bit-flip mutation
    Site = rand(2*N,D) < proM/D;
    Offspring(Site) = ~Offspring(Site);
end

function Offspring = GApermutation(Parent1,Parent2,proC)
% Genetic operators for permutation variables

    %% Order crossover
    [N,D]     = size(Parent1);
    Offspring = [Parent1;Parent2];
    k = randi(D,1,2*N);
    for i = 1 : N
        if rand < proC
            Offspring(i,k(i)+1:end)   = setdiff(Parent2(i,:),Parent1(i,1:k(i)),'stable');
            Offspring(i+N,k(i)+1:end) = setdiff(Parent1(i,:),Parent2(i,1:k(i)),'stable');
        end
    end
    
    %% Slight mutation
    k = randi(D,1,2*N);
    s = randi(D,1,2*N);
    for i = 1 : 2*N
        if s(i) < k(i)
            Offspring(i,:) = Offspring(i,[1:s(i)-1,k(i),s(i):k(i)-1,k(i)+1:end]);
        elseif s(i) > k(i)
            Offspring(i,:) = Offspring(i,[1:k(i)-1,k(i)+1:s(i)-1,k(i),s(i):end]);
        end
    end
end
```

### `PRDH.m`
```matlab
classdef PRDH < ALGORITHM
% <2024> <multi> <binary> 
% Problem reformulation and duplication handling

%------------------------------- Reference --------------------------------
% R. Jiao, B. Xue, and M. Zhang. Solving multiobjective feature selection
% problems in classification via problem reformulation and duplication
% handling. IEEE Transactions on Evolutionary Computation, 2024, 28(4):
% 846-860.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    methods
        function main(Algorithm,Problem)
            %% Generate initial population
            Population = InitializePopulation(Problem);
            [~, FrontNo, CrowdDis] = EnvironmentalSelection(Population, Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2, Problem.N, FrontNo, -CrowdDis);
                Offspring  = OffspringReproduction(Problem, Population(MatingPool));
                [Population, FrontNo, CrowdDis] = EnvironmentalSelection([Population, Offspring], Problem.N);
            end
        end
    end
end

function Population = InitializePopulation(Problem)
    T   = min(Problem.D, Problem.N * 3);
    Pop = zeros(Problem.N, Problem.D);
    for i = 1 : Problem.N
        k = randperm(T, 1);
        j = randperm(Problem.D, k);
        Pop(i, j) = 1;
    end
    Population = Problem.Evaluation(Pop);
end
```
