# MFFS

**Tags**: <2023> <multi> <binary>

## Description
Multiform feature selection

## Reference
R. Jiao, B. Xue, and M. Zhang. Benefiting from single-objective feature selection to multiobjective feature selection: A multiform approach. IEEE Transactions on Cybernetics, 2023, 53(12): 7773-7786.

## Source Code

### `CalPartitionPoint.m`
```matlab
function PartitionSet = CalPartitionPoint(Population, FrontNo, Pset)
% Calculate partion point between two largest nondominated soltuions
    
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    [PopObj, ic] = unique(Population.objs, 'rows');
    FrontNo      = FrontNo(ic);
    CrowdAngle   = zeros(size(PopObj, 1), size(PopObj, 1));
    Fronts       = setdiff(unique(FrontNo), inf);
    Front        = find(FrontNo==Fronts(1));
    Fmin         = min(PopObj(Front, :), [], 1);
    [~, Rank] = sortrows(PopObj(Front, 1));
    for j = 1 : length(Front) - 1
        CrowdAngle(Front(Rank(j)), Front(Rank(j+1))) = acos(1-pdist2((PopObj(Front(Rank(j+1)),:) - Fmin), (PopObj(Front(Rank(j)),:) - Fmin), 'cosine'));
    end
    
    if length(Front)==1
        PartitionSet = zeros(2, 2);
    elseif length(Front)==2
        [r, c] = find(CrowdAngle == max(max(CrowdAngle)));
        PartitionSet = [(PopObj(r,:) + PopObj(c,:))./2; (PopObj(r,:) + PopObj(c,:))./2];
        CrowdAngle(r, c) = 0;
    else
        [r, c] = find(CrowdAngle == max(max(CrowdAngle)));
        MidP1  = (PopObj(r,:) + PopObj(c,:))./2;
        CrowdAngle(r, c) = 0;
        [r, c] = find(CrowdAngle == max(max(CrowdAngle)));
        MidP2 = (PopObj(r,:) + PopObj(c,:))./2;
        CrowdAngle(r,c) = 0;
        PartitionSet    = [MidP1; MidP2];
    end
    if ismember(PartitionSet(1,:), Pset, 'rows') 
        if max(max(CrowdAngle))~=0
            [r, c] = find(CrowdAngle == max(max(CrowdAngle)));
            PartitionSet(1,:) = (PopObj(r,:) + PopObj(c,:))./2;
            CrowdAngle(r, c)  = 0;
        else
            PartitionSet(1,:) = PartitionSet(2,:);
        end
    end
    if ismember(PartitionSet(2,:), Pset, 'rows')
        if max(max(CrowdAngle))~=0
            [r, c] = find(CrowdAngle == max(max(CrowdAngle)));
            PartitionSet(2,:) = (PopObj(r,:) + PopObj(c,:))./2;
        else
            PartitionSet(2,:) = PartitionSet(1,:);
        end
    end
end
```

### `EnvironmentalSelectionMOP.m`
```matlab
function [Population, FrontNo, CrowdDis] = EnvironmentalSelectionMOP(Population, N)
% Environmental selection of MFFS for the multi-objective task

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    % Calculate average distance of the parent population in the search space
    PopDec = Population(1:N).decs;
    aDist  = pdist2(round(PopDec), round(PopDec), "hamming");
    aveDis = sum(sum(tril(aDist, 0)))./sum(1:(size(PopDec,1)-1));
    
    % Remove duplicated solutions in the population
    PopDec     = Population.decs;
    [~, index] = unique(PopDec, 'rows');
    Population = Population(index);
    PopObj     = Population.objs;
    PopDec     = Population.decs;
    
    % Nondominated sorting
    FrontNo = NDSort(PopObj, inf);
    NextNo  = false(1, size(PopObj, 1));
    
    % Directly save solutions in the first front
    First         = find(FrontNo==1);
    NextNo(First) = 1;
    NDpoints      = Population(First);
    
    while sum(NextNo) < N || sum(NextNo)==size(PopObj, 1)
        for i = 2 : max(FrontNo)
            FrontPop = [];
            No       = find(FrontNo==i);
            No       = No(NextNo(No)==0);
            Pop      = Population(No);
            PopObj   = Pop.objs;
            [Obj, ~, c] = unique(PopObj, 'rows', 'stable');
            for j = 1 : max(c)
                index = find(c==j);
                if size(index, 1) > 1
                    selectedNo = DuplicationSelection(No(index), Population, NDpoints, NextNo, aveDis);
                    FrontPop = [FrontPop, selectedNo];
                else
                    FrontPop = [FrontPop, No(index)];
                end
            end
            NextNo(FrontPop) = 1;
            if sum(NextNo) >= N
                break;
            end
        end
    end
    Population = Population(NextNo);
    FrontNo    = FrontNo(NextNo);
    MaxFNo     = max(FrontNo);
    Next       = FrontNo < MaxFNo;

    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo == MaxFNo);
    [~,Rank] = sort(CrowdDis(Last), 'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
    
 function selectedNo = DuplicationSelection(index, Population, NDpoints, Next, aveDis)
     % Choose one solution from many duplicated solutions (objective space)
     NDobj   = NDpoints.objs;
     NDdec   = NDpoints.decs;
     Nextobj = Population(Next).objs;
     Nextdec = Population(Next).decs;
     Obj     = Population(index).objs;
     Dec     = Population(index).decs;
     [r, c]  = ismember(Obj(1,1), Nextobj(:,1));
     if r == 1
         aDist = pdist2(Dec, Nextdec(c,:), "hamming");
         if sum(index(aDist>=aveDis)) >= 1
             selectedNo = index(aDist>=aveDis);
         else
            [~, No]    = max(sum(aDist,2), [], 1);
            selectedNo = index(No);
         end
     else
         Dis     = abs(repmat(Obj(1,1),size(NDobj,1),1) - NDobj(:,1));
         [~, no] = min(Dis, [], 1);
         aDist   = pdist2(Dec, NDdec(no,:), "hamming");
         if sum(index(aDist>=aveDis)) >= 1
              selectedNo = index(aDist>=aveDis);
         else
            [~, No]    = max(sum(aDist,2), [], 1);
            selectedNo = index(No);
         end
     end
 end
```

### `MFFS.m`
```matlab
classdef MFFS < ALGORITHM
% <2023> <multi> <binary> 
% Multiform feature selection

%------------------------------- Reference --------------------------------
% R. Jiao, B. Xue, and M. Zhang. Benefiting from single-objective feature 
% selection to multiobjective feature selection: A multiform approach. 
% IEEE Transactions on Cybernetics, 2023, 53(12): 7773-7786.
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
            % Parameter setting
            N  = round(Problem.N*0.5);
            N1 = round(Problem.N*0.25);
            N2 = Problem.N - N - N1;
            P  = InitializePopulation(Problem);
            alphaSet     = [0.01; 0.99]; % Note: We assume the first objective is the selected feature ratio, and the second objective is the classification error rate. 
            PartitionSet = zeros(2, 2);
            [Population, FrontNo, CrowdDis] = EnvironmentalSelectionMOP(P, N);
            [SubPop1, Fitness1] = EnvironmentalSelectionSOP(P, N1, alphaSet(1,:), min(P.objs, [], 1)); % Store solutions with better classification performance 
            [SubPop2, Fitness2] = EnvironmentalSelectionSOP(P, N2, alphaSet(2,:), min(P.objs, [], 1)); % Store solutions with low selected feature ratio         
            BestFitnessSet      = [SubPop1(1).objs; SubPop2(1).objs];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Offspring generation for multi-objective task
                MatingPool = TournamentSelection(2, N, FrontNo, -CrowdDis);
                Offspring  = OffspringReproduction(Problem, Population(MatingPool), [Population, SubPop1, SubPop2]);
                % Offspring generation for single-objective task I
                MatingPoolPop1 = TournamentSelection(2, N1, Fitness1);
                OffPop1        = OffspringReproduction(Problem, SubPop1(MatingPoolPop1), [Population, SubPop1, SubPop2, Offspring]);
                % Offspring generation for single-objective task II
                MatingPoolPop2 = TournamentSelection(2, N2, Fitness2);
                OffPop2        = OffspringReproduction(Problem, SubPop2(MatingPoolPop2), [Population, SubPop1, SubPop2, Offspring, OffPop1]);
                % Environmental selection for multi-objective task
                [Population, FrontNo, CrowdDis] = EnvironmentalSelectionMOP([Population, Offspring, OffPop1, OffPop2], N);
                % Update ideal vector
                Fmin = min(Population.objs, [], 1);
                % Environmental selection for single-objective task I
                [SubPop1, Fitness1] = EnvironmentalSelectionSOP([Offspring, OffPop2, SubPop1, OffPop1], N1, alphaSet(1,:), Fmin); 
                % Environmental selection for single-objective task II
                [SubPop2, Fitness2] = EnvironmentalSelectionSOP([Offspring, OffPop1, SubPop2, OffPop2], N2, alphaSet(2,:), Fmin);                
                % Judge whether single-objective tasks convergenced
                [flag, BestFitnessSet] = boolImprovement(BestFitnessSet, [SubPop1(1).objs; SubPop2(1).objs], 5);
                % Calculate direction vectors
                [alphaSet, PartitionSet] = CalWeight(PartitionSet, alphaSet, Population, FrontNo, flag, Fmin);
                if flag == 1
                    [SubPop1, Fitness1, SubPop2, Fitness2] = ReInitialization(Problem, Population, N1, N2, alphaSet, Fmin);
                end
            end
        end
    end
end


function [alphaSet, PartitionSet] = CalWeight(PartitionSet, alphaSet, Population, FrontNo, flag, Fmin)
    % Calculate direction vectors based on partion point and ideal point
    if flag == 1
        PartitionSet = CalPartitionPoint(Population, FrontNo, PartitionSet);
        alphaSet     = UpdateWeight(PartitionSet, Fmin);
    else
        if ~isequal(alphaSet, [0.01; 0.99])
            alphaSet = UpdateWeight(PartitionSet, Fmin);
        end
    end
end

function alphaSet = UpdateWeight(PartitionSet, Fmin)
    % Update direction vectors
    if isequal(PartitionSet(1, :), PartitionSet(2, :)) && ~isequal(PartitionSet(1, :), zeros(1, 2))
        Alpha1 = round(rand(), -2);
        Alpha2 = (PartitionSet(2, 1) - Fmin(1, 1))./((PartitionSet(2, 1) - Fmin(1, 1)) + (PartitionSet(1, 2) - Fmin(1, 2)));
        Alpha2 = round(Alpha2, -2);
    elseif isequal(PartitionSet(1, :), zeros(1, 2)) && isequal(PartitionSet(2, :), zeros(1, 2))
        Alpha1 = round(rand(), -2);
        Alpha2 = round(rand(), -2);
    else
        Alpha1 = (PartitionSet(1, 1)-Fmin(1,1))./((PartitionSet(1, 1) - Fmin(1, 1)) + (PartitionSet(1, 2) - Fmin(1, 2)));
        Alpha1 = round(Alpha1, -2);
        Alpha2 = (PartitionSet(2, 1) - Fmin(1, 1))./((PartitionSet(2, 1) - Fmin(1, 1)) + (PartitionSet(2, 2) - Fmin(1, 2)));
        Alpha2 = round(Alpha2, -2);
    end
    alphaSet = [Alpha1; Alpha2];
end


function Population = InitializePopulation(Problem)
    % Initialization for all tasks
    Pop = zeros(Problem.N, Problem.D);
    for i = 1 : Problem.N
        k = randperm(round(Problem.D), 1);
        j = randperm(Problem.D, k);
        Pop(i, j) = 1;
    end
    Population = Problem.Evaluation(Pop);
end

function [SubPop1, Fitness1, SubPop2, Fitness2] = ReInitialization(Problem, Pop, N1, N2, alphaSet, Fmin)
    % Reinitialize subpopulations for the single-objective task
    Pop = Pop.decs;
    for i = 1 : size(Pop, 1)
            index1 = find( Pop(i, :));
            index2 = find(~Pop(i, :));
            if size(index1, 2) > 0
                Pop(i, index1(randi(end, 1, 1))) = 0;
            end
            if size(index2, 2) > 0
                Pop(i, index2(randi(end, 1, 1))) = 1;
            end
    end
    Population = Problem.Evaluation(Pop);
    [SubPop1, Fitness1] = EnvironmentalSelectionSOP(Population, N1, alphaSet(1,:), Fmin); 
    [SubPop2, Fitness2] = EnvironmentalSelectionSOP(Population, N2, alphaSet(2,:), Fmin);
end

function [flag, BestFitnessSet] = boolImprovement(BestFitnessSet, bestF, gen)
    % Judge whether the best fitness has improved in gen generations
    flag = 0;
    if size(BestFitnessSet, 2)/2 < gen
        BestFitnessSet = [BestFitnessSet, bestF];
    else
        for j=1:gen - 1
            BestFitnessSet(:, 2*j-1:2*j) = BestFitnessSet(:, 2*(j+1)-1:2*(j+1));
        end
        BestFitnessSet(:, 2*gen-1:2*gen) = bestF;
        if isequal(BestFitnessSet(:,1:2), BestFitnessSet(:,2*gen-1:2*gen))
            flag = 1;
            BestFitnessSet = [];
        end
    end
end

function fitness = FitnessSOP(Population, alpha, z)
    % The fitness function of single-objective task
    Obj     = Population.objs;
    Obj1    = Obj(:, 1); % Selected feature ratio
    Obj2    = Obj(:, 2); % Classification error rate
    fitness = (Obj1 - z(:, 1)).*(1-alpha) + (Obj2 - z(:, 2)).*alpha;
end

function [Population, fitness] = EnvironmentalSelectionSOP(Population, N, alpha, z)
    % Environmental selection for single-objective task
    Fit        = FitnessSOP(Population, alpha, z);
    [~, Rank]  = sort(Fit, 'ascend');
    Population = Population(Rank(1:N));
    fitness    = Fit(Rank(1:N));
end
```

### `OffspringReproduction.m`
```matlab
function Offspring = OffspringReproduction(Problem, Parent, Pop, Parameter)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    if nargin > 3
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
        flag = sum(Offspring, 2) == 0;
        if sum(flag, 1) > 0
            Offspring(flag, 1:end) = randi([0,1], sum(flag, 1), D);
        end
        
        % repair duplicated solutions
        boolis = ismember(Offspring, Pop.decs, 'rows');
        normal = Offspring(boolis==0, 1:end);
        duplic = Offspring(boolis==1, 1:end);
        for i = 1 : size(duplic, 1)
            index1 = find( duplic(i, :));
            index2 = find(~duplic(i, :));
            if size(index1, 2) > 0
                duplic(i, index1(randi(end,1,1))) = 0;
            end
            if size(index2, 2) > 0
                duplic(i, index2(randi(end,1,1))) = 1;
            end
        end
        Offspring = [normal; duplic];
        
        % get unique offspring and individuals (function evaluated)
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
