# AC-MMEA

**Tags**: <2023> <multi> <real/integer> <large/none> <multimodal> <sparse>

## Description
Adaptive merging and coordinated offspring generation based multi-modal multi-objective evolutionary algorithm

## Reference
X. Wang, T. Zheng, and Y. Jin. Adaptive merging and coordinated offspring generation in multi-population evolutionary multi-modal multi-objective optimization. Proceedings of the International Conference on Data-driven Optimization of Complex Systems, 2023.

## Source Code

### `ACMMEA.m`
```matlab
classdef ACMMEA < ALGORITHM
% <2023> <multi> <real/integer> <large/none> <multimodal> <sparse>
% Adaptive merging and coordinated offspring generation based multi-modal multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% X. Wang, T. Zheng, and Y. Jin. Adaptive merging and coordinated offspring
% generation in multi-population evolutionary multi-modal multi-objective
% optimization. Proceedings of the International Conference on Data-driven
% Optimization of Complex Systems, 2023.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    methods
        function main(Algorithm,Problem)
            %% Population initialization
            Dec  = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            Mask = zeros(Problem.N,Problem.D);
            GV   = ones(1,Problem.D);
            for i = 1 : Problem.N
                Mask(i,TournamentSelection(2,ceil(rand*Problem.D),GV)) = 1;
                GV(Mask(i,:)==1) = GV(Mask(i,:)==1)+1;
            end
            Population  = Problem.Evaluation(Dec.*Mask);
            [slst] = Clustering(Population.decs, 20, [Problem.lower,Problem.upper], Problem.D);
            K=size(slst,2);
            Masks       = cell(1,K);
            Decs        = cell(1,K);
            Populations = cell(1,K);
            GV          = cell(1,K);
            FrontNo     = cell(1,K);
            CrowdDis    = cell(1,K);
            Fitness     = cell(1,K); 
            for i = 1 : K
                Populations{i} = Population(slst{i});
                Masks{i}       = Mask(slst{i},:);
                Decs{i}        = Dec(slst{i},:);
                [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},length(Populations{i}));
                GV{i}          = UpdateGV(zeros(1,Problem.D),Masks{i},FrontNo{i});
            end
            StageIFlag = 1;
            Timea      = 0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Problem.FE < 0.3 * Problem.maxFE 
                    [~,rank] = sort(SubPopRank(Populations));
                    for i = 1 : K
                        GV{rank(i)}          = UpdateGV(GV{rank(i)},Masks{rank(i)},FrontNo{rank(i)});
                        Mating               = TournamentSelection(2,2*length(Populations{rank(i)}),FrontNo{rank(i)},-CrowdDis{rank(i)});
                        [OffDec,OffMask]     = Operator(Problem,Decs{rank(i)}(Mating,:),Masks{rank(i)}(Mating,:),GV{rank(i)}, StageIFlag);
                        Offspring            = Problem.Evaluation(OffDec.*OffMask);
                        Populations{rank(i)} = [Populations{rank(i)},Offspring];
                        Decs{rank(i)}        = [Decs{rank(i)};OffDec];
                        Masks{rank(i)}       = [Masks{rank(i)};OffMask];
                        if i > 1
                            for j = 1 : i-1
                                [~,fs(rank(j))] = min(mean(Populations{rank(j)}.objs,2));
                            end
                            R = zeros(1,Problem.D);
                            for j = 1 : i-1
                                R = R + Masks{rank(j)}(fs(rank(j)),:);
                            end
                            R(R>0) = 1;
                            dis = sum(repmat(R,length(Populations{rank(i)}),1)&Masks{rank(i)},2);
                            [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},CrowdDis{rank(i)}] = EnvironmentalSelection(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},floor(Problem.N/K),dis);
                        else
                            [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},CrowdDis{rank(i)}] = EnvironmentalSelection(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},floor(Problem.N/K));
                        end
                    end
                    if mod(ceil(Problem.FE/Problem.N),50)==0 
                        [Populations,Masks,Decs,GV,K]=SubPopSimility(Populations,Masks,Decs,GV);
                        FrontNo  = cell(1,K);
                        CrowdDis = cell(1,K);
                        for i = 1 : K
                            [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},floor(Problem.N/K));
                        end
                    end
                else 
                    if Timea == 0
                        for i = 1 : K
                            [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},Fitness{rank(i)}] = EnvironmentalSelectionS(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},length(Populations{rank(i)}));
                            GV{rank(i)} = UpdateGV(zeros(1,Problem.D),Masks{rank(i)},FrontNo{rank(i)});
                        end
                    end
                    Timea = Timea + 1;
                    StageIFlag = 0;
                    for i=1:K
                        GV{rank(i)}          = UpdateGV(GV{rank(i)},Masks{rank(i)},FrontNo{rank(i)});
                        Mating               = TournamentSelection(2,2*length(Populations{rank(i)}),FrontNo{rank(i)},Fitness{rank(i)});
                        [OffDec,OffMask]     = Operator(Problem,Decs{rank(i)}(Mating,:),Masks{rank(i)}(Mating,:),GV{rank(i)}, StageIFlag);
                        Offspring            = Problem.Evaluation(OffDec.*OffMask);
                        Populations{rank(i)} = [Populations{rank(i)},Offspring];
                        Decs{rank(i)}        = [Decs{rank(i)};OffDec];
                        Masks{rank(i)}       = [Masks{rank(i)};OffMask]; 
                        [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},Fitness{rank(i)}] = EnvironmentalSelectionS(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},floor(Problem.N/K));

                    end
                    if mod(ceil(Problem.FE/Problem.N),50)==0
                        [Populations,Masks,Decs,GV,K]=SubPopSimility(Populations,Masks,Decs,GV);
                        FrontNo  = cell(1,K);
                        CrowdDis = cell(1,K);
                        for i = 1 : K
                            [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},floor(Problem.N/K));
                        end
                        Timea    = 0;
                        [~,rank] = sort(SubPopRank(Populations));
                    end
                end
                Population = [Populations{:}];
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    N = size(PopObj,1);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
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

### `Clustering.m`
```matlab
function slst = Clustering(X, n, bound, dim)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    a = size(X,1);
    p = X;
    p(:,dim+1) = [1:a];
    for i = 1 : a
        G{i} = i;
    end
    M = [];
    for j = 1 : a
        M(j,:) = ((sum((repmat(X(j,:),size(X,1),1) - X).^2,2)).^0.5)';%M(i, j) reprents the distance of i-th cluster to the j-th cluster
    end
    found = 1;
    while found == 1
        found = 0;
        min_dist = (dim*((bound(2) - bound(1)).^2)).^0.5;
        for i = 1:size(G,2)-1
            for j = i + 1 :size(G,2)
                if size(G{i},2) + size(G{j},2) < n+1
                    if min_dist >= M(i, j)
                        min_dist = M(i, j);
                        r = i;
                        s = j;
                        found = 1;
                    end
                end
            end

        end
        if ~isempty(r) && ~isempty(s)
            G{r} = [G{r}, G{s}]; % merge clusters r and s;
            for k = 1 : size(G{s},2)
                p(p(:,dim+1)==G{s}(k),:) = [];
            end
            G(s)   = []; % delete cluster s;
            M(:,s) = []; % update the M
            M(s,:) = []; % update the M
            h      = p(:,1:dim);
            for k = 1 : size(G{r},2)
                D(k,:) = ((sum((repmat(X(G{r}(k),:),size(h,1),1) - h).^2,2)).^0.5)';
            end
            D = min(D,[],1);
            M(:,r) = D';M(r,:) = D;
            D = [];
            s = [];
            r = [];
            c = 0;
            for i = 1 : size(G,2)
                if size(G{i},2) == 1
                    c = c + 1;
                end
            end
            if c == 0
                found = 0;
            end
        end
    end
    slst = G;
end
```

### `CpuGroup.m`
```matlab
function [Index,MAX] = CpuGroup(numberOfGroups,xPrime,numberOfVariables)       

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    varsPerGroup = floor(numberOfVariables/numberOfGroups);
    if varsPerGroup == 1
        Index = linspace(1,numberOfVariables,numberOfVariables);
        MAX   = numberOfVariables;
    else
        B      = ones(1,varsPerGroup*numberOfGroups);
        remain = ones(1,(numberOfVariables-varsPerGroup*numberOfGroups))*(numberOfGroups+1);
        R      = reshape(B,varsPerGroup,numberOfGroups);
        k      = linspace(1,numberOfGroups,numberOfGroups);
        index  = R.*repmat(k,varsPerGroup,1);
        index  = reshape(index,1,varsPerGroup*numberOfGroups);
        INDEX  = [index remain];
        [~,I]  = sort(xPrime);
        Index(I) = INDEX;
        if(mod(numberOfVariables,numberOfGroups)==0)
            MAX = numberOfGroups;
        else
            MAX = numberOfGroups + 1;
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,N,dis)
% The environmental selection of MP-MMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    if nargin > 4
        PopObj = [Population.objs,dis];
    else
        PopObj = Population.objs;
    end

    %% Delete duplicated solutions
    [~,uni]    = unique(PopObj,'rows');
    PopObj     = PopObj(uni,:);
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(PopObj,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `EnvironmentalSelectionS.m`
```matlab
function [Population,Dec,Mask,FrontNo,Fitness] = EnvironmentalSelectionS(Population,Dec,Mask,N,dis)
% The environmental selection of AC-MMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)
    
    if nargin > 4
        PopObj = [Population.objs,dis];
    else
        PopObj = Population.objs;
    end

    %% Delete duplicated solutions
    [~,uni]    = unique(PopObj,'rows');
    PopObj     = PopObj(uni,:);
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Calculate the fitness of each solution
    Fitness = CalFitness(PopObj);

    %% Non-dominated sorting
    [FrontNo,~] = NDSort(PopObj,Population.cons,N);

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
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    Fitness    = Fitness(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
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

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Score, StageIFlag)
% Operator of AC-MMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)
    
    %% Parameter setting
    N = size(ParentDec,1);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2    
        index1 = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
        index2 = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
        p1 = 1./(1+exp(-Score(index1)));
        p2 = 1./(1+exp(-Score(index2)));
        idx1 = index1(p1<rand(size(p1)));
        idx2 = index2(p2>rand(size(p2)));          
        OffMask(i,idx1) = 0;
        OffMask(i,idx2) = 1;
    end
    
    %% Mutation for mask
    for i = 1 : N/2    
        if rand < 0.5
            index = find(OffMask(i,:));
            index = index(TS(Score(index)));
            OffMask(i,index) = 0;
        else
            index = find(~OffMask(i,:));
            index = index(TS(-Score(index)));
            OffMask(i,index) = 1;
        end
    end
    
    %% select the suitable index to give highly mutation 
    % using the group method by SLMEA
    Ratio = sum(OffMask,1)./N;
    [indexM,MAX] = CpuGroup(5,Ratio,size(ParentDec,2));

    %% Crossover and mutation for dec
    OffDec = OperatorGAhalfSMM(Problem, ParentDec, indexM, MAX,StageIFlag, {1,20,1,0.5});
end

function index = TS(Fitness)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2,1,Fitness);
    end
end
```

### `OperatorGAhalfSMM.m`
```matlab
function Offspring = OperatorGAhalfSMM(Problem,Parent, indexM, MAX, Flag, Parameter)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    if nargin > 4
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
    Offspring = zeros(size(Parent1,1),size(Parent1,2));
    Type      = arrayfun(@(i)find(Problem.encoding==i),1:5,'UniformOutput',false);
    if ~isempty([Type{1:2}])    % Real and integer variables
        Offspring(:,[Type{1:2}]) = GAreal(Parent1(:,[Type{1:2}]),Parent2(:,[Type{1:2}]),Problem.lower([Type{1:2}]),Problem.upper([Type{1:2}]),proC,disC,proM,disM, indexM, MAX, Flag);
    end
    if ~isempty(Type{3})        % Label variables
        Offspring(:,Type{3}) = GAlabel(Parent1(:,Type{3}),Parent2(:,Type{3}),proC,proM);
    end
    if ~isempty(Type{4})        % Binary variables
        Offspring(:,Type{4}) = GAbinary(Parent1(:,Type{4}),Parent2(:,Type{4}),proC,proM);
    end
    if ~isempty(Type{5})        % Permutation variables
        Offspring(:,Type{5}) = GApermutation(Parent1(:,Type{5}),Parent2(:,Type{5}),proC);
    end
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end

function Offspring = GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM, indexM, MAX, Flag)
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
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;

    %% Polynomial mutation
    Lower = repmat(lower,N,1);
    Upper = repmat(upper,N,1);
    if Flag
        % highest mutation
        Site  = rand(N,D) < 4 * proM/D;
        mu    = rand(N,D);
        indx_mu = indexM == MAX ;
        temp  = Site & mu<=0.5 & indx_mu;
        Offspring       = min(max(Offspring,Lower),Upper);
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
            (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5 & indx_mu;
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
            (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

        % highest mutation
        Site  = rand(N,D) < 4 * proM/D;
        mu    = rand(N,D);
        indx_mu = indexM == MAX-2 | indexM == MAX-1;
        temp  = Site & mu<=0.5 & indx_mu;
        Offspring       = min(max(Offspring,Lower),Upper);
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
            (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5 & indx_mu;
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
            (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

        % highest mutation
        Site  = rand(N,D) < 4 * proM/D ;
        mu    = rand(N,D);
        indx_mu = indexM == MAX-3 | indexM == MAX-4;
        temp  = Site & mu<=0.5 & indx_mu;
        Offspring       = min(max(Offspring,Lower),Upper);
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
            (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5 & indx_mu;
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
            (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    else
        % highest mutation
        Site  = rand(N,D) < 4 * proM/D;
        mu    = rand(N,D);
        indx_mu = indexM == MAX ;
        temp  = Site & mu<=0.5 & indx_mu;
        Offspring       = min(max(Offspring,Lower),Upper);
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
            (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5 & indx_mu;
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
            (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

        % high mutation
        Site  = rand(N,D) < 2 * proM/D;
        mu    = rand(N,D);
        indx_mu = indexM == MAX-2 | indexM == MAX-1;
        temp  = Site & mu<=0.5 & indx_mu;
        Offspring       = min(max(Offspring,Lower),Upper);
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
            (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5 & indx_mu;
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
            (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

        % low mutation
        Site  = rand(N,D) < proM/D/2;
        mu    = rand(N,D);
        indx_mu = indexM == MAX-3 | indexM == MAX-4;
        temp  = Site & mu<=0.5 & indx_mu;
        Offspring       = min(max(Offspring,Lower),Upper);
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
            (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp = Site & mu>0.5 & indx_mu;
        Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
            (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    end
end

function Offspring = GAlabel(Parent1,Parent2,proC,proM)
% Genetic operators for label variables

    %% Uniform crossover
    [N,D] = size(Parent1);
    k     = rand(N,D) < 0.5;
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring    = Parent1;
    Offspring(k) = Parent2(k);
    
    %% Bitwise mutation
    Site = rand(N,D) < proM/D;
    Rand = randi(D,N,D);
    Offspring(Site) = Rand(Site);
    
    %% Repair
    for i = 1 : N
        Off = zeros(1,D);
        while ~all(Off)
            x = find(~Off,1);
            Off(Offspring(i,:)==Offspring(i,x)) = max(Off) + 1;
        end
        Offspring(i,:) = Off;
    end
end

function Offspring = GAbinary(Parent1,Parent2,proC,proM)
% Genetic operators for binary variables

    %% Uniform crossover
    [N,D] = size(Parent1);
    k     = rand(N,D) < 0.5;
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring    = Parent1;
    Offspring(k) = Parent2(k);
    
    %% Bit-flip mutation
    Site = rand(N,D) < proM/D;
    Offspring(Site) = ~Offspring(Site);
end

function Offspring = GApermutation(Parent1,Parent2,proC)
% Genetic operators for permutation variables

    %% Order crossover
    [N,D]     = size(Parent1);
    Offspring = Parent1;
    k         = randi(D,1,N);
    for i = 1 : N
        if rand < proC
            Offspring(i,k(i)+1:end) = setdiff(Parent2(i,:),Parent1(i,1:k(i)),'stable');
        end
    end
    
    %% Slight mutation
    k = randi(D,1,N);
    s = randi(D,1,N);
    for i = 1 : N
        if s(i) < k(i)
            Offspring(i,:) = Offspring(i,[1:s(i)-1,k(i),s(i):k(i)-1,k(i)+1:end]);
        elseif s(i) > k(i)
            Offspring(i,:) = Offspring(i,[1:k(i)-1,k(i)+1:s(i)-1,k(i),s(i):end]);
        end
    end
end
```

### `SubPopRank.m`
```matlab
function [FNmean,FNbest] = SubPopRank(Populations)
% Calculate the mean and best front number of each subpopulation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    Population = [Populations{:}];
    K          = length(Populations);
    Flag       = [];
    j          = 1;
    for i = 1 : K
        Flag(j:j+length(Populations{i})-1) = i;
        j = j + length(Populations{i});
    end
    FrontNoAll = NDSort(Population.objs,inf);
    FNmean     = zeros(1,K);
    FNbest     = zeros(1,K);
    for i = 1 : K
        FNmean(i) = mean(FrontNoAll(Flag==i));
        FNbest(i) = min(FrontNoAll(Flag==i));
    end
end
```

### `SubPopSimility.m`
```matlab
function [Populations2,Masks2,Decs2,GV2,numK] = SubPopSimility(Populations,Masks,Decs,GV)
% Calculate the similarity between subpopulations

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    K  = length(Populations);
    fs = zeros(1,K);
    for m  = 1 : K
        [~,fs(m)] = min(mean(Populations{m}.objs,2));
    end
    ss    = 0;
    flagmerge=zeros(K,1);
    for i = 1 : K-1
        if flagmerge(i,:)==0||flagmerge(i,:)==i
            for j = i+1 : K
                if flagmerge(j,:)==0
                        ss    = simility(Masks{i}(fs(i),:),Masks{j}(fs(j),:));
                        if ss>0.7
                            flagmerge(i,:)=i;
                            flagmerge(j,:)=i;
                        end
                else
                    continue;
                end
            end
        else
            continue;
        end
    end
    a    = find(flagmerge==0);
    b    = max(flagmerge);
    numK = size(a,1) + b;
    for k = unique(flagmerge)'
        temP2 = [];
        temD2 = [];
        temM2 = [];
        temG2 = [];
        current = find(flagmerge==k);
        if k ~= 0
            for h = 1 : size(current,1)
                temP  = Populations{current(h)};
                temP2 = [temP2,temP];
                temD  = Decs{current(h)};
                temD2 = [temD2;temD];
                temM  = Masks{current(h)};
                temM2 = [temM2;temM];
                temG  = GV{current(h)};
                temG2 = [temG2;temG];
            end
            Populations2{k+size(a,1)} = temP2;
            Decs2{k+size(a,1)}        = temD2;
            Masks2{k+size(a,1)}       = temM2;
            GV2{k+size(a,1)}          = temG2;
        else
            for h = 1 : size(current,1)
                temP = Populations{current(h)};
                temD = Decs{current(h)};
                temM = Masks{current(h)};
                temG = GV{current(h)};
                Populations2{h} = temP;
                Decs2{h} = temD;
                Masks2{h}= temM;
                GV2{h}   = temG;
            end
        end
    end
    Populations2(cellfun(@isempty,Populations2)) = [];
    Decs2(cellfun(@isempty,Decs2))   = [];
    Masks2(cellfun(@isempty,Masks2)) = [];
    GV2(cellfun(@isempty,GV2)) = [];
    numK = size(Populations2,2);
end

function s = simility(subPop1,subPop2)
	s = sum(subPop1&subPop2)/min(sum(subPop1),sum(subPop2));
end
```

### `UpdateGV.m`
```matlab
function gv = UpdateGV(gv,Mask,FrontNo)
% Update the guiding vectors

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    Mask = Mask(FrontNo==1,:);
    k    = ceil(0.1*size(Mask,1));
    v    = zeros(1,size(Mask,2));
    for i = 1 : size(Mask,1)
        rand_Mask = Mask(i,:);          
        dis       = pdist2(rand_Mask,Mask,'hamming');
        [~,index] = sort(dis);
        knearest_Mask = Mask(index(1:k),:);
        rand_Mask(rand_Mask==1) = 0.5 ;
        knearest_Mask(knearest_Mask==1) = 0.5 ;
        v = v + sum(repmat(rand_Mask,k,1)+knearest_Mask,1)/(size(Mask,1)*k);
    end
    if all(gv==0)
        gv = v;
    else
        gv = 0.9*gv + 0.1*v;
    end
end
```
