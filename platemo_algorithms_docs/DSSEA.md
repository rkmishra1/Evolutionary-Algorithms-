# DSSEA

**Tags**: <2025> <multi/many> <real/integer> <large> <constrained>

## Description
Dynamic subspace search-based evolutionary algorithm

## Reference
X. Ban, J. Liang, K. Yu, B. Qu, K. Qiao, P. N. Suganthan, and Y. Wang. A subspace search-based evolutionary algorithm for large-scale constrained multi-objective optimization and application. IEEE Transactions on Cybernetics, 2025, 55(5): 2486-2499.

## Source Code

### `ALDVA.m`
```matlab
function rank_DV = ALDVA(Problem,Population,nSel,nPer)
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

    [N,D] = size(Population.decs);
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end
    
    %% Calculate the proper values of each decision variable
    Angle = zeros(D,nSel);
    RMSE  = zeros(D,nSel);
    
    for i = 1 : D
        drawnow('limitrate');
        Sample    = randi(N,1,nSel);    % Randomly select nSel solutions for perturbation
        % Generate several random solutions by perturbing the i-th dimension
        Decs      = repmat(Population(Sample).decs,nPer,1);
        Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
        newPopu   = Problem.Evaluation(Decs);
        for j = 1 : nSel
            % Normalize the objective values of the current perturbed solutions
            Points      = newPopu(j:nSel:end).objs;
            After_sort  = sortrows(Points);
            Length(i,j) = norm(After_sort(1,:) - After_sort(end,:));%% Calculate the length after perturbation
            Points      = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
            Points      = Points - repmat(mean(Points,1),nPer,1);%%
            % Calculate the direction vector of the determining line
            [~,~,V] = svd(Points);
            Vector  = V(:,1)'./norm(V(:,1)');
            % Calculate the root mean square error
            error = zeros(1,nPer);
            for k = 1 : nPer
                error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
            end
            RMSE(i,j)  = sqrt(sum(error.^2));
            % Calculate the angle between the line and the hyperplane
            normal     = ones(1,size(Vector,2));
            sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
            Angle(i,j) = real(asin(sine)/pi*180);%% asin: the larger it is, the more convergent it is
        end
    end

    Length = mean(Length,2);
    Angle  = mean(Angle,2);
    Length = (Length-min(Length))/(max(Length)-min(Length));
    Angle  = (Angle-min(Angle))/(max(Angle)-min(Angle));

    R           = Angle + Length;
    [~,rank_DV] = sort(R,'descend');
    [~,rank_DV] = sort(rank_DV);
    rank_DV     = rank_DV';
end
```

### `DSSEA.m`
```matlab
classdef DSSEA < ALGORITHM
% <2025> <multi/many> <real/integer> <large> <constrained>
% Dynamic subspace search-based evolutionary algorithm

%------------------------------- Reference --------------------------------
% X. Ban, J. Liang, K. Yu, B. Qu, K. Qiao, P. N. Suganthan, and Y. Wang. A
% subspace search-based evolutionary algorithm for large-scale constrained
% multi-objective optimization and application. IEEE Transactions on
% Cybernetics, 2025, 55(5): 2486-2499.
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
            %% Parameter setting
            [nSel,nPer] = Algorithm.ParameterSet(2,4);
    
            %% Generate random population
            Population = Problem.Initialization();
            Rank_DV   = ALDVA(Problem,Population,nSel,nPer);
    
            cons     = Population.cons;
            cons(cons<=0) = 0;
            conss    = sum(cons,2);
            epsilon0 = sum(conss);
    
            if epsilon0 == 0
                epsilon0 = 100;
            end
    
            Fitness = CalFitness_E(Population.objs,Population.cons,epsilon0);
            X = 0;
    
            %% Optimization
            while Algorithm.NotTerminated(Population)
                cp = (-log(epsilon0)-6)/log(1-0.75);
                epsilon = epsilon0*(1-X)^cp;
    
                IP = (Rank_DV-length(Rank_DV))/(1-length(Rank_DV));
                OP = (1-IP)*tanh(9*Problem.FE/Problem.maxFE)+IP;
    
                rand_num = rand(1,length(Rank_DV));
                DV_OP    = (rand_num<=OP);
                DV_NOP   = (rand_num>OP);
    
                if rand <= 0.5
                    Offspring = OperatorDE_pbest_1_main_OP(Population, Problem.N, Problem, Fitness, DV_OP, DV_NOP, 0.1);
                else
                    MatingPool1 = TournamentSelection(2,Problem.N,Fitness);
                    Offspring   = OperatorGA_OP(Problem,Population, Population(MatingPool1),DV_OP,DV_NOP);
                end
    
                [Population,Fitness] = EnvironmentalSelection([Population,Offspring],Problem.N,epsilon,true);
    
                X = X + 1/(Problem.maxFE/Problem.N);
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection(Population,N,epsilon,isOrigin)
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
        Fitness = CalFitness_E(Population.objs,Population.cons,epsilon);
    else
        Fitness = CalFitness_E(Population.objs);
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
        evaluated = true;
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

### `OperatorDE_pbest_1_main_OP.m`
```matlab
function [ Offspring ] = OperatorDE_pbest_1_main_OP(Population, popsize, Problem, Fitness, DV_OP, DV_NOP, p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    permutation = randperm(Problem.N);
    r0          = permutation;
    [r1,r2,~]   = gnR1R2R3(Problem.N, r0);
    
    array  = permutation(1:popsize);
    OffDec = Population(array).decs;
    
    [~,indBest] = sort(Fitness, 'ascend');
    pNP         = max(round(p * Problem.N), 2);     % choose at least two best solutions
    randindex   = ceil(rand(1, popsize) * pNP);     % select from [1, 2, 3, ..., pNP]
    randindex   = max(1, randindex);                % to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest       = Population(indBest(randindex));   % randomly choose one of the top 100p% solutions
    
    NewDec = OperatorDE_pbest_1(Problem,Population(array).decs,pbest.decs,Population(r1(1:popsize)).decs,Population(r2(1:popsize)).decs);
    OffDec(:,DV_OP) = NewDec(:,DV_OP);
    
    %% DV_NOP: Randomly select individuals to assign dimensional information
    if sum(DV_NOP)>1
        rand_ind = randi([1 popsize],popsize,length(DV_NOP));
        Decc     = Population.decs;
        for h = 1 : popsize
            Ori = Decc(rand_ind(h,:), DV_NOP);
            Oth(h,:) = diag(Ori);
        end
        OffDec(:,DV_NOP) = Oth;
    end
    Offspring = OffDec;
    Offspring = Problem.Evaluation(Offspring);
end
```

### `OperatorGA_OP.m`
```matlab
function Offspring = OperatorGA_OP(Problem,Population, Parent,DV_OP, DV_NOP, Parameter)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    if nargin > 5
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

    OffDec = Parent;
    OffDec(:,DV_OP) = Offspring(:,DV_OP);
    
    %% DV_NOP: Randomly select individuals to assign dimensional information
    popsize = Problem.N;
    if sum(DV_NOP)>1
        rand_ind = randi([1 popsize],popsize,length(DV_NOP));
        Decc = Population.decs;
        for h=1:popsize
            Ori = Decc(rand_ind(h,:), DV_NOP);
            Oth(h,:) = diag(Ori);
        end
        OffDec(:,DV_NOP) = Oth;
    end
    Offspring = OffDec;
    if evaluated
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
