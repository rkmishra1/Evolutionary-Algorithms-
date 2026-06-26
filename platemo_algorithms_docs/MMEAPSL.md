# MMEAPSL

**Tags**: <2024> <multi> <real/integer/label/binary/permutation> <multimodal>

## Description
Multimodal multi-objective evolutionary algorithm assisted by Pareto set learning

## Reference
F. Ming, W. Gong, and Y. Jin. Growing neural gas network-based surrogate-assisted Pareto set learning for multimodal multi-objective optimization. Swarm and Evolutionary Computation, 2024, 87: 101541.

## Source Code

### `CalFitness.m`
```matlab
function [D_Dec,D_Pop,Fitness] = CalFitness(PopObj,PopDec)
% Calculate the fitness of each solution based on original objective space

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

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
    D_Pop = 1./(Distance(:,floor(sqrt(N)))+2);
    
    Distance = pdist2(PopDec,PopDec);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D_Dec = 1./(Distance(:,floor(sqrt(N)))+2);

    D = D_Pop + D_Dec;
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFitnessSup.m`
```matlab
function D_Dec = CalFitnessSup(PopDec,V)
% Calculate the fitness of each solution of the sup population
    
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    %% Calculate D(i)    
    Distance = pdist2(PopDec,V);
    Distance = sort(Distance,2);
    D_Dec    = Distance(:,1);
end
```

### `EnvironmentalSelectionOrig.m`
```matlab
function [Population,Fitness,D_Dec,D_Pop,net,genFlag] = EnvironmentalSelectionOrig(Population,N,Problem,params,net,genFlag)
% The environmental selection of SPEA2 based on objective space

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    [FrontNo,~] = NDSort(Population.objs,N);
    valid  = FrontNo == 1;
    NDPop  = Population(valid);
    temp1  = NDPop.decs;
    gen    = ceil(Problem.FE/Problem.N);
    maxgen = ceil(Problem.maxFE/Problem.N);
    if size(temp1,1) > 2&&isempty(find(isnan(temp1)==true))&& gen <= round(1*maxgen) && isempty(genFlag)
        [net,genFlag] = TrainGrowingGasNet(temp1,net,params,Problem,genFlag);
    end
    
     %% Calculate the fitness of each solution
    [D_Dec,D_Pop,Fitness] = CalFitness(Population.objs,Population.decs);
    
    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,Population(Next).decs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
    D_Dec      = D_Dec(Next);
    D_Pop      = D_Pop(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population     = Population(rank);
    D_Dec          = D_Dec(rank);
    D_Pop          = D_Pop(rank);
end

function Del = Truncation(PopObj,PopDec,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance_Pop = pdist2(PopObj,PopObj);
    Distance_Pop(logical(eye(length(Distance_Pop)))) = inf;
    
    Distance_Dec = pdist2(PopDec,PopDec);
    Distance_Dec(logical(eye(length(Distance_Dec)))) = inf;
    
    D = Distance_Pop + Distance_Dec;
    
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(D(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `EnvironmentalSelectionSup.m`
```matlab
function [Population,Fitness] = EnvironmentalSelectionSup(Population,N,net)
% The environmental selection of sup population using net

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    %% Calculate max min distance between reference points in V
    V = net.w;
    C = net.C;
    Distance = pdist2(V,V);
    Distance = Distance.*~C;
    d = zeros(1,length(V));
    for i = 1:length(V)
        Dis = Distance(i,:);
        Dis(Dis==0) = [];
        d(i) = min(Dis);
    end
    theta = max(d);
    
    %% Calculate the fitness of each solution
    Fitness = CalFitnessSup(Population.decs,V);
    
    %% Environmental selection
    Next = Fitness < theta;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).decs,sum(Next)-N);
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

function Del = Truncation(PopDec,K)
% Select part of the solutions by truncation

    %% Truncation
    D = pdist2(PopDec,PopDec);
    D(logical(eye(length(D)))) = inf;
    
    Del = false(1,size(PopDec,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(D(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `InitilizeGrowingGasNet.m`
```matlab
function net = InitilizeGrowingGasNet(Population,params,Fitness)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    N         = params.N;
    MaxIt     = params.MaxIt;
    L         = params.L;
    epsilon_b = params.epsilon_b;
    epsilon_n = params.epsilon_n;
    alpha     = params.alpha;
    delta     = params.delta;
    T         = params.T;
    
    % Choose solutions for initializing the GNG network net
    FrontNo = NDSort(Population.objs,N);
    valid   = FrontNo == 1;
    RefSize = sum(valid);
    if RefSize > 2
        NDPop = Population(valid);
    else
        NDPop   = Population(Fitness<1);
        RefSize = length(Population);
    end
    PopDecs = NDPop.decs;
    [NP,M]  = size(PopDecs);
    
    %% Initialization
    Ni = 2;
    if RefSize >= 2
        w = PopDecs(randperm(RefSize,Ni),:);
    else
        w = PopDecs;
    end
    
    E  = zeros(Ni,1);
    C  = zeros(Ni, Ni);
    t  = zeros(Ni, Ni);
    nx = 0;
    
    %% Loop
    for it = 1 : MaxIt
        for kk = 3 : RefSize
            % Select Input
            nx = nx + 1;
            x  = PopDecs(kk,:);

            % Competion and Ranking
            d = pdist2(x, w);
            [~, SortOrder] = sort(d);
            s1 = SortOrder(1);
            s2 = SortOrder(2);

            % Aging
            t(s1, :) = t(s1, :) + 1;
            t(:, s1) = t(:, s1) + 1;

            % Add Error
            E(s1) = E(s1) + d(s1)^2;

            % Adaptation
            w(s1,:) = w(s1,:) + epsilon_b*(x-w(s1,:));
            Ns1 = find(C(s1,:)==1);
            for j = Ns1
                w(j,:) = w(j,:) + epsilon_n*(x-w(j,:));
            end

            % Create Link
            C(s1,s2) = 1;
            C(s2,s1) = 1;
            t(s1,s2) = 0;
            t(s2,s1) = 0;

            % Remove Old Links
            C(t>T)     = 0;
            nNeighbor  = sum(C);
            AloneNodes = (nNeighbor==0);
            C(AloneNodes, :) = [];
            C(:, AloneNodes) = [];
            t(AloneNodes, :) = [];
            t(:, AloneNodes) = [];
            w(AloneNodes, :) = [];
            E(AloneNodes)    = [];

            % Add New Nodes
            if mod(nx, L) == 0 && size(w,1) < N
                [~, q]  = max(E);
                [~, f]  = max(C(:,q).*E);
                r       = size(w,1) + 1;
                w(r,:)  = (w(q,:) + w(f,:))/2;
                C(q,f)  = 0;
                C(f,q)  = 0;
                C(q,r)  = 1;
                C(r,q)  = 1;
                C(r,f)  = 1;
                C(f,r)  = 1;
                t(r,:)  = 0;
                t(:, r) = 0;
                E(q)    = alpha*E(q);
                E(f)    = alpha*E(f);
                E(r)    = E(q);
            end

            % Decrease Errors
            E = delta*E;
        end
    end

    for ii = 1:size(w,1)
        ageSum(ii,:) = sum(t(ii,find(C(ii,:) == 1),:),2);
        ageSumBefore = ageSum;
        flag(ii,:)   = 0;
    end
    
    net.w  = w;
    net.E  = E;
    net.C  = C;
    net.t  = t;
    net.nx = nx;
    net.ageSumBefore = ageSumBefore;
    net.flag         = flag;
end
```

### `MMEAPSL.m`
```matlab
classdef MMEAPSL < ALGORITHM
% <2024> <multi> <real/integer/label/binary/permutation> <multimodal>
% Multimodal multi-objective evolutionary algorithm assisted by Pareto set learning

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, and Y. Jin. Growing neural gas network-based
% surrogate-assisted Pareto set learning for multimodal multi-objective
% optimization. Swarm and Evolutionary Computation, 2024, 87: 101541.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    methods
        function main(Algorithm,Problem)
            %% Parameters for GNG network
            params.N         = Problem.N;
            params.MaxIt     = 50;
            params.L         = 30;
            params.epsilon_b = 0.2;
            params.epsilon_n = 0.006;
            params.alpha     = 0.5;
            params.delta     = 0.995;
            params.T         = 30;
            genFlag          = [];
            MaxGen           = ceil(Problem.maxFE/Problem.N);
            netInitialized   = 0;
            Pop1get          = 0;
            gen              = 0;
            
            %% Generate random population
            Population = Problem.Initialization();
            
            %% calculate fitness of populations
            [Fitness,D_Dec,~] = CalFitness(Population.objs,Population.decs);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                gen = gen + 1;
                %% Initial the GNG network when can
                if ~netInitialized
                    NDNum = sum(Fitness<1);
                    if NDNum >= 2
                        net = InitilizeGrowingGasNet(Population,params,Fitness);
                        netInitialized = 1;
                    end
                end
                
                if ~netInitialized || gen < 0.2 * MaxGen
                    MatingPool = TournamentSelection(2,Problem.N,D_Dec,Fitness);
                    Offspring  = OperatorGA(Problem,Population(MatingPool));
                    [Population,Fitness,D_Dec,~,net,genFlag] = EnvironmentalSelectionOrig([Population,Offspring],Problem.N,Problem,params,net,genFlag);
                else
                    if Pop1get == 0
                        Population1 = Population;
                        Fitness1    = CalFitnessSup(Population1.decs,net.w);
                        Pop1get     = 1;
                    end
                     % use the network in generating offspring
                    MatingPool1 = TournamentSelection(2,Problem.N,D_Dec,Fitness);
                    Offspring1  = OperatorGAhalf(Problem,Population(MatingPool1));
                    V           = net.w;
                    MatingPool2 = randi(length(V),1,Problem.N);
                    Offspring2  = Problem.Evaluation(OperatorGAhalf(Problem,V(MatingPool2,:)));
                    
                    MatingPool1 = TournamentSelection(2,Problem.N,-Fitness1);
                    Offspring3  = OperatorGAhalf(Problem,Population1(MatingPool1));
                    
                    Offspring = [Offspring1,Offspring2,Offspring3];
                    
                    [Population,Fitness,D_Dec,~,net,genFlag] = EnvironmentalSelectionOrig([Population,Offspring],Problem.N,Problem,params,net,genFlag);
                    [Population1,Fitness1] = EnvironmentalSelectionSup([Population1,Offspring],Problem.N,net);
                end
            end
        end
    end
end
```

### `TrainGrowingGasNet.m`
```matlab
function [net,genFlag] = TrainGrowingGasNet(temp1,net,params,Problem,genFlag)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    %% Parameters
    N = params.N;
    L = params.L;
    epsilon_b = params.epsilon_b;
    epsilon_n = params.epsilon_n;
    alpha = params.alpha;
    delta = params.delta;
    T     = params.T;
    
    C  = net.C;
    w  = net.w;
    t  = net.t;
    E  = net.E;
    nx = net.nx;
    ageSumBefore = net.ageSumBefore;
    flag         = net.flag;
    
    for i = 1 : size(w,1)
        neighbor    = find(C(i,:)==1);
        ageSum(i,:) = sum(t(i,neighbor,:),2);
        if ageSum(i,:) == ageSumBefore(i,:)
            flag(i,:) = flag(i,:) + 1;
        end
    end

    ageSumBefore = ageSum;
    maxN         = 1.5;

    gen    = ceil(Problem.FE/Problem.N);
    maxgen = ceil(Problem.maxFE/Problem.N);
    if gen <= round(0.9*maxgen)
        maxIter = 1;
        maxPZ   = maxN;
        if size(w,1) == round(maxN*N)
            [~,rankFlag] = sort(flag,'descend');
            r = rankFlag(1:round(maxN*N)-N);
            C(r, :) = [];
            C(:, r) = [];
            t(r, :) = [];
            t(:, r) = [];
            w(r, :) = [];
            E(r)    = [];
            ageSumBefore(r,:) = [];
            flag(r,:) = [];
            flag      = zeros(N,1);
        end
    else
        if size(w,1) < round(maxN*N) && isempty(genFlag)
            maxPZ   = maxN;
            maxIter = 1;
        else
            maxPZ   = 1;
            maxIter = 0;
        end
        if size(w,1) == round(maxN*N)
            maxPZ   = 1;
            maxIter = 0;
            genFlag = gen;
        end
    end

    if isempty(genFlag)
        for iter = 1 : maxIter
            for kk = 1 : size(temp1,1)
                nx = nx + 1;
                x  = temp1(kk,:);
                d  = pdist2(x, w);
                [~, SortOrder] = sort(d);
                s1 = SortOrder(1);
                s2 = SortOrder(2);

                % Aging: the age of all neighbours of s1 is increased by 1

                t(s1, :) = t(s1, :) + 1;
                t(:, s1) = t(:, s1) + 1;

                % Add Error
                E(s1) = E(s1) + d(s1)^2;

                % Adaptation
                w(s1,:) = w(s1,:) + epsilon_b*(x-w(s1,:));
                Ns1     = find(C(s1,:)==1);
                for j = Ns1
                    w(j,:) = w(j,:) + epsilon_n*(x-w(j,:));
                end

                % Create Link
                C(s1,s2) = 1;
                C(s2,s1) = 1;
                t(s1,s2) = 0;
                t(s2,s1) = 0;

                % Remove Old Links
                C(t>T)     = 0;
                nNeighbor  = sum(C);
                AloneNodes = (nNeighbor==0);
                if ~isempty(find(AloneNodes == true))
                    %         AloneNodes
                end
                C(AloneNodes, :) = [];
                C(:, AloneNodes) = [];
                t(AloneNodes, :) = [];
                t(:, AloneNodes) = [];
                w(AloneNodes, :) = [];
                E(AloneNodes)    = [];
                ageSumBefore(AloneNodes,:) = [];
                flag(AloneNodes,:)         = [];

                % Add New Nodes
                if mod(nx, L) == 0 && size(w,1) < round(maxPZ*N)
                    [~, q]  = max(E);
                    [~, f]  = max(C(:,q).*E);
                    r       = size(w,1) + 1;
                    w(r,:)  = (w(q,:) + w(f,:))/2;
                    C(q,f)  = 0;
                    C(f,q)  = 0;
                    C(q,r)  = 1;
                    C(r,q)  = 1;
                    C(r,f)  = 1;
                    C(f,r)  = 1;
                    t(r,:)  = 0;
                    t(:, r) = 0;
                    E(q)    = alpha*E(q);
                    E(f)    = alpha*E(f);
                    E(r)    = E(q);
                    ageSumBefore(r,:) = 0;
                    flag(r,:)         = 0;
                    
                    if mod(nx, 2*L) == 0 && size(w,1) < round(maxPZ*N)
                        for i = 1:size(w,1)
                            edgeSum(i) = sum(C(i,:)) + sum(C(:,i));
                        end
                        
                        [~, q] = min(edgeSum);
                        
                        D = pdist2(w,w,'cityblock');
                        D(logical(eye(length(D)))) = inf;
                        D = D(q,:);
                        D(C(q,:)==1) = inf;
                        D(C(:,q)==1) = inf;
                        [~, f] = min(D);

                        
                        r       = size(w,1) + 1;
                        w(r,:)  = (w(q,:) + w(f,:))/2;
                        C(q,f)  = 0;
                        C(f,q)  = 0;
                        C(q,r)  = 1;
                        C(r,q)  = 1;
                        C(r,f)  = 1;
                        C(f,r)  = 1;
                        t(r,:)  = 0;
                        t(:, r) = 0;
                        E(q)    = alpha*E(q);
                        E(f)    = alpha*E(f);
                        E(r)    = E(q);
                        ageSumBefore(r,:) = 0;
                        flag(r,:)         = 0;
                    end
                end

                % Decrease Errors
                E = delta*E;
            end
        end
        net.w  = w;
        net.E  = E;
        net.C  = C;
        net.t  = t;
        net.nx = nx;
        net.ageSumBefore = ageSumBefore;
        net.flag         = flag;
    end
end
```
