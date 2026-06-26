# RVEA-iGNG

**Tags**: <2022> <multi/many> <real/integer/label/binary/permutation>

## Description
RVEA based on improved growing neural gas

## Reference
Q. Liu, Y. Jin, M. Heiderich, T. Rodemann, and G. Yu. An adaptive reference vector-guided evolutionary algorithm using growing neural gas for many-objective optimization of irregular problems. IEEE Transactions on Cybernetics, 2022, 52(5): 2698-2711.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,net,V,Archive,scale,genFlag] = EnvironmentalSelection(Population,V,theta,net,params,Archive,Problem,scale,zmin,genFlag)
% The environmental selection of RVEA

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qiqi Liu

    Population = Population(NDSort(Population.objs,1)==1);
    PopObj = Population.objs;
    [N,M]  = size(PopObj);

    %% Translate the population
    PopObj = PopObj - repmat(zmin,N,1);

    %% delete the outliers
    d      = sqrt(sum(PopObj.^2,2));
    meanD  = sum(d,1)/size(PopObj,1);
    delete = find(d>10*meanD);
    Population(delete) = [];
    SavePopObj         = Population.objs;


    Archive = UpdateArchive(Population,Archive,2*Problem.N);
    ArcObj  = Archive.objs;

    wholeObj   = [SavePopObj;ArcObj];
    Population = [Population Archive];
    [c,ia,ic]  = unique(wholeObj,'rows');
    Population = Population(ia);
    wholeObj   = wholeObj(ia,:);
    wholeObj   = wholeObj - repmat(zmin,size(wholeObj,1),1);



    wholeObj1 = wholeObj./scale;
    temp1     = wholeObj1./sum(wholeObj1,2);


    PopObj = wholeObj;
    [N,M]  = size(PopObj);
    fr     = 0.1;

    gen    = ceil(Problem.FE/Problem.N);
    maxgen = ceil(Problem.maxFE/Problem.N);
    if ~mod(gen,ceil(fr*maxgen))&& gen <= round(1*maxgen)
        scale = max(ArcObj,[],1)-min(ArcObj,[],1);
    end

    if size(temp1,1) > 2&&isempty(find(isnan(temp1)==true))&& gen <= round(1*maxgen) && isempty(genFlag)
        [V,net,genFlag] = TrainGrowingGasNet(V,temp1,net,scale,params,Problem,[[];ArcObj],genFlag,zmin);
    end
    NV = size(V,1);

    %% Calculate the degree of violation of each solution
    CV = sum(max(0,Population.cons),2);

    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current1 = find(associate==i & CV==0);
        current2 = find(associate==i & CV~=0);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD = (1+M*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
            % Select the one with the minimum APD value
            [~,best] = min(APD);
            Next(i)  = current1(best);
        elseif ~isempty(current2)
            % Select the one with the minimum CV value
            [~,best] = min(CV(current2));
            Next(i)  = current2(best);
        end
    end

    %% select the corner solutions in each generation
    fm = [];
    selectedFirst = unique([Next(Next~=0) fm]);
    Population    = [Population(selectedFirst)];

    if length(Population) > Problem.N 
        PopObj = Population.objs;
        PopObj = PopObj - repmat(zmin,size(PopObj,1),1);

        temp1 = PopObj;

        Choose           = false(1,size(temp1,1));
        [~,Extreme1]     = min(temp1,[],1);
        [~,Extreme2]     = max(temp1,[],1);
        Choose(Extreme1) = true;
        Choose(Extreme2) = true;  

        while sum(Choose) < Problem.N
            ind   = find(Choose== false);
            choId = setdiff(1:size(temp1,1),ind); 
            PopObj1temp   = temp1(choId,:);     
            WholeObjtemp  = temp1(ind,:);
            dis   = pdist2(WholeObjtemp,PopObj1temp);
            [mindis,]     = min(dis,[],2);
            [~,associate] = max(mindis,[],1);
            Choose(ind(associate)) = true;
        end
        Population = Population(Choose);
    end
end
```

### `InitilizeGrowingGasNet.m`
```matlab
function net = InitilizeGrowingGasNet(V,Population,params)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qiqi Liu

    N     = params.N;
    MaxIt = params.MaxIt;
    L     = params.L;
    epsilon_b = params.epsilon_b;
    epsilon_n = params.epsilon_n;
    alpha     = params.alpha;
    delta     = params.delta;
    T         = params.T;

    PopObj  = Population.objs;
    [NP,M]  = size(PopObj);
    PopObj  = PopObj - repmat(min(PopObj,[],1),NP,1);
    Angle   = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);
    valid         = unique(associate);
    RefSize       = size(valid,1);

    %% Initialization
    Ni = 2;
    w  = zeros(Ni, M);
    if RefSize >= 2
        for i = 1 : Ni
            w(i,:) = V(valid(i,:),:);
        end
    else
        w(1:Ni,:) = V(randperm(N,Ni),:);
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
            x  = V(valid(kk,:),:);

            % Competion and Ranking
            d  = pdist2(x, w);
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

    for ii = 1 : size(w,1)
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

### `RVEAiGNG.m`
```matlab
classdef RVEAiGNG < ALGORITHM
% <2022> <multi/many> <real/integer/label/binary/permutation>
% RVEA based on improved growing neural gas
% alpha --- 2 --- The parameter controlling the rate of change of penalty

%------------------------------- Reference --------------------------------
% Q. Liu, Y. Jin, M. Heiderich, T. Rodemann, and G. Yu. An adaptive
% reference vector-guided evolutionary algorithm using growing neural gas
% for many-objective optimization of irregular problems. IEEE Transactions
% on Cybernetics, 2022, 52(5): 2698-2711.
%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qiqi Liu

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            params.N         = Problem.N;
            params.MaxIt     = 50;
            params.L         = 50;
            params.epsilon_b = 0.2;
            params.epsilon_n = 0.006;
            params.alpha     = 0.5;
            params.delta     = 0.995;
            params.T         = 50;

            alpha         = Algorithm.ParameterSet(2);
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();
            net           = InitilizeGrowingGasNet(V,Population,params);
            Archive       = UpdateArchive(Population,[],Problem.N);
            scale         = ones(1,Problem.M);
            zmin          = min(Population.objs,[],1);
            genFlag       = [];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = randi(length(Population),1,Problem.N);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                zmin       = min([zmin;Offspring.objs],[],1); 
                [Population,net,V,Archive,scale,genFlag] = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^alpha,net,params,Archive,Problem,scale,zmin,genFlag);
            end
        end
    end
end
```

### `TrainGrowingGasNet.m`
```matlab
function [V,net,genFlag] = TrainGrowingGasNet(V,temp1,net,scale,params,Problem,wholeObj,genFlag,zmin)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qiqi Liu

    %% Parameters
    N = params.N;
    L = params.L;
    epsilon_b = params.epsilon_b;
    epsilon_n = params.epsilon_n;
    alpha = params.alpha;
    delta = params.delta;
    T     = params.T;

    t  = net.t;
    E  = net.E;
    nx = net.nx;
    ageSumBefore = net.ageSumBefore;
    flag = net.flag;

    % select the corner solutions using GNG
    output = [];
    C      = net.C;
    w      = net.w;
    for i = 1 : size(w,1)
        neighbor = find(C(i,:)==1);

        ageSum(i,:) = sum(t(i,neighbor,:),2);
        if ageSum(i,:) == ageSumBefore(i,:)
            flag(i,:) = flag(i,:) + 1;
        end

    end

    cw = w(output,:);

    ageSumBefore = ageSum;
    maxN         = 1.5;

    gen    = ceil(Problem.FE/Problem.N);
    maxgen = ceil(Problem.maxFE/Problem.N);    
    if gen <= round(0.9*maxgen)
        maxIter = 1;
        maxPZ   = maxN;
        if size(w,1) == round(maxN*N)
            [~,rankFlag] = sort(flag,'descend');
            r       = rankFlag(1:round(maxN*N)-N);
            C(r, :) = [];
            C(:, r) = [];
            t(r, :) = [];
            t(:, r) = [];
            w(r, :) = [];
            E(r)    = [];
            ageSumBefore(r,:) = [];
            flag = zeros(N,1);
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
                w  = w./sum(w,2);
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
                end

                % Decrease Errors
                E = delta*E;
            end
        end

        w      = w./sum(w,2);
        net.w  = w;
        net.E  = E;
        net.C  = C;
        net.t  = t;
        net.nx = nx;
        net.ageSumBefore = ageSumBefore;
        net.flag         = flag;
    end

    if isempty(genFlag)
        output = [];
        for i = 1 : size(w,1)
            neighbor   = find(C(i,:)==1);
            [~,minInd] = min(w([i neighbor],:),[],1);
            [~,maxInd] = max(w([i neighbor],:),[],1);
            if ~isempty(find(minInd==1))
                output = [output i];
            elseif ~isempty(find(maxInd==1))
                output = [output i];
            end
        end

        for t = 1 : size(output,2)
            s = output(:,t);
            x = mean(w(find(C(s,:)==1),:),1);
            w(s,:) = w(s,:) - 1*(x-w(s,:));
        end

        t = any(w<0,2);
        l = find(t == true);
        if ~isempty(l)
            for tt = 1:size(l,1)
                w(l(tt,1),any(w(l(tt,1),:)<0,1)) = 0;
            end
        end

        V = w;
        V = V.*repmat(scale,size(V,1),1);
    end

    if gen == genFlag
        N         = size(wholeObj,1);
        wholeObj1 = (wholeObj - repmat(zmin,N,1));
        temp2     = wholeObj1./sum(wholeObj1,2);

        output = [];
        for i = 1 : size(w,1)
            neighbor   = find(C(i,:)==1);
            [~,minInd] = min(w([i neighbor],:),[],1);
            [~,maxInd] = max(w([i neighbor],:),[],1);
            if ~isempty(find(minInd==1))
                output = [output i];
            elseif ~isempty(find(maxInd==1))
                output = [output i];
            end
        end

        for t = 1 : size(output,2)
            s = output(:,t);
            x = mean(w(find(C(s,:)==1),:),1);
            w(s,:) = w(s,:) - 1*(x-w(s,:));
        end

        t = any(w<0,2);
        l = find(t == true);
        if ~isempty(l)
            for tt = 1 : size(l,1)
                w(l(tt,1),any(w(l(tt,1),:)<0,1)) = 0;
            end
        end

        V = [w.*repmat(scale,size(w,1),1);temp2];
    end
end
```

### `Truncation.m`
```matlab
function Population = Truncation(Population,MaxSize)
% Limit the size of final popualtion in RVEA*

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qiqi Liu

    PopObj = Population.objs;
    zmax   = max(PopObj,[],1);
    zmin   = min(PopObj,[],1);

    PopObj = (PopObj - repmat(zmin,size(PopObj,1),1))./(zmax-zmin);
    M      = size(PopObj,2);
    H      = [eye(M-1)-ones(M-1)/M;-ones(1,M-1)/M];
    Pe     = H*inv(H'*H)*H';
    f      = PopObj*Pe';
    minf   = min(f,[],1);
    maxf   = max(f,[],1);
    f      = (f-minf)./(maxf-minf);
    temp1  = f./sum(f,2);

    dis = pdist2(temp1,temp1);
    dis(logical(eye(length(dis)))) = inf;
    Choose = true(1,length(Population));

    while sum(Choose) > MaxSize
        Remain   = find(Choose);
        Temp     = sort(dis(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Choose(Remain(Rank(1))) = false;
    end
    Population = Population(Choose);
end
```

### `UpdateArchive.m`
```matlab
function Archive = UpdateArchive(Population,Archive,MaxSize)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Qiqi Liu

    Archive   = [Archive,Population];
    o         = Archive.objs;
    [c,ia,ic] = unique(o,'rows');
    Archive   = Archive(ia);
    ND        = NDSort(Archive.objs,1);
    Archive   = Archive(ND==1);
    N         = length(Archive);
    if N <= MaxSize
        return;
    end
    
    %% Calculate the fitness of each solution
    CAObj = Archive.objs;
    CAObj = (CAObj-repmat(min(CAObj),N,1))./(repmat(max(CAObj)-min(CAObj),N,1));
    I = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(CAObj(i,:)-CAObj(j,:));
        end
    end
    C = max(abs(I));
    F = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
    
    %% Delete part of the solutions by their fitnesses
    Choose = 1 : N;
    while length(Choose) > MaxSize
        [~,x] = min(F(Choose));
        F     = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
        Choose(x) = [];
    end
    Archive = Archive(Choose);

    %% Delete those which are too far from the archive
    o      = Archive.objs;
    o      = o-repmat(min(o),size(o,1),1);
    d      = sqrt(sum(o.^2,2));
    meanD  = sum(d,1)/size(o,1);
    delete = find(d>10*meanD);
    Archive(delete) = [];
end
```
