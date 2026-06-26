# HHC-MMEA

**Tags**: <2023> <multi> <real> <large/none> <multimodal> <sparse>

## Description
Hybrid hierarchical clustering based multi-modal multi-objective evolutionary algorithm

## Reference
Z. Ding, L. Cao, L. Chen, D. Sun, X. Zhang, and Z. Tao. Large-scale multimodal multiobjective evolutionary optimization based on hybrid hierarchical clustering. Knowledge-Based Systems, 2023, 266: 110398.

## Source Code

### `Divide.m`
```matlab
function [Population1,Mask1,Dec1,FrontNo1,SV1,Population2,Mask2,Dec2,FrontNo2,SV2] = Divide(Population1,Mask1,Dec1,FrontNo1,row,col)
% The divide operation of HHC-MMEA
%The Divisionflag() function returns the sequence number to perform this split

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % subpopulation2
    Population2 = Population1(col);
    Dec2        = Dec1(col,:);
    Mask2       = Mask1(col,:);
    FrontNo2    = FrontNo1(col);
    mask        = Mask2(FrontNo2==1,:);
    SV2         = sum(mask,1)/size(mask,1);

    %subpopulation1
    Population1 = Population1(row);
    Dec1        = Dec1(row,:);
    Mask1       = Mask1(row,:);
    FrontNo1    = FrontNo1(row);
    mask        = Mask1(FrontNo1==1,:);
    SV1         = sum(mask,1)/size(mask,1);
end
```

### `DivisionFlag.m`
```matlab
function [subp1,subp2,DivisionFlag] = DivisionFlag(Mask,FrontNo)
% Calculate the farthest two solutions and their similarity

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    mask = Mask(FrontNo==1,:);
    dis  = pdist2(mask,mask,'hamming');
    M    = max(max(dis));
    [leader1,leader2] = find(dis==M,1);
    [~,subleader1]    = ismember(mask(leader1,:),Mask,'rows');
    [~,subleader2]    = ismember(mask(leader2,:),Mask,'rows');
    subp1 = subleader1;
    subp2 = subleader2;
    s     = sum(mask(leader1,:)&mask(leader2,:))/min(sum(mask(leader1,:)),sum(mask(leader2,:)));%相似度
    if M>0.05 && s<=0.5
        DivisionFlag = 1;
        %% hierarchical clustering(divide operation)
        for i = 1:size(Mask,1)
            if i~=subleader1 && i~=subleader2
                dis1 = pdist2(mask(leader1,:),Mask(i,:),'hamming');
                dis2 = pdist2(mask(leader2,:),Mask(i,:),'hamming');
                if dis1 <= dis2
                    subp1 = [subp1,i];
                else
                    subp2 = [subp2,i];
                end
            end
        end
    else
        DivisionFlag = 0;
    end
    if size(Mask,1) < 50
        DivisionFlag = 0;
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,N,dis)
% The environmental selection of HHC-MMEA

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
    N = min(N,length(Population));

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

### `HHCMMEA.m`
```matlab
classdef HHCMMEA < ALGORITHM
% <2023> <multi> <real> <large/none> <multimodal> <sparse>
% Hybrid hierarchical clustering based multi-modal multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Z. Ding, L. Cao, L. Chen, D. Sun, X. Zhang, and Z. Tao. Large-scale
% multimodal multiobjective evolutionary optimization based on hybrid
% hierarchical clustering. Knowledge-Based Systems, 2023, 266: 110398.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lve Cao
% Fitness vector is the global guide vector and GV is the local guide
% vector in the original paper

    methods
        function main(Algorithm,Problem)
            %% Calculate the fitness of each decision variable
            Fitness = zeros(1,Problem.D);
            REAL    = ~strcmp(Problem.encoding,'binary');
            for i = 1 : 1+4*REAL
                if REAL
                    Dec = unifrnd(repmat(Problem.lower,Problem.D,1),repmat(Problem.upper,Problem.D,1));
                else
                    Dec = ones(Problem.D,Problem.D);
                end
                Mask       = eye(Problem.D);
                Population = Problem.Evaluation(Dec.*Mask);
                Fitness    = Fitness + NDSort([Population.objs,Population.cons],inf);
            end
            %% Population initialization
            P    = UniformPoint(Problem.N,Problem.D,'Latin');
            Dec  = P.*repmat(Problem.upper-Problem.lower,Problem.N,1) + repmat(Problem.lower,Problem.N,1);
            Mask = double(UniformPoint(Problem.N,Problem.D,'Latin') > 0.5);
            Population  = Problem.Evaluation(Dec.*Mask);
            K           = 1;    % Number of subpopulations
            Masks       = cell(1,K);
            Decs        = cell(1,K);
            Populations = cell(1,K);
            GV          = cell(1,K);
            FrontNo     = cell(1,K);
            CrowdDis    = cell(1,K);
            leader      = cell(1,K);
            index       = randperm(floor(Problem.N/K)*K);
            temp        = reshape(index,K,floor(Problem.N/K));
            for i = 1 : K
                Populations{i} = Population(temp(i,:));
                Masks{i}       = Mask(temp(i,:),:);
                Decs{i}        = Dec(temp(i,:),:);
                [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},length(Populations{i}));
                GV{i}          = UpdateGV(zeros(1,Problem.D),Masks{i},FrontNo{i});
            end
            %% Optimization
            while Algorithm.NotTerminated(Population)
                [~,rank] = sort(SubPopRank(Populations));
                for i = 1 : K
                    GV{rank(i)} = UpdateGV(GV{rank(i)},Masks{rank(i)},FrontNo{rank(i)});
                    Mating      = TournamentSelection(2,2*length(Populations{rank(i)}),FrontNo{rank(i)},-CrowdDis{rank(i)});
                    % Adaptive Variation
                    fgv   = GV{rank(i)};
                    fgv(fgv<1e-2) = 0;
                    idx   = kmeans(fgv',2);
                    idx   = idx';
                    s1    = sum(GV{rank(i)}(idx==1))/sum(idx==1);
                    s2    = sum(GV{rank(i)}(idx==2))/sum(idx==2);
                    s     = max(s1,s2);
                    delta = Problem.FE/Problem.maxFE;
                    if s < 0.5 || K==1
                        [OffDec,OffMask] = Operator1(Problem,Decs{rank(i)}(Mating,:),Masks{rank(i)}(Mating,:),Fitness);
                    else
                        leader{rank(i)}  = Leader(Masks{rank(i)},FrontNo{rank(i)},GV{rank(i)},Problem.D);
                        [OffDec,OffMask] = Operator2(Problem,Decs{rank(i)}(Mating,:),Masks{rank(i)}(Mating,:),GV{rank(i)},delta);
                    end
                    Offspring = Problem.Evaluation(OffDec.*OffMask);
                    Populations{rank(i)} = [Populations{rank(i)},Offspring];
                    Decs{rank(i)}        = [Decs{rank(i)};OffDec];
                    Masks{rank(i)}       = [Masks{rank(i)};OffMask];

                    %  Improve Environmental Selection
                    if K == 1
                        [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},CrowdDis{rank(i)}] = EnvironmentalSelection(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},floor(length(Populations{rank(i)})/2));
                    else
                        r = zeros(1,Problem.D);
                        for j = 1 : K
                            if j~=i
                                fgv = GV{rank(j)};
                                fgv(fgv < 1e-2) = 0;
                                idx = kmeans(fgv',2);
                                idx = idx';
                                s1  = sum(fgv(idx==1))/sum(idx==1);
                                s2  = sum(fgv(idx==2))/sum(idx==2);
                                s   = max(s1,s2);
                                if s > 0.5
                                    if s1 > s2
                                        r(idx==1) = r(idx==1)+ fgv(idx==1);
                                    else
                                        r(idx==2) = r(idx==2)+ fgv(idx==2);
                                    end
                                end
                            end
                        end
                        r(r>0) = 1;
                        dis    = sum(repmat(r,length(Populations{rank(i)}),1)&Masks{rank(i)},2);
                        [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},CrowdDis{rank(i)}] = EnvironmentalSelection(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},floor(length(Populations{rank(i)})/2),dis);
                    end
                end

                %% hybrid hierarchical clustering
                if mod(ceil(Problem.FE/Problem.N),10) == 0
                    k = K;
                    %divide operation(Line103-118)
                    while 1
                        for i = 1 : K
                            [row,col,divisionFlag] = DivisionFlag(Masks{i},FrontNo{i});
                            if divisionFlag == 1
                                k = k + 1;
                                [Populations{i},Masks{i},Decs{i},FrontNo{i},GV{i},Populations{k},Masks{k},Decs{k},FrontNo{k},GV{k}] = Divide(Populations{i},Masks{i},Decs{i},FrontNo{i},row,col);
                                leader{i} = Masks{i}(1,:);
                                leader{k} = Masks{k}(1,:);
                            end
                        end
                        K = k;
                        if divisionFlag == 0
                            break;
                        end
                    end
                    % merge operation（Line120-143）
                    fgv = GV;
                    for i = 1 : K
                        fgv{i}(fgv{i} < 0.5) = 0 ;
                        fgv{i}(fgv{i} > 0)   = 1;
                        mask   = Masks{i}(FrontNo{i}==1,:);
                        dis    = pdist2(mask,fgv{i},'hamming');
                        index  = find(dis==min(dis),1);
                        fgv{i} = mask(index,:);
                    end
                    [ss,index] = SubPopSimility(Populations,leader);
                    while K>2 && ss>0.5
                        i = index(1);
                        j = index(2);
                        [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection([Populations{i},Populations{j}],[Decs{i};Decs{j}],[Masks{i};Masks{j}],length(Populations{i}));
                        Populations(j) = [];
                        Decs(j)        = [];
                        Masks(j)       = [];
                        FrontNo(j)     = [];
                        GV(j)          = [];
                        fgv(j)         = [];
                        K              = K-1;
                        [ss,index]     = SubPopSimility(Populations(1:K),Masks);
                    end
                    % fill operation
                    popsize  = floor(Problem.N/K);  % average size of each Subpopulation
                    [~,rank] = sort(SubPopRank(Populations));
                    for i = 1 : K
                        len = popsize - length(Populations{rank(i)});
                        if len > 0
                            P    = UniformPoint(len,Problem.D,'Latin');
                            dec  = P.*unifrnd(repmat(Problem.lower,len,1),repmat(Problem.upper,len,1));
                            mask = zeros(len,Problem.D);
                            F    = zeros(1,Problem.D);
                            for j = 1 : K
                                if j ~= i
                                    F = F + GV{rank(j)};
                                end
                            end
                            for j = 1 : len
                                mask(j,TournamentSelection(2,floor(rand*Problem.D),F)) = 1;
                            end
                            Populations{rank(i)} = [Populations{rank(i)},Problem.Evaluation(dec.*mask)];
                            Masks{rank(i)}       = [Masks{rank(i)};mask];
                            Decs{rank(i)}        = [Decs{rank(i)};dec];
                        end
                        [Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},FrontNo{rank(i)},CrowdDis{rank(i)}] = EnvironmentalSelection(Populations{rank(i)},Decs{rank(i)},Masks{rank(i)},popsize);
                    end
                end
                Population = [Populations{:}];
            end
        end
    end
end
```

### `Leader.m`
```matlab
function leader = Leader(Mask,Front,GV,D)
% update the leader by GV

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    fs            = zeros(1,D);
    sample        = rand(1,D);
    fs(GV>sample) = 1;
    mask          = Mask(Front==1,:);
    dis           = pdist2(mask,fs,'hamming');
    index         = find(dis==min(dis),1);
    leader        = mask(index,:);
end
```

### `Operator1.m`
```matlab
function [OffDec,OffMask] = Operator1(Problem, ParentDec,ParentMask,Fitness)
% The operator of SparseEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        if rand < 0.5
            index = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
            index = index(TS(-Fitness(index)));
            OffMask(i,index) = 0;
        else
            index = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
            index = index(TS(Fitness(index)));
            OffMask(i,index) = Parent2Mask(i,index);
        end
    end
    
    
    %% Mutation for mask
    for i = 1 : N/2
        if rand < 0.5
            index = find(OffMask(i,:));
            index = index(TS(-Fitness(index)));
            OffMask(i,index) = 0;
        else
            index = find(~OffMask(i,:));
            index = index(TS(Fitness(index)));
            OffMask(i,index) = 1;
        end
    end
    
    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
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

### `Operator2.m`
```matlab
function [OffDec,OffMask] = Operator2(Problem,ParentDec,ParentMask,Score, delta)
% Operator of HHC-MMEA

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------  
    
    %% Parameter setting
    N           = size(ParentDec,1);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2    
        index1 = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
        index2 = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
        p1     = 1./(1+exp(-Score(index1)));
        p2     = 1./(1+exp(-Score(index2)));
        idx1   = index1(p1<rand(size(p1)));
        idx2   = index2(p2>rand(size(p2)));          
        OffMask(i,idx1) = 0;
        OffMask(i,idx2) = 1;
    end
    
    %% Mutation for mask
    if rand<(1-delta)
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
    else
        [N,D]       = size(ParentDec);
        Mutation_p  = 1/D;                      % Probability of mutation
        Mu_exchange = rand(N/2,D)<Mutation_p;   % The decision variables less than 1/D will be mutated
        rate0       = Score;                    % The probability that  0 inverts to  1
        rate1       = 1-rate0;                  % The probability that  1 inverts to  0
        for i = 1 : N/2
            if sum(Mu_exchange(i,:))
                subscript = find(Mu_exchange(i,:)==1);
                rate      = zeros(1,size(subscript,2));
                rate(logical(OffMask(i,subscript)))  = rate1(subscript(logical(OffMask(i,subscript))));
                rate(logical(~OffMask(i,subscript))) = rate0(subscript(logical(~OffMask(i,subscript))));
                exchange  = rand(1,size(subscript,2)) < rate;
                OffMask(i,subscript(exchange)) = ~OffMask(i,subscript(exchange));
            end
        end 
    end

    %% Crossover and mutation for dec
    OffDec = OperatorGAhalf(Problem,ParentDec);
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
function [ss,index] = SubPopSimility(Populations,leader)
% Calculate the similarity between subpopulations

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    K  = length(Populations);
    fs = cell(1,K);
    for m  = 1 : K
        fs{m} = leader{m};
    end
    ss    = 0;
    index = [];
    for i = 1 : K-1
        for j = i+1 : K
            if simility(fs{i}(1,:),fs{j}(1,:)) > ss 
                ss = simility(fs{i}(1,:),fs{j}(1,:)); 
                index = [i,j];
            end
        end
    end
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
