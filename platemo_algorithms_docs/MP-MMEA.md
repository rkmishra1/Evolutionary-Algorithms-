# MP-MMEA

**Tags**: <2021> <multi> <real/integer> <large/none> <multimodal> <sparse>

## Description
Multi-population multi-modal multi-objective evolutionary algorithm

## Reference
Y. Tian, R. Liu, X. Zhang, H. Ma, K. C. Tan, and Y. Jin. A multipopulation evolutionary algorithm for solving large-scale multimodal multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2021, 25(3): 405-418.

## Source Code

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
    [~,uni] = unique(PopObj,'rows');
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

### `MPMMEA.m`
```matlab
classdef MPMMEA < ALGORITHM
% <2021> <multi> <real/integer> <large/none> <multimodal> <sparse>
% Multi-population multi-modal multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, R. Liu, X. Zhang, H. Ma, K. C. Tan, and Y. Jin. A
% multipopulation evolutionary algorithm for solving large-scale multimodal
% multiobjective optimization problems. IEEE Transactions on Evolutionary
% Computation, 2021, 25(3): 405-418.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

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
            K           = 2;  % Number of subpopulations
            Masks       = cell(1,K);
            Decs        = cell(1,K);
            Populations = cell(1,K);
            GV          = cell(1,K);
            FrontNo     = cell(1,K);
            CrowdDis    = cell(1,K);
            index       = randperm(floor(Problem.N/K)*K);
            temp        = reshape(index,K,floor(Problem.N/K));
            for i = 1 : K
                Populations{i} = Population(temp(i,:));
                Masks{i}       = Mask(temp(i,:),:);
                Decs{i}        = Dec(temp(i,:),:);
                [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},length(Populations{i}));
                GV{i}          = UpdateGV(zeros(1,Problem.D),Masks{i},FrontNo{i});
            end
            endingFlag = 0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Subpopulation evolution
                [~,rank] = sort(SubPopRank(Populations));
                for i = 1 : K                
                    GV{rank(i)}          = UpdateGV(GV{rank(i)},Masks{rank(i)},FrontNo{rank(i)});
                    Mating               = TournamentSelection(2,2*length(Populations{rank(i)}),FrontNo{rank(i)},-CrowdDis{rank(i)});
                    [OffDec,OffMask]     = Operator(Problem,Decs{rank(i)}(Mating,:),Masks{rank(i)}(Mating,:),GV{rank(i)});
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
                % Merge-and-divide operation
                if mod(ceil(Problem.FE/Problem.N),50)==0
                	[~,best]     = SubPopRank(Populations);
                    divisionFlag = all(best==1);
                    [ss,index]   = SubPopSimility(Populations,Masks);
                    if ss > 0.5
                        K = K-1;
                        i = index(1);
                        j = index(2);
                        [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection([Populations{i},Populations{j}],[Decs{i};Decs{j}],[Masks{i};Masks{j}],floor(Problem.N/K));
                        Populations(j) = [];
                        Decs(j)        = [];
                        Masks(j)       = [];
                        GV(j)          = [];
                        endingFlag     = endingFlag + 1;
                    elseif divisionFlag == 1
                        for i = 1 : K
                            [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},floor(Problem.N/(K+1)));
                        end
                        K    = K + 1;
                        Dec  = unifrnd(repmat(Problem.lower,floor(Problem.N/K),1),repmat(Problem.upper,floor(Problem.N/K),1));
                        Mask = zeros(floor(Problem.N/K),Problem.D);
                        F    = zeros(1,Problem.D);
                        for i = 1: K-1
                            F = F + GV{i};
                        end
                        for i = 1 : floor(Problem.N/K)
                            Mask(i,TournamentSelection(2,floor(rand*Problem.D),F)) = 1;
                        end
                        Populations{K} = Problem.Evaluation(Dec.*Mask);
                        Masks{K}       = Mask;
                        Decs{K}        = Dec;
                        GV{K}          = zeros(1,Problem.D);
                        [Populations{K},Decs{K},Masks{K},FrontNo{K},CrowdDis{K}] = EnvironmentalSelection(Populations{K},Decs{K},Masks{K},length(Populations{K}));
                        GV{K}          = UpdateGV(zeros(1,Problem.D),Masks{K},FrontNo{K});
                    end
                end
                Population = [Populations{:}];
            end
        end
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Score)
% Operator of MP-MMEA

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------  
    
    %% Parameter setting
    N = size(ParentDec,1);
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
function [ss,index] = SubPopSimility(Populations,Masks)
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
    fs = zeros(1,K);
    for m  = 1 : K
        [~,fs(m)] = min(mean(Populations{m}.objs,2));
    end
    ss    = 0;
    index = [];
    for i = 1 : K-1
        for j = i+1 : K
            if simility(Masks{i}(fs(i),:),Masks{j}(fs(j),:)) > ss
                ss    = simility(Masks{i}(fs(i),:),Masks{j}(fs(j),:));
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
