# TS-NSGA-II

**Tags**: <2022> <multi/many> <real/integer/label/binary/permutation>

## Description
Two stage NSGA-II

## Reference
F. Ming, W. Gong, and L. Wang. A two-stage evolutionary algorithm with balanced convergence and diversity for many-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(10): 6222-6234.

## Source Code

### `DensityEstimate.m`
```matlab
function Density = DensityEstimate(Population,W)
% Estimate the density of each individual

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    Zmax       = max(Population.objs,[],1);
    Zmin       = min(Population.objs,[],1);
    SPopObj    = (Population.objs-repmat(Zmin,size(Population.objs,1),1))./(repmat(Zmax,size(Population.objs,1),1)-repmat(Zmin,size(Population.objs,1),1));
    [~,Region] = max(1-pdist2(SPopObj,W,'cosine'),[],2);
    [value,~]  = sort(Region,'ascend');
    flag       = max(value);
    counter    = histc(value,1:flag);
    Density    = counter(Region);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,d2] = EnvironmentalSelection(OffSpring,W,N)
% The selection of TSNSGAII

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    %% Normalization
    PopObj = OffSpring.objs;
    Fmin   = min(PopObj,[],1);
    Fmax   = max(PopObj,[],1);
    PopObj = (PopObj-repmat(Fmin,size(PopObj,1),1))./repmat(Fmax-Fmin,size(PopObj,1),1);
    
    %% Association
    normP   = sqrt(sum(PopObj.^2,2));
    Cosine  = 1 - pdist2(PopObj,W,'cosine');
    d1      = repmat(normP,1,size(W,1)).*Cosine;
    d2      = repmat(normP,1,size(W,1)).*sqrt(1-Cosine.^2);
    [d2,RP] = min(d2,[],2);
    d1      = d1((1:length(RP))'+(RP-1)*length(RP));
    
    %% Favor extreme solutions
    ND              = find(NDSort(PopObj,1)==1);
    [~,Extreme]     = max(PopObj(ND,:),[],1);
    d1(ND(Extreme)) = 0;
    d2(ND(Extreme)) = 0;
    
    %% SPDsorting
    [FrontNo,MaxFNo] = SPDSort(PopObj,d1,d2,RP,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(d2(Last));
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = OffSpring(Next);
    FrontNo    = FrontNo(Next);
    d2         = d2(Next);
end
```

### `EnvironmentalSelection1.m`
```matlab
function Population = EnvironmentalSelection1(OffSpring,W,N,M)
% The selection of TSNSGAII

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    n = 2*M;

    ndsPopulation = [];
    Objs = OffSpring.objs;
    [FrontNO,MaxNO] = NDSort(Objs,inf);
    for i = 1 : MaxNO
        ndsPopulation = cat(2,ndsPopulation,OffSpring(FrontNO==i));
        if length(ndsPopulation) >= N
            break;
        end
    end

    Level = LevelSort(ndsPopulation,n);
    levelPopulation = [];
    for i = 1 : n
        levelPopulation = cat(2,levelPopulation,ndsPopulation(Level==i));
        if size(levelPopulation,1) >= N
            break;
        end
    end

    ndsPopulation = levelPopulation;

    Population = [];
    for i = 1 : size(W,1)
        Zmax          = max(ndsPopulation.objs,[],1);
        Zmin          = min(ndsPopulation.objs,[],1);
        SPopObj       = (ndsPopulation.objs-repmat(Zmin,size(ndsPopulation.objs,1),1))./(repmat(Zmax,size(ndsPopulation.objs,1),1)-repmat(Zmin,size(ndsPopulation.objs,1),1));
        [~,index]     = max(1-pdist2(SPopObj,W(i,:),'cosine'));
        Population    = [Population,ndsPopulation(index)];
        ndsPopulation = setdiff(ndsPopulation,ndsPopulation(index));
    end
end
```

### `LevelSort.m`
```matlab
function Level = LevelSort(Population,n)
% Ssort the level of Population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    Zmax     = max(Population.objs,[],1);
    Zmin     = min(Population.objs,[],1);
    interval = (Zmax-Zmin)./n;
    Level    = zeros(length(Population),1);
    objs     = Population.objs;
    for i = 1 : length(Population)
        t = 0;
        leveled = 0;
        obj = objs(i,:);
        while leveled == 0
            t = t + 1;
            leveled = 1;
            for j = 1 : size(objs,2)
                if obj(1,j) > Zmin(j)+(t+1)*interval(1,j)
                    leveled = 0;
                    break;
                end
            end
        end
        Level(i) = t;
    end 
end
```

### `SPDSort.m`
```matlab
function [FrontNo,MaxFNo] = SPDSort(PopObj,d1,d2,RP,nSort)
% SPD sorting

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    [N,M]   = size(PopObj);
    FrontNo = inf(1,N);
    MaxFNo  = 0;
    while sum(FrontNo<inf) < min(nSort,N)
        MaxFNo = MaxFNo + 1;
        Sorted = FrontNo ~= inf;
        D      = Sorted;
        for i = 1 : N
            if ~D(i)
                for j = i+1 : N
                    if ~D(j)
                        domi = 0;
                        for m = 1 : M
                            if PopObj(i,m) < PopObj(j,m)
                                if domi == -1
                                    domi = 0;
                                    break;
                                else
                                    domi = 1;
                                end
                            elseif PopObj(i,m) > PopObj(j,m)
                                if domi == 1
                                    domi = 0;
                                    break;
                                else
                                    domi = -1;
                                end
                            end
                        end
                        if domi==0 && RP(i)==RP(j)
                            if d1(i)+5*d2(i) < d1(j)+5*d2(j)
                            	domi = 1;
                            elseif d1(i)+5*d2(i) > d1(j)+5*d2(j)
                            	domi = -1;
                            end
                        end
                        if domi == 1
                            D(j) = true;
                        elseif domi == -1
                            D(i) = true;
                            break;
                        end
                    end
                end
                if ~D(i)
                    FrontNo(i) = MaxFNo;
                end
            end
        end
    end
end
```

### `TSNSGAII.m`
```matlab
classdef TSNSGAII < ALGORITHM
% <2022> <multi/many> <real/integer/label/binary/permutation>
% Two stage NSGA-II

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, and L. Wang. A two-stage evolutionary algorithm with
% balanced convergence and diversity for many-objective optimization. IEEE
% Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(10):
% 6222-6234.
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
            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);

            %% Generate random population
            Population     = Problem.Initialization();
            [~,FrontNo,d2] = EnvironmentalSelection(Population,W,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,d2);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                if Problem.FE < (0.8+0.2*(1-3/Problem.M))*Problem.maxFE
                    [Population,FrontNo,d2] = EnvironmentalSelection([Population,Offspring],W,Problem.N);
                else
                    Population = EnvironmentalSelection1([Population,Offspring],W,Problem.N,Problem.M);
                    FrontNo    = NDSort(Population.objs,inf);
                    d2 = DensityEstimate(Population,W);
                end
            end
        end
    end
end
```
