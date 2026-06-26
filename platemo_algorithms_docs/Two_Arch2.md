# Two_Arch2

**Tags**: <2015> <multi/many> <real/integer/label/binary/permutation>

## Description
Two-archive algorithm 2

## Reference
H. Wang, L. Jiao, and X. Yao. Two_Arch2: An improved two-archive algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 524-541.

## Source Code

### `MatingSelection.m`
```matlab
function [ParentC,ParentM] = MatingSelection(CA,DA,N)
% The mating selection of Two_Arch2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CAParent1 = randi(length(CA),1,ceil(N/2));
    CAParent2 = randi(length(CA),1,ceil(N/2));
    Dominate  = any(CA(CAParent1).objs<CA(CAParent2).objs,2) - any(CA(CAParent1).objs>CA(CAParent2).objs,2);  
    ParentC   = [CA([CAParent1(Dominate==1),CAParent2(Dominate~=1)]),...
                 DA(randi(length(DA),1,ceil(N/2)))];
    ParentM   = CA(randi(length(CA),1,N));
end
```

### `Two_Arch2.m`
```matlab
classdef Two_Arch2 < ALGORITHM
% <2015> <multi/many> <real/integer/label/binary/permutation>
% Two-archive algorithm 2
% CAsize --- --- Convergence archive size
% p      --- --- The parameter of fractional distance

%------------------------------- Reference --------------------------------
% H. Wang, L. Jiao, and X. Yao. Two_Arch2: An improved two-archive
% algorithm for many-objective optimization. IEEE Transactions on
% Evolutionary Computation, 2015, 19(4): 524-541.
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
            %% Parameter setting
            [CAsize,p] = Algorithm.ParameterSet(Problem.N,1/Problem.M);

            %% Generate random population
            Population = Problem.Initialization();
            CA = UpdateCA([],Population,CAsize);
            DA = UpdateDA([],Population,Problem.N,p);

            %% Optimization
            while Algorithm.NotTerminated(DA)
                [ParentC,ParentM] = MatingSelection(CA,DA,Problem.N);
                Offspring         = [OperatorGA(Problem,ParentC,{1,20,0,0}),OperatorGA(Problem,ParentM,{0,0,1,20})];
                CA = UpdateCA(CA,Offspring,CAsize);
                DA = UpdateDA(DA,Offspring,Problem.N,p);
            end
        end
    end
end
```

### `UpdateCA.m`
```matlab
function CA = UpdateCA(CA,New,MaxSize)
% Update CA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CA = [CA,New];
    N  = length(CA);
    if N <= MaxSize
        return;
    end
    
    %% Calculate the fitness of each solution
    CAObj = CA.objs;
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
        F = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
        Choose(x) = [];
    end
    CA = CA(Choose);
end
```

### `UpdateDA.m`
```matlab
function DA = UpdateDA(DA,New,MaxSize,p)
% Update DA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Find the non-dominated solutions
    DA = [DA,New];
    ND = NDSort(DA.objs,1);
    DA = DA(ND==1);
    N  = length(DA);
    if N <= MaxSize
        return;
    end
    
    %% Select the extreme solutions first
    Choose = false(1,N);
    [~,Extreme1] = min(DA.objs,[],1);
    [~,Extreme2] = max(DA.objs,[],1);
    Choose(Extreme1) = true;
    Choose(Extreme2) = true;
    
    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > MaxSize
        % Randomly delete several solutions
        Choosed = find(Choose);
        k = randperm(sum(Choose),sum(Choose)-MaxSize);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < MaxSize
        % Add several solutions by truncation strategy
        Distance = inf(N);
        for i = 1 : N-1
            for j = i+1 : N
                Distance(i,j) = norm(DA(i).obj-DA(j).obj,p);
                Distance(j,i) = Distance(i,j);
            end
        end
        while sum(Choose) < MaxSize
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
    end
    DA = DA(Choose);
end
```
