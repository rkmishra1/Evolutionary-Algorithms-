# NNIA

**Tags**: <2008> <multi> <real/integer/label/binary/permutation>

## Description
Nondominated neighbor immune algorithm

## Reference
M. Gong, L. Jiao, H. Du, and L. Bo. Multiobjective immune algorithm with nondominated neighbor-based selection. Evolutionary Computation, 2008, 16(2): 225-255.

## Source Code

### `Cloning.m`
```matlab
function C = Cloning(A,nC)
% Proportional cloning

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(A.objs);
    if all(CrowdDis==inf)
        CrowdDis = ones(size(CrowdDis));
    else
        CrowdDis(CrowdDis==inf) = 2*max(CrowdDis(CrowdDis~=inf));
    end

    %% Clone
    q = ceil(nC*CrowdDis/sum(CrowdDis));
    C = [];
    for i = 1 : length(A)
        C = [C,repmat(A(i),1,q(i))];
    end
end
```

### `NNIA.m`
```matlab
classdef NNIA < ALGORITHM
% <2008> <multi> <real/integer/label/binary/permutation>
% Nondominated neighbor immune algorithm
% nA ---  20 --- Size of active population
% nC --- 100 --- Size of clone population

%------------------------------- Reference --------------------------------
% M. Gong, L. Jiao, H. Du, and L. Bo. Multiobjective immune algorithm with
% nondominated neighbor-based selection. Evolutionary Computation, 2008,
% 16(2): 225-255.
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
            [nA,nC] = Algorithm.ParameterSet(20,100);

            %% Generate random population
            B = Problem.Initialization();               % Antibody population
            D = UpdateDominantPopulation(B,Problem.N);	% Dominant population

            %% Optimization
            while Algorithm.NotTerminated(D)
                A  = D(1:min(nA,length(D)));            % Active population
                C  = Cloning(A,nC);                     % Clone population
                C1 = OperatorGAhalf(Problem,[C,A(randi(length(A),1,length(C)))]);
                D  = UpdateDominantPopulation([D,C1],Problem.N);
            end
        end
    end
end
```

### `UpdateDominantPopulation.m`
```matlab
function D = UpdateDominantPopulation(D,N)
% Update the dominant population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    D = D(NDSort(D.objs,1)==1);
    [~,rank] = sort(CrowdingDistance(D.objs),'descend');
    D = D(rank(1:min(N,length(rank))));
end
```
