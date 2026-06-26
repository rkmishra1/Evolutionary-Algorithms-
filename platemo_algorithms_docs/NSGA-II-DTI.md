# NSGA-II-DTI

**Tags**: <2006> <multi> <real/integer/label/binary/permutation> <constrained/none> <robust>

## Description
NSGA-II of Deb's type I robust version

## Reference
K. Deb and H. Gupta. Introducing robustness in multi-objective optimization. Evolutionary Computation, 2006, 14(4): 463-494.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis,PopObjV,PopConV] = EnvironmentalSelection(Population,N,PopObjV,PopConV)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(PopObjV,PopConV,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObjV,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    PopObjV    = PopObjV(Next,:);
    PopConV    = PopConV(Next,:);
end
```

### `MeanEffective.m`
```matlab
function [PopObjV,PopConV] = MeanEffective(Problem,Population)
% Calculate the mean objective values of each solution in the vicinity

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for i = 1 : length(Population)
        PopX         = Problem.Perturb(Population(i).dec);
        PopObjV(i,:) = mean(PopX.objs,1);
        PopConV(i,:) = mean(PopX.cons,1);
    end
end
```

### `NSGAIIDTI.m`
```matlab
classdef NSGAIIDTI < ALGORITHM
% <2006> <multi> <real/integer/label/binary/permutation> <constrained/none> <robust>
% NSGA-II of Deb's type I robust version

%------------------------------- Reference --------------------------------
% K. Deb and H. Gupta. Introducing robustness in multi-objective
% optimization. Evolutionary Computation, 2006, 14(4): 463-494.
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
            %% Generate random population
            Population = Problem.Initialization();
            [PopObjV,PopConV]    = MeanEffective(Problem,Population);
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N,PopObjV,PopConV);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool),{1,10,1,50});
                [OffObjV,OffConV] = MeanEffective(Problem,Offspring);
                [Population,FrontNo,CrowdDis,PopObjV,PopConV] = EnvironmentalSelection([Population,Offspring],Problem.N,[PopObjV;OffObjV],[PopConV;OffConV]);
            end
        end
    end
end
```
