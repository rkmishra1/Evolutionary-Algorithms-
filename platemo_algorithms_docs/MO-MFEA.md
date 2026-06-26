# MO-MFEA

**Tags**: <2017> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>

## Description
Multi-objective multifactorial evolutionary algorithm

## Reference
A. Gupta, Y. Ong, L. Feng, and K. C. Tan. Multiobjective multifactorial optimization in evolutionary multitasking. IEEE Transactions on Cybernetics, 2017, 47(7): 1652-1665.

## Source Code

### `CreateOff.m`
```matlab
function SubOffspring = CreateOff(Problem,Population,rmp)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parent selection
    Parent11 = [];
    Parent21 = [];
    Parent12 = [];
    Parent22 = [];
    for i = 1 : floor(length(Population)/2)
        P1 = Population(i);
        P2 = Population(i+floor(length(Population)/2));
        if (P1.dec(end) == P2.dec(end)) || (rand<rmp)
            Parent11 = [Parent11,P1];
            Parent21 = [Parent21,P2];
        else
            Parent12 = [Parent12,P1];
            Parent22 = [Parent22,P2];
        end
    end

    %% Offspring generation
    if ~isempty(Parent11)
        Parent1Dec     = Parent11.decs;
        Parent2Dec     = Parent21.decs;
        OffDec1        = OperatorGA(Problem,[Parent1Dec;Parent2Dec]);
        OffDec1(:,end) = [Parent1Dec(:,end);Parent2Dec(:,end)];
    else
        OffDec1 = [];
    end
    if ~isempty(Parent12)
        Parent1Dec     = Parent12.decs;
        Parent2Dec     = Parent22.decs;
        OffDec2        = OperatorGA(Problem,[Parent1Dec;Parent2Dec],{0,20,1,20});
        OffDec2(:,end) = [Parent1Dec(:,end);Parent2Dec(:,end)];
    else
        OffDec2 = [];
    end
    SubOffspring = Divide(Problem.Evaluation([OffDec1;OffDec2]),length(Problem.SubD));
end
```

### `Divide.m`
```matlab
function SubPopulation = Divide(Population,SubCount)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopDec = Population.decs;
    skills = PopDec(:,end);
    for i = 1 : SubCount
        SubPopulation{i} = Population(skills==i);
    end
end
```

### `EnviSelect.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnviSelect(Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;

    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);

    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;

    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `MOMFEA.m`
```matlab
classdef MOMFEA < ALGORITHM
% <2017> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>
% Multi-objective multifactorial evolutionary algorithm
% rmp --- 1 --- Random mating probability

%------------------------------- Reference --------------------------------
% A. Gupta, Y. Ong, L. Feng, and K. C. Tan. Multiobjective multifactorial
% optimization in evolutionary multitasking. IEEE Transactions on
% Cybernetics, 2017, 47(7): 1652-1665.
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
            rmp = Algorithm.ParameterSet(1);
            
            %% Initialize population
            Population    = Problem.Initialization();
            SubPopulation = Divide(Population,length(Problem.SubD));
            
            %% Optimization
            while Algorithm.NotTerminated([SubPopulation{:}])
                [SubPopulation,Rank] = Sort(Problem,SubPopulation);
                Population           = [SubPopulation{:}];
                ParentPool           = Population(TournamentSelection(2,length(Population),[Rank{:}]));
                SubOffspring         = CreateOff(Problem,ParentPool,rmp);
                for i = 1 : length(Problem.SubD)
                    SubPopulation{i} = EnviSelect([SubPopulation{i},SubOffspring{i}],length(SubPopulation{i}));
                end
            end
        end
    end
end
```

### `Sort.m`
```matlab
function [SubPopulation,Rank] = Sort(Problem,SubPopulation)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Rank = {};
    for i = 1 : length(Problem.SubD)
        [~,FrontNo,CrowdDis] = EnviSelect(SubPopulation{i},length(SubPopulation{i}));
        [~,rank]             = sortrows([FrontNo',-CrowdDis']);
        SubPopulation{i}     = SubPopulation{i}(rank);
        Rank{i}              = 1 : length(rank);
    end
end
```
