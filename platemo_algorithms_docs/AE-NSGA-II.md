# AE-NSGA-II

**Tags**: <2020> <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>

## Description
Autoencoding NSGA-II

## Reference
L. Feng, W. Zhou, W. Liu, Y. S. Ong, and K. C. Tan. Solving dynamic multiobjective problem via autoencoding evolutionary search. IEEE Transations on Cybernetics, 2020, 52(5): 2649-2662.

## Source Code

### `AENSGAII.m`
```matlab
classdef AENSGAII < ALGORITHM
% <2020> <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>
% Autoencoding NSGA-II

%------------------------------- Reference --------------------------------
% L. Feng, W. Zhou, W. Liu, Y. S. Ong, and K. C. Tan. Solving dynamic
% multiobjective problem via autoencoding evolutionary search. IEEE
% Transations on Cybernetics, 2020, 52(5): 2649-2662.
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
            ChangeCount = 0;
            NDS = cell(1,100);
            % Reset the number of saved populations (only for dynamic optimization)
            Algorithm.save = sign(Algorithm.save)*inf;
            
            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            % Archive for storing all populations before each change
            AllPop = [];
            NDS{1} = Population(FrontNo==1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Changed(Problem,Population)
                    ChangeCount        = ChangeCount+1;
                    NDS{ChangeCount+1} = Population(FrontNo==1);
                    AllPop             = [AllPop,Population];
                    % React to the change
                    curr_NDS = NDS{ChangeCount+1};
                    his_NDS  = NDS{ChangeCount};
                    [Population,FrontNo,CrowdDis] = AE_prediction(Problem,curr_NDS,his_NDS,Population,Problem.N);
                end
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                if Problem.FE >= Problem.maxFE
                    % Return all populations
                    Population = [AllPop,Population];
                    [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    Population = Population(rank);
                end
            end
        end
    end
end
```

### `AE_prediction.m`
```matlab
function [Population,FrontNo,CrowdDis] = AE_prediction(Problem,curr_NDS,his_NDS,curr_POS,NP)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    index1  = curr_NDS.decs;
    index2  = his_NDS.decs;
    [d, N1] = size(index1');
    [~, N2] = size(index2');
    if N1>N2
        [FrontNo,~] = NDSort(curr_NDS.objs,curr_NDS.cons,N1);
        CrowdDis    = CrowdingDistance(curr_NDS.objs,FrontNo);
        [~,index]   = sort(CrowdDis);
        curr_NDS    = curr_NDS(index(1:N2));
        NL = N2;
    else
        [FrontNo,~] = NDSort(his_NDS.objs,his_NDS.cons,N2);
        CrowdDis    = CrowdingDistance(his_NDS.objs,FrontNo);
        [~,index]   = sort(CrowdDis);
        b           = index(1:N1);
        his_NDS     = his_NDS(b);
        NL = N1;
    end
    index1 = curr_NDS.decs;
    index2 = his_NDS.decs;
    Q      = index2*index2';
    P      = index1*index2';
    lambda = 1e-5;
    reg    = lambda*eye(NL);
    reg(end,end) = 0;
    M    = P/(Q+reg);
    varM = M*index2;
    for i = 1 : NL
        for j = 1 : d
            var(i,j) = (index1(i,j)-varM(i,j)).^2;
        end
    end
    v = mean2(var);
    
    pre_solution = M*index1+v;
    POP          = Problem.Evaluation(pre_solution);
    curr_len     = length(POP);
    if curr_len > NP/2
        [FrontNo,~] = NDSort(POP.objs,POP.cons,N1);
        CrowdDis    = CrowdingDistance(POP.objs,FrontNo);
        [~,index]   = sort(CrowdDis);
        POP         = POP(index(1:NP/2));
    end
    Selected   = randperm(length(curr_POS),NP/2);
    POP2       = curr_POS(Selected);
    Population = [POP,POP2];
    if length(Population) < NP
        N    = NP-length(Population);
        POP3 = Problem.Initialization(N);
        Population = [Population,POP3];
    end
    [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
end
```

### `Changed.m`
```matlab
function changed = Changed(Problem,Population)
% Detect whether the problem changes

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    RePop1  = Population(randperm(end,ceil(end/10)));
    RePop2  = Problem.Evaluation(RePop1.decs);
    changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
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
