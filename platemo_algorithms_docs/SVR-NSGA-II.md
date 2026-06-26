# SVR-NSGA-II

**Tags**: <2019> <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>

## Description
Support vector regression based NSGA-II

## Reference
L. Cao, L. Xu, E. D. Goodman, C. Bao, and S. Zhu. Evolutionary dynamic multiobjective optimization assisted by a support vector regression predictor. IEEE Transactions on Evolutionary Computation, 2019, 24(2): 305-319.

## Source Code

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

### `Reinitialization.m`
```matlab
function Population=Reinitialization(Problem,NDS,ChangeCount)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Dec = NDS{ChangeCount+1}.decs;
    P   = 4;
    Up  = Problem.upper;
    Lp  = Problem.lower;
    if ChangeCount < P
        for i = 1 : size(Dec,1)
            for z = 1 : size(Dec,2)
                for j = 1 : ChangeCount+1
                    index     = NDS{j}.decs;
                    data(i,j) = index(i,z);
                end
                Dec(i,z) = modeltrain(data,Up(z),Lp(z));
            end
        end
    else
        for i = 1 : size(Dec,1)
            for z = 1 : size(Dec,2)
                for j = 1 : P+1
                    index     = NDS{ChangeCount-P+j}.decs;
                    data(i,j) = index(i,z);
                end
                Dec(i,z) = modeltrain(data,Up(z),Lp(z));
            end
        end
    end
    Population = Problem.Evaluation(Dec);
end

function predata = modeltrain(data,Up,Lp)
    x_train = data(:, 1:end-1);
    y_train = data(:, end);
    svr     = fitrsvm(x_train, y_train, 'KernelFunction', 'rbf', 'Epsilon', 0.05, 'BoxConstraint', 1e3);
    a       = predict(svr,data(:,2:end));
    predata = a(end);
    predata = max(min(predata, Up), Lp);
end
```

### `SVRNSGAII.m`
```matlab
classdef SVRNSGAII < ALGORITHM
% <2019> <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>
% Support vector regression based NSGA-II

%------------------------------- Reference --------------------------------
% L. Cao, L. Xu, E. D. Goodman, C. Bao, and S. Zhu. Evolutionary dynamic
% multiobjective optimization assisted by a support vector regression
% predictor. IEEE Transactions on Evolutionary Computation, 2019, 24(2):
% 305-319.
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
            ChangeCount = Algorithm.ParameterSet(0);
            % Reset the number of saved populations (only for dynamic optimization)
            Algorithm.save = sign(Algorithm.save)*inf;
            
            %% Generate random population
            Population           = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            % Archive for storing all populations before each change
            AllPop = [];
            NDS    = cell(1,200);
            NDS{1} = Population;
            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Changed(Problem,Population)
                    % Save the population before the change
                    ChangeCount        = ChangeCount+1;
                    NDS{ChangeCount+1} = Population;
                    AllPop = [AllPop,Population];
                    % React to the change
                    Population           = Reinitialization(Problem,NDS,ChangeCount);
                    [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,length(Population));
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
