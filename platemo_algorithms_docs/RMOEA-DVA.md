# RMOEA-DVA

**Tags**: <2022> <multi> <real/integer> <robust>

## Description
Robust multi-objective evolutionary algorithm with decision variable assortment

## Reference
J. Liu, Y. Liu, Y. Jin, and F. Li. A decision variable assortment-based evolutionary algorithm for dominance robust multiobjective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(5): 3360-3375.

## Source Code

### `DVA.m`
```matlab
function  [HR,LR] = DVA(Problem,Population,nDVA,theta)
% Decision variable assortment

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,D] = size(Population.decs);
    sd    = zeros(nDVA,D);
    for i = 1 : D
        Decs = Population(randperm(N,nDVA)).decs;
        for j = 1 : nDVA
            % Perturb the i-th variable of the j-th solution
            delta     = Problem.delta*(Problem.upper(i)-Problem.lower(i));
            PDec      = repmat(Decs(j,:),Problem.H,1);
            PDec(:,i) = PDec(:,i) + 2*delta*rand(Problem.H,1) - delta;
            NewP      = Problem.Evaluation(PDec);
            FrontNo   = NDSort(NewP.objs,inf);
            % Calculate the variance
            sd(j,i) = std(FrontNo);
        end
    end
    sd = (sd-min(sd(:)))./(max(sd(:))-min(sd(:)));
    M  = mean(sd,1);
    HR = find(M>theta);
    LR = find(M<=theta);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,Population,N,type)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if type
        %% Optimization of HDRVs
        DRI = zeros(length(Population),1);
        for i = 1 : length(Population)
            PopX = Problem.Perturb(Population(i).dec);
            [FrontNo,MaxFNo] = NDSort(PopX.objs,inf);
            for j = 1 : MaxFNo
                DRI(i) = DRI(i) + numel(find(FrontNo==j))*(j-1);
            end
            DRI(i) = DRI(i)/length(PopX);
        end
        DRI    = (DRI-min(DRI))./(max(DRI)-min(DRI));
        PopObj = Population.objs + repmat(1./(1-DRI+1e-6),1,Problem.M);
        % Non-dominated sorting
        [FrontNo,MaxFNo] = NDSort(PopObj,N);
        Next = FrontNo < MaxFNo;
        % DRI based selection
        Last     = find(FrontNo==MaxFNo);
        [~,Rank] = sort(DRI(Last));
        Next(Last(Rank(1:N-sum(Next)))) = true;
        % Population for next generation
        Population = Population(Next);
        FrontNo    = FrontNo(Next);
        CrowdDis   = DRI(Next);
    else
        %% Optimization of LDRVs
        % Non-dominated sorting
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
        Next = FrontNo < MaxFNo;
        % Crowding distance based selection
        CrowdDis = CrowdingDistance(Population.objs,FrontNo);
        Last     = find(FrontNo==MaxFNo);
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:N-sum(Next)))) = true;
        % Population for next generation
        Population = Population(Next);
        FrontNo    = FrontNo(Next);
        CrowdDis   = CrowdDis(Next);
    end
end
```

### `RMOEADVA.m`
```matlab
classdef RMOEADVA< ALGORITHM
% <2022> <multi> <real/integer> <robust>
% Robust multi-objective evolutionary algorithm with decision variable assortment
% nDVA  ---  50 --- Number of solutions for decision variable assortment
% theta --- 0.3 --- Threshold for decision variable assortment

%------------------------------- Reference --------------------------------
% J. Liu, Y. Liu, Y. Jin, and F. Li. A decision variable assortment-based
% evolutionary algorithm for dominance robust multiobjective optimization.
% IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(5):
% 3360-3375.
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
            [nDVA,theta] = Algorithm.ParameterSet(50,0.3);
            
            %% Generate random population
            Population = Problem.Initialization();
            [HR,LR]    = DVA(Problem,Population,nDVA,theta);
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,Population,Problem.N,false);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                if ~isempty(LR)
                    OffDec       = Population(TournamentSelection(2,end,FrontNo,-CrowdDis)).decs;
                    NewDec       = OperatorGA(Problem,Population(randi(end,1,end)).decs);
                    OffDec(:,LR) = NewDec(:,LR);
                    Offspring    = Problem.Evaluation(OffDec);
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,[Population,Offspring],Problem.N,false);
                end
                if ~isempty(HR)
                    OffDec       = Population(TournamentSelection(2,end,FrontNo,-CrowdDis)).decs;
                    NewDec       = OperatorGA(Problem,Population(randi(end,1,end)).decs);
                    OffDec(:,HR) = NewDec(:,HR);
                    Offspring    = Problem.Evaluation(OffDec);
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,[Population,Offspring],Problem.N,true);
                end
            end
        end
    end
end
```
