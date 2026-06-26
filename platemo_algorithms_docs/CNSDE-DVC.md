# CNSDE-DVC

**Tags**: <2019> <multi> <real/integer> <robust>

## Description
Constrained nondominated sorting differential evolution based on decision variable classification

## Reference
W. Du, W. Zhong, Y. Tang, W. Du, and Y. Jin. High-dimensional robust multi-objective optimization for order scheduling: A decision variable classification approach. IEEE Transactions on Industrial Informatics, 2019, 15(1): 293-304.

## Source Code

### `CNSDEDVC.m`
```matlab
classdef CNSDEDVC < ALGORITHM
% <2019> <multi> <real/integer> <robust>
% Constrained nondominated sorting differential evolution based on decision variable classification
% SN     ---     4 --- Number of perturbed solutions
% PN     ---     6 --- Number of perturbations
% TN     ---    15 --- Number of repeated times of perturbation
% theta  --- 0.001 --- Threshold for DVC operation
% eta    --- 0.001 --- Desired level of robustness

%------------------------------- Reference --------------------------------
% W. Du, W. Zhong, Y. Tang, W. Du, and Y. Jin. High-dimensional robust
% multi-objective optimization for order scheduling: A decision variable
% classification approach. IEEE Transactions on Industrial Informatics,
% 2019, 15(1): 293-304.
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
            [SN,PN,TN,theta,eta] = Algorithm.ParameterSet(4,6,15,0.001,0.001);

            %% Generate random population
            Population = Problem.Initialization();
            [HR,LR]    = DVC(Problem,Population,SN,PN,TN,theta);
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,Population,Problem.N,false,eta);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                if ~isempty(HR)
                    for subgen = 1 : 10
                        OffDec       = Population(TournamentSelection(2,end,FrontNo,-CrowdDis)).decs;
                        NewDec       = OperatorDE(Problem,Population.decs,Population(randi(end,1,end)).decs,Population(randi(end,1,end)).decs,{0.9,0.5,1,20});
                        OffDec(:,HR) = NewDec(:,HR);
                        Offspring    = Problem.Evaluation(OffDec);
                        [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,[Population,Offspring],Problem.N,true,eta);
                    end
                end
                if ~isempty(LR)
                    for subgen = 1 : 2
                        OffDec       = Population(TournamentSelection(2,end,FrontNo,-CrowdDis)).decs;
                        NewDec       = OperatorDE(Problem,Population.decs,Population(randi(end,1,end)).decs,Population(randi(end,1,end)).decs,{0.9,0.5,1,20});
                        OffDec(:,LR) = NewDec(:,LR);
                        Offspring    = Problem.Evaluation(OffDec);
                        [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,[Population,Offspring],Problem.N,false,eta);
                    end
                end
            end
        end
    end
end
```

### `DVC.m`
```matlab
function  [HR,LR] = DVC(Problem,Population,SN,PN,TN,theta)
% Decision variable classification

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

	[N,D]  = size(Population.decs);
    AllVal = zeros(TN,D);
    for T = 1 : TN
        Var = zeros(SN,D);
        for i = 1 : D
            a    = randperm(N,SN);
            Decs = Population(a).decs;
            Objs = Population(a).objs;
            for j = 1 : SN
                % Perturb the i-th variable of the j-th solution for PN times
                delta     = Problem.delta*(Problem.upper(i)-Problem.lower(i));
                PDec      = repmat(Decs(j,:),PN,1);
                PDec(:,i) = PDec(:,i) + 2*delta*rand(PN,1) - delta;
                NewP      = Problem.Evaluation(PDec);
                % Calculate the variance
                vcv      = sum(abs(NewP.objs-repmat(Objs(j,:),PN,1)),2);
                Var(j,i) = std(vcv,0,1)^2;
            end
        end
        AllVal(T,:) = mean(Var,1);
    end
    TVal = zeros(1,D);
    for i = 1 : D
        TVal(i) = numel(find(AllVal(:,i)<theta));
    end
    HR = find(TVal<=mean(TVal));	% Highly robustness-related variables
    LR = find(TVal>mean(TVal));     % Weakly robustness-related variables
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Problem,Population,N,type,eta)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    PopObj = Population.objs;
    if type
        % Selection of DV1
        for i = 1 : length(Population)
            PopX         = Problem.Perturb(Population(i).dec);
            PopObjV(i,:) = mean(PopX.objs,1);
        end
        DR = sum(abs(PopObjV-PopObj),2);
        DR(DR<=eta) = 0;
        [FrontNo,MaxFNo] = NDSort(PopObj,DR,N);
    else
        % Selection of DV2
        [FrontNo,MaxFNo] = NDSort(PopObj,N);
    end
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
end
```
