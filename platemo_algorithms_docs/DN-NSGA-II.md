# DN-NSGA-II

**Tags**: <2016> <multi> <real/integer> <multimodal>

## Description
Decision space based niching NSGA-II

## Reference
J. Liang, C. Yue, and B. Qu. Multimodal multi-objective optimization: A preliminary study. Proceedings of the IEEE Congress on Evolutionary Computation, 2016, 2454-2461.

## Source Code

### `DNNSGAII.m`
```matlab
classdef DNNSGAII < ALGORITHM
% <2016> <multi> <real/integer> <multimodal>
% Decision space based niching NSGA-II

%------------------------------- Reference --------------------------------
% J. Liang, C. Yue, and B. Qu. Multimodal multi-objective optimization: A
% preliminary study. Proceedings of the IEEE Congress on Evolutionary
% Computation, 2016, 2454-2461.
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
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection_Mod(round(Problem.N/2),round(Problem.N/2),Population.decs,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,SpCrowdDis_Obj] = EnvironmentalSelection(Population,N)
% The environmental selection of DN-NSGA-II

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
    SpCrowdDis_Obj = CrowdingDistance(Population.objs,FrontNo);
    SpCrowdDis_Dec = CrowdingDistance(Population.decs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(SpCrowdDis_Dec(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    SpCrowdDis_Obj = SpCrowdDis_Obj(Next);
end
```

### `TournamentSelection_Mod.m`
```matlab
function index = TournamentSelection_Mod(K,N,PopDec,varargin)
% Tournament selection of DN-NSGA-II

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    varargin = cellfun(@(S)reshape(S,length(varargin{1}),1),varargin,'UniformOutput',false);
    [~,rank] = sortrows([varargin{:}]);
    [~,rank] = sort(rank);    
    Parents  = randi(length(varargin{1}),K,N);
    for i = 1 : N
        [~,min_index] = min(pdist2(PopDec(Parents(1,i),:),PopDec(Parents(2:K,i),:)));
        Parents(2,i)  = Parents(min_index+1,i);
    end 
    Parents  = Parents(1:2,:);
    [~,best] = min(rank(Parents),[],1);
    index    = Parents(best+(0:N-1)*2);
end
```
