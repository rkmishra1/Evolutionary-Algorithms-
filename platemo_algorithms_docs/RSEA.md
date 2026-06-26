# RSEA

**Tags**: <2017> <multi/many> <real/integer/label/binary/permutation>

## Description
Radial space division based evolutionary algorithm

## Reference
C. He, Y. Tian, Y. Jin, X. Zhang, and L. Pan. A radial space division based evolutionary algorithm for many-objective optimization. Applied Soft Computing, 2017, 61: 603-621.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,Range] = EnvironmentalSelection(Problem,Population,Range,N)
% The environmental selection of RSEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNO,MaxFNO] = NDSort(Population.objs,N);
    Next             = find(FrontNO<=MaxFNO);
    
    %% Environmental selection
    if any(Range(1,:)==Range(2,:))
        Choose = LastSelection(Problem,Population(Next).objs,ismember(Next,find(FrontNO<MaxFNO)),N,ceil(sqrt(N)));
    else
        Choose = LastSelection(Problem,(Population(Next).objs-repmat(Range(1,:),length(Next),1))./repmat(Range(2,:)-Range(1,:),length(Next),1),ismember(Next,find(FrontNO<MaxFNO)),N,ceil(sqrt(N))); 
    end
    Population = Population(Next(Choose));
	Range(1,:) = min([Range(1,:);Population.objs],[],1);
    Range(2,:) = max(Population(NDSort(Population.objs,1)==1).objs,[],1);
end

function Choose = LastSelection(Problem,PopObj,Choose,N,div)
% Select part of the solutions based on the radar grid

    %% Identify the extreme solutions
    [~,Extreme] = min(repmat(sqrt(sum(PopObj.^2,2)),1,size(PopObj,2)).*sqrt(1-(1-pdist2(PopObj,eye(size(PopObj,2)),'cosine')).^2),[],1); %Calculate the extreme points based on PBI
    Choose      = Choose | ismember(1:size(PopObj,1),Extreme);

    %% Calculate the convergence of each solution
	Con = sum(PopObj.^2,2).^0.5;
    Con = Con./max(Con);
    
    %% Calculate the radar grid of each solution
    [Site,RLoc] = RadarGrid(PopObj,div);
    RDis        = pdist2(RLoc,RLoc);
    RDis(logical(eye(length(RDis)))) = inf;
    CrowdG      = zeros(1,max(Site));
    temp        = tabulate(Site(Choose));
    CrowdG(temp(:,1)) = temp(:,2);

    %% Select N solutions one by one
    while sum(Choose) < N
        remainS  = find(~Choose);
        remainG  = unique(Site(remainS));
        bestG    = CrowdG(remainG) == min(CrowdG(remainG));
        current  = remainS(ismember(Site(remainS),remainG(bestG)));
        r        = 1-(Problem.FE/Problem.maxFE)^2;
        fitness  = Problem.M.*r.*Con(current) - min(RDis(current,Choose),[],2);
        [~,best] = min(fitness);
        Choose(current(best))       = true;
        CrowdG(Site(current(best))) = CrowdG(Site(current(best))) + 1;
    end
end
```

### `MatingSelection.m`
```matlab
function MatingPool = MatingSelection(PopObj,Range,N)
% The mating selection of RSEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

	%% Calculate the convergence of each solution
    PopObj = (PopObj-repmat(Range(1,:),size(PopObj,1),1))./repmat(Range(2,:)-Range(1,:),size(PopObj,1),1);
    Con = sum(PopObj.^2,2).^0.5;

    %% Calculate the radar grid of each solution
    [Site,~] = RadarGrid(PopObj,ceil(sqrt(size(PopObj,1)))); 
    temp     = tabulate(Site);
    CrowdG   = temp(:,2);
    
    %% Binary tournament selection
    MatingPool = zeros(1,ceil(N/2)*2);
    grids      = TournamentSelection(2,length(MatingPool),CrowdG);
    for i = 1 : length(MatingPool)
        current = find(Site==grids(i));
        if isempty(current)
             MatingPool(i) = randi(size(PopObj,1),1);
        else
            parents       = current(randi(length(current),1,2));
            [~,best]      = min(Con(parents));
            MatingPool(i) = parents(best);
        end
    end
end
```

### `RSEA.m`
```matlab
classdef RSEA < ALGORITHM
% <2017> <multi/many> <real/integer/label/binary/permutation>
% Radial space division based evolutionary algorithm

%------------------------------- Reference --------------------------------
% C. He, Y. Tian, Y. Jin, X. Zhang, and L. Pan. A radial space division
% based evolutionary algorithm for many-objective optimization. Applied
% Soft Computing, 2017, 61: 603-621.
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
            Range      = inf(2,Problem.M);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Range(1,:) = min([Range(1,:);Population.objs],[],1);
                Range(2,:) = max(Population(NDSort(Population.objs,1)==1).objs,[],1);
                MatingPool = MatingSelection(Population.objs,Range,ceil(Problem.N/2)*2);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Population = EnvironmentalSelection(Problem,[Population,Offspring],Range,Problem.N);
            end
        end
    end
end
```

### `RadarGrid.m`
```matlab
function [Site,RLoc] = RadarGrid(P,div)
% Calcualte the index of radar grid of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(P);
     
    %% Calculate the radar coordinate of each solution
    theta     = 0 : 2*pi/M : 2*pi/M*(M-1);
    RLoc(:,1) = sum(P.*repmat(cos(theta),N,1),2)./sum(P,2);
    RLoc(:,2) = sum(P.*repmat(sin(theta),N,1),2)./sum(P,2);
    RLoc      = (RLoc+1)/2;
    % Lower bound of the transferred points
    YL = min(RLoc,[],1);
    % Upper bound of the transferred points
    YU = max(RLoc,[],1);                                          
    if any(YU==YL)
        NRLoc = RLoc;
    else
        % Normalized points
        NRLoc = (RLoc-repmat(YL,N,1))./repmat(YU-YL,N,1);                   
    end
    
    %% Identify the index of grid of each solution
    GLoc            = floor(NRLoc.*div);
    GLoc(GLoc>=div) = div - 1;
    UniqueGLoc      = sortrows(unique(GLoc,'rows'));
    [~,Site]        = ismember(GLoc,UniqueGLoc,'rows');
end
```
