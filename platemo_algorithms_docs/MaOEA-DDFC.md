# MaOEA-DDFC

**Tags**: <2015> <multi/many> <real/integer/label/binary/permutation>

## Description
Many-objective evolutionary algorithm based on directional diversity and

## Reference
J. Cheng, G. G. Yen, and G. Zhang. A many-objective evolutionary algorithm with enhanced mating and environmental selections. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 592-605.

## Source Code

### `CalFC.m`
```matlab
function FC = CalFC(PopObj,Zmin)
% Calculate the FC value of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M]  = size(PopObj);
    PopObj = PopObj - repmat(Zmin,N,1);
    
    %% Favorable weight
    w     = zeros(N,M);
    bound = any(PopObj==repmat(Zmin,N,1),2);
    w(repmat(bound,1,M) & PopObj==0) = 1;
    w(repmat(bound,1,M) & PopObj~=0) = 0;
    w(~bound,:) = 1./PopObj(~bound,:)./repmat(sum(1./PopObj(~bound,:),2),1,M);
    
    %% Calculate the FC value
    FC = max(w.*PopObj,[],2);
    FC = max(FC,1e-6);
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,Zmin,N,K,L)
% The environmental selection of MaOEA-DDFC

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,Zmin,N,K,L);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Choose = LastSelection(P,F,Zmin,Total,K,L)
% Select part of the solutions in the last front

    PopObj = [P;F];
    [N,M]  = size(PopObj);

    %% Calculate the FC value
    FC = CalFC(PopObj,Zmin);

    %% Projection
    % Identify the ideal point
    Zmin = min(PopObj,[],1);
    % Identify the extreme points
    W = zeros(M) + 1e-6;
    W(logical(eye(M))) = 1;
    ASF = zeros(N,M);
    for i = 1 : M
        ASF(:,i) = max((PopObj-repmat(Zmin,N,1))./repmat(W(i,:),N,1),[],2);
    end
    [~,extreme] = min(ASF,[],1);
    % Calculate the intercepts
    Hyperplane = PopObj(extreme,:)\ones(M,1);
    a = (1./Hyperplane)';
    if any(isnan(a))
        a = max(PopObj,[],1);
    end
    % Normalization
    PopObj = (PopObj-repmat(Zmin,N,1))./repmat(a-Zmin,N,1);
    % Projection
    PopObj = PopObj./(repmat(sum(PopObj,2),1,M));
    
    %% Select the solutions in the last front one by one
    Choose = false(1,N);
    Choose(1:size(P,1)) = true;
    % The distance between each two solutions
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    while sum(Choose) < Total
        % Direction-based selection
        if ~any(Choose)
            Dis = sort(Distance(~Choose,~Choose),2);
        else
            Dis = sort(Distance(~Choose,Choose),2);
        end
        DD       = sum(1./(Dis(:,1:min(K,end))),2);
        [~,rank] = sort(DD);
        Remain   = find(~Choose);
        R        = Remain(rank(1:min(L,end)));
        % Convergence-based roulette-wheel selection
        Choose(R(RouletteWheelSelection(1,FC(R)))) = true;
    end
    Choose = Choose(size(P,1)+1:end);
end
```

### `MaOEADDFC.m`
```matlab
classdef MaOEADDFC < ALGORITHM
% <2015> <multi/many> <real/integer/label/binary/permutation>
% Many-objective evolutionary algorithm based on directional diversity and
% favorable convergence
% K --- 5 --- The number of neighbors for estimating density
% L --- 3 --- The number of candidates for convergence-based selection

%------------------------------- Reference --------------------------------
% J. Cheng, G. G. Yen, and G. Zhang. A many-objective evolutionary
% algorithm with enhanced mating and environmental selections. IEEE
% Transactions on Evolutionary Computation, 2015, 19(4): 592-605.
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
            [K,L] = Algorithm.ParameterSet(5,3);

            %% Generate random population
            Population = Problem.Initialization();
            Zmin       = min(Population.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = MatingSelection(Population.objs,Zmin);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Zmin       = min([Zmin;Offspring.objs],[],1);
                Population = EnvironmentalSelection([Population,Offspring],Zmin,Problem.N,K,L);
            end
        end
    end
end
```

### `MatingSelection.m`
```matlab
function MatingPool = MatingSelection(PopObj,Zmin)
% The mating selection of MaOEA-DDFC

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N  = size(PopObj,1);
    FC = CalFC(PopObj,Zmin);
    Parents1   = randi(N,1,N);
    Parents2   = randi(N,1,N);
    Dominate   = any(PopObj(Parents1,:)<PopObj(Parents2,:),2) - any(PopObj(Parents1,:)>PopObj(Parents2,:),2);
    MatingPool = [Parents1(Dominate==1),...
                  Parents2(Dominate==-1),...
                  Parents1(Dominate==0 & FC(Parents1)<=FC(Parents2)),...
                  Parents2(Dominate==0 & FC(Parents1)>FC(Parents2))];
end
```
