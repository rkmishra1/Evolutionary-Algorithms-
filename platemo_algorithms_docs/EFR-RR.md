# EFR-RR

**Tags**: <2016> <multi/many> <real/integer/label/binary/permutation>

## Description
Ensemble fitness ranking with a ranking restriction scheme

## Reference
Y. Yuan, H. Xu, B. Wang, B. Zhang, and X. Yao. Balancing convergence and diversity in decomposition-based many-objective optimizers. IEEE Transactions on Evolutionary Computation, 2016, 20(2): 180-198.

## Source Code

### `EFRRR.m`
```matlab
classdef EFRRR < ALGORITHM
% <2016> <multi/many> <real/integer/label/binary/permutation>
% Ensemble fitness ranking with a ranking restriction scheme
% K --- 2 --- Number of nearest weight vectors

%------------------------------- Reference --------------------------------
% Y. Yuan, H. Xu, B. Wang, B. Zhang, and X. Yao. Balancing convergence and
% diversity in decomposition-based many-objective optimizers. IEEE
% Transactions on Evolutionary Computation, 2016, 20(2): 180-198.
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
            K = Algorithm.ParameterSet(2);

            %% Generate the reference points and random population
            [W,Problem.N]   = UniformPoint(Problem.N,Problem.M);
            Population      = Problem.Initialization();
            [PopObj,z,znad] = Normalization(Population.objs,min(Population.objs),max(Population.objs));
            RgFrontNO       = MaximumRanking(PopObj,W,K);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,RgFrontNO);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,z,znad] = EnvironmentalSelection([Population,Offspring],W,Problem.N,K,z,znad);
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,z,znad] = EnvironmentalSelection(Population,W,N,K,z,znad)
% The environmental selection of EFR-RR

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Normalization
    [PopObj,z,znad] = Normalization(Population.objs,z,znad);

    %% Environmental selection
    RgFrontNo = MaximumRanking(PopObj,W,K);
    MaxFNo    = find(cumsum(hist(RgFrontNo,1:max(RgFrontNo)))>=N,1);
    LastFront = find(RgFrontNo==MaxFNo);
    LastFront = LastFront(randperm(length(LastFront)));
    RgFrontNo(LastFront(1:sum(RgFrontNo<=MaxFNo)-N)) = inf;
    Next      = RgFrontNo <= MaxFNo;
    % Population for next generation
    Population = Population(Next);
end
```

### `MaximumRanking.m`
```matlab
function RgFrontNO = MaximumRanking(PopObj,W,K)
% Get the front of each solution by calculating the global rank

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N  = size(PopObj,1);
    NW = size(W,1);

    %% Find the K nearest weight vectors of each solution
    [~,rank] = sort(1-pdist2(PopObj,W,'cosine'),2,'descend');
    L        = rank(:,1:K);
    
    %% Calculate the modified Tchebycheff function value
	g = inf(N,NW);
    for i = 1 : N
        g(i,L(i,:)) = max(repmat(PopObj(i,:),K,1)./W(L(i,:),:),[],2)';
    end

    %% Calculate the global rank of each solution
    [~,rank]  = sort(g,1);
    [~,r]     = sort(rank,1);
    g(g~=inf) = 1;
    Rg        = min(r.*g,[],2);
    
    %% Get the front of each solution
    [~,~,RgFrontNO] = unique(Rg);
end
```

### `Normalization.m`
```matlab
function [PopObj,z,znad] = Normalization(PopObj,z,znad)
% Normalize the population and update the ideal point and the nadir point

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);

    %% Update the ideal point
    z = min(z,min(PopObj,[],1));
    
    %% Update the nadir point
    % Identify the extreme points
    W = zeros(M) + 1e-6;
    W(logical(eye(M))) = 1;
    ASF = zeros(N,M);
    for i = 1 : M
        ASF(:,i) = max(abs((PopObj-repmat(z,N,1))./(repmat(znad-z,N,1)))./repmat(W(i,:),N,1),[],2);
    end
    [~,extreme] = min(ASF,[],1);
    % Calculate the intercepts
    Hyperplane = (PopObj(extreme,:)-repmat(z,M,1))\ones(M,1);
    a = (1./Hyperplane)' + z;
    if any(isnan(a)) || any(a<=z)
        a = max(PopObj,[],1);
    end
    znad = a;
    
    %% Normalize the population
    PopObj = (PopObj-repmat(z,N,1))./(repmat(znad-z,N,1));
end
```
