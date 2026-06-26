# EMyO-C

**Tags**: <2014> <multi/many> <real/integer>

## Description
Evolutionary many-objective optimization algorithm with clustering-based

## Reference
R. Denysiuk, L. Costa, and I. E. Santo. Clustering-based selection for evolutionary many-objective optimization. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2014, 538-547.

## Source Code

### `Clustering.m`
```matlab
function [C] = Clustering(data, k, varargin)
% Perform hierarchical agglomerative clustering to group points in
% matrix data into k clusters

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    % get size of data
    n = size(data, 1);
    
    % compute distances between points
    distPoints = zeros(n);
    for i = 1:n
        distPoints(i,:) = sqrt(sum(power(repmat(data(i,:),n,1)-data, 2), 2));
        distPoints(i,i) = inf;
    end
    
    % initialize distances between clusters
    distClusters = distPoints;
    
    % initially, each points belongs to a distinct cluster
    C = cell(n, 1);
    for i = 1:n;
        C{i} = i;
    end
    
    update = 0;
    toRemove = zeros(n-k, 1);
    for i = 1:n-k
        
        % update distances between clusters
        if update>0
            
            for jj = 1:n
                
                if isempty(C{jj}) || jj == update
                    continue
                end
                
                % calculate distances between clusters
                distClusters(update, jj) = sum(sum(distPoints(C{update}, C{jj})))/(numel(C{update})*numel(C{jj}));
                
            end
            
        end
        
        % find two clusters with min distance
        [a, b] = find(distClusters==min(min(distClusters)));
        
        % merge these clusters
        C{a(1)} = horzcat(C{a(1)}, C{b(1)});
        
        % removed cluster
        toRemove(i) = b(1);
        C{toRemove(i)} = [];
        update = a(1);
        
        % remove from distance matrix
        distClusters(toRemove(i), :) = inf;
        distClusters(:, toRemove(i)) = inf;
        
    end
    
    % final clusters
    C(toRemove) = [];
end
```

### `EMyOC.m`
```matlab
classdef EMyOC < ALGORITHM
% <2014> <multi/many> <real/integer>
% Evolutionary many-objective optimization algorithm with clustering-based
% selection

%------------------------------- Reference --------------------------------
% R. Denysiuk, L. Costa, and I. E. Santo. Clustering-based selection for
% evolutionary many-objective optimization. Proceedings of the
% International Conference on Parallel Problem Solving from Nature, 2014,
% 538-547.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            % Initialize population
            Population = Problem.Initialization();
            % Initialize ideal point
            Z = min(Population.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Offspring  = Operator(Problem,Population,Population(randi(Problem.N,1,Problem.N)),Population(randi(Problem.N,1,Problem.N)));
                Z          = min(Z,min(Offspring.objs,[],1));
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,Z);
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population] = EnvironmentalSelection(Population,N,Z)
% The environmental selection of EMyOC

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo<MaxFNo;
    Last = find(FrontNo==MaxFNo);
    
    %% Select the solutions in the last front using clustering based truncation
    index = Truncation(Population(Last).objs,Z,N-sum(Next));
    Next(Last(index)) = true;
    
    %% Population for next generation
    Population = Population(Next);
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Parent1,Parent2,Parent3,Parameter)
% Differential evolution with variable-wise mutation restriction

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    %% Parameter setting
    if nargin > 4
        [CR,proM,disM] = deal(Parameter{:});
    else
        [CR,proM,disM] = deal(0.15,1,20);
    end
    Parent1 = Parent1.decs;
    Parent2 = Parent2.decs;
    Parent3 = Parent3.decs;
    [N,D]   = size(Parent1);

    %% Calculate the difference vectors
    V = Parent2 - Parent3;
    
    %% Do polynomial mutation on the difference vectors
    Lower   = repmat(Problem.lower,N,1);
    Upper   = repmat(Problem.upper,N,1);
    Site    = rand(N,D) < proM/D;
    mu      = rand(N,D);
    temp    = Site & mu<=0.5;
    V(temp) = V(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)).^(1/(disM+1))-1);
    temp    = Site & mu>0.5; 
    V(temp) = V(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))).^(1/(disM+1)));

    %% Restrict the difference vectors
    V = min(max(V,(Lower-Upper)/2),(Upper-Lower)/2);

    %% Generate offsprings
    Site         = rand(N,D) < CR;
    OffDec       = Parent1;
    OffDec(Site) = OffDec(Site) + V(Site);
    Offspring    = Problem.Evaluation(OffDec);
end
```

### `Truncation.m`
```matlab
function [index] = Truncation(F, Z, remaining)
% Truncation of EMyO/C, based on clustering

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    % get size
    [n, m] = size(F);
    
    % calculate distances to reference point
    Dist = sqrt(sum((F - repmat(Z, n, 1)).^2, 2));
    
    % project to the hyperplane
    F = (F - repmat(Z, n, 1))./repmat(sum(F - repmat(Z, n, 1), 2), 1, m);
    
    % compute clusters
    [C] = Clustering(F, remaining);
    
    % find number of individuals in each cluster
    numInd = zeros(numel(C), 1);
    for i = 1 : numel(C)
        numInd(i) = numel(C{i});
    end
    
    index = zeros(remaining,1);
    for i = 1 : remaining
        
        % get ind from cluster
        tempInd = C{i};
        
        % get individuals' distances to reference point 
        tempDist = Dist(tempInd);
        
        % find cluster's representative
        [~, idx] = min(tempDist);
        index(i) = tempInd(idx(1));
        
    end

end
```
