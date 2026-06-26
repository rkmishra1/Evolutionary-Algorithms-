# MyO-DEMR

**Tags**: <2013> <multi/many> <real/integer>

## Description
Many-objective differential evolution with mutation restriction

## Reference
R. Denysiuk, L. Costa, and I. E. Santo. Many-objective optimization using differential evolution with variable-wise mutation restriction. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2013, 591-598.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population] = EnvironmentalSelection(Population,N,P)
% The environmental selection of MyO-DEMR

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
    [index] = Truncation(Population(Last).objs, P, N-sum(Next));
    Next(Last(index)) = true;
    
    %% Population for next generation
    Population = Population(Next);
end
```

### `MyODEMR.m`
```matlab
classdef MyODEMR < ALGORITHM
% <2013> <multi/many> <real/integer>
% Many-objective differential evolution with mutation restriction
% nP --- 500 --- Number of reference points for IGD calculation

%------------------------------- Reference --------------------------------
% R. Denysiuk, L. Costa, and I. E. Santo. Many-objective optimization using
% differential evolution with variable-wise mutation restriction.
% Proceedings of the Annual Conference on Genetic and Evolutionary
% Computation, 2013, 591-598.
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
            %% Parameter setting
            nP = Algorithm.ParameterSet(500);

            %% Generate hyperplane 
            P = UniformPoint(nP,Problem.M);

            %% Generate random population
            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Offspring  = Operator(Problem,Population(1:Problem.N),Population(randi(Problem.N,1,Problem.N)),Population(randi(Problem.N,1,Problem.N)));
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,P);
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Parent1,Parent2,Parent3,Parameter)
% Differental evolution with variable-wise mutation restriction

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
function [next] = Truncation(F, P, remaining)
% Truncation of MyO-DEMR, based on IGD

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    % get sizes
    [N1, ~] = size(F);
    [N2, ~] = size(P);
    
    % normalize objectives
    range = max(F)-min(F);
    range(range==0) = 1;
    F = (F - repmat(min(F), N1, 1))./repmat(range, N1, 1);
    
    % build utopian front
    R = P + min( min(sum(F,2)-1,0) );
    
    % calculate distance matrix
    dist = pdist2(R,F);
    
    % select based on IGD
    order = randperm(N2);
    next  = zeros(remaining,1);
    for i = 1 : remaining
        p = rem(i,N2); % point in ref set
        if p == 0
            p = N2;
        end
        [~, indexes] = sort( dist(order(p),:) );
        for idx = indexes
            if ~any(idx == next)
                next(i) = idx;
                break;
            end
        end
    end

end
```
