# NSLS

**Tags**: <2015> <multi> <real/integer>

## Description
Multiobjective optimization framework based on nondominated sorting and

## Reference
B. Chen, W. Zeng, Y. Lin, and D. Zhang. A new local search-based multiobjective optimization algorithm. IEEE Transactions on Evolutionary Computation, 2015, 19(1): 50-73.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N)
% The environmental selection of NSLS

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
    Choose = LastSelection(Population(Last).objs,N-sum(Next));
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Choose = LastSelection(PopObj,K)
% Select part of the solutions in the last front

    N = size(PopObj,1);
    
    %% Select the extreme solutions first
    Choose = false(1,N);
    [~,Extreme] = min(PopObj,[],1);
    Choose(Extreme) = true;
    [~,Extreme] = max(PopObj,[],1);
    Choose(Extreme) = true;
    
    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > K
        Choosed = find(Choose);
        k = randperm(sum(Choose),sum(Choose)-K);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < K
        Distance = pdist2(PopObj,PopObj);
        Distance(logical(eye(length(Distance)))) = inf;
        while sum(Choose) < K
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
    end
end
```

### `NSLS.m`
```matlab
classdef NSLS < ALGORITHM
% <2015> <multi> <real/integer>
% Multiobjective optimization framework based on nondominated sorting and
% local search

%------------------------------- Reference --------------------------------
% B. Chen, W. Zeng, Y. Lin, and D. Zhang. A new local search-based
% multiobjective optimization algorithm. IEEE Transactions on Evolutionary
% Computation, 2015, 19(1): 50-73.
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

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Offspring  = Operator(Problem,Population);
                Population = EnvironmentalSelection([Population,Offspring],Problem.N);
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Population = Operator(Problem,Population,Parameter)
% <operator> <real>
% The local search operator in NSLS
% mu    --- 0.5 --- The mean value of the Gaussian distribution
% delta --- 0.1 --- The standard deviation of the Gaussian distribution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    if nargin > 2
        [mu,delta] = deal(Parameter{:});
    else
        [mu,delta] = deal(0.5,0.1);
    end
    [N,D] = size(Population.decs);
    
    %% Local search
    for i = 1 : N
        for d = 1 : D
            c = mu + delta*randn;
            k = randi(N,1,2);
            w = repmat(Population(i).dec,2,1);
            w(1,d) = w(1,d) + c*(Population(k(1)).dec(d) - Population(k(2)).dec(d));
            w(2,d) = w(2,d) - c*(Population(k(1)).dec(d) - Population(k(2)).dec(d));
            w = Problem.Evaluation(w);
           	for j = 1 : 2
                k(j) = any(w(j).obj<Population(i).obj) - any(w(j).obj>Population(i).obj);
            end
            if k(1) == -1 && k(2) == -1
                continue;
            elseif k(1) > k(2)
                k = 1;
            elseif k(1) < k(2)
                k = 2;
            else
                k = randi(2);
            end
            Population(i) = w(k);
        end
    end
end
```
