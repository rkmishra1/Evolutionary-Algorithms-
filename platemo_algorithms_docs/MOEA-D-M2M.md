# MOEA-D-M2M

**Tags**: <2014> <multi> <real/integer>

## Description
MOEA/D based on MOP to MOP

## Reference
H. Liu, F. Gu, and Q. Zhang. Decomposition of a multiobjective optimization problem into a number of simple multiobjective subproblems. IEEE Transactions on Evolutionary Computation, 2014, 18(3): 450-455.

## Source Code

### `Associate.m`
```matlab
function Population = Associate(Population,W,S)
% Allocation of solutions to subproblems

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    K = size(W,1);
    
    %% Allocation of solutions to subproblems
    % Transformation
    [~,transformation] = max(1-pdist2(Population.objs,W,'cosine'),[],2);
    partition          = zeros(S,K);
    % Allocation
    for i = 1 : K
        current = find(transformation==i);
        if length(current) < S
            % Randomly select solutions and join to the current subproblem
            current = [current;randi(length(Population),S-length(current),1)];
        elseif length(current) > S
            % Delete solutions from the current subproblem by non-dominated
            % sorting and crowding distance
            [FrontNo,MaxFNo] = NDSort(Population(current).objs,S);
            Last     = find(FrontNo==MaxFNo);
            CrowdDis = CrowdingDistance(Population(current(Last)).objs);
            [~,rank] = sort(CrowdDis);
            FrontNo(Last(rank(1:sum(FrontNo<=MaxFNo)-S))) = inf;
            current  = current(FrontNo<=MaxFNo);
        end
        partition(:,i) = current;
    end
    Population = Population(partition(:));
end
```

### `MOEADM2M.m`
```matlab
classdef MOEADM2M < ALGORITHM
% <2014> <multi> <real/integer>
% MOEA/D based on MOP to MOP
% K --- 10 --- Number of reference vectors

%------------------------------- Reference --------------------------------
% H. Liu, F. Gu, and Q. Zhang. Decomposition of a multiobjective
% optimization problem into a number of simple multiobjective subproblems.
% IEEE Transactions on Evolutionary Computation, 2014, 18(3): 450-455.
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
            K = Algorithm.ParameterSet(10);

            %% Generate random population
            [W,K]      = UniformPoint(K,Problem.M);
            Problem.N  = ceil(Problem.N/K)*K;
            S          = Problem.N/K;
            Population = Problem.Initialization();
            Population = Associate(Population,W,S);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPoolLocal      = randi(S,S,K) + repmat(0:S:S*(K-1),S,1);
                MatingPoolGlobal     = randi(Problem.N,1,Problem.N);
                rnd                  = rand(S,K) < 0.7;
                MatingPoolLocal(rnd) = MatingPoolGlobal(rnd);
                Offspring  = Operator(Problem,Population,Population(MatingPoolLocal(:)));
                Population = Associate([Population,Offspring],W,S);
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Parent1,Parent2)
% Crossover and mutation used in MOEA/D-M2M
% proM --- 1 --- The expectation of number of bits doing mutation 

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    Parent1 = Parent1.decs;
    Parent2 = Parent2.decs;
    [N,D]   = size(Parent1);

    %% Crossover
    rc     = (2*rand(N,1)-1).*(1-rand(N,1).^(-(1-Problem.FE/Problem.maxFE).^0.7));
    OffDec = Parent1 + repmat(rc,1,D).*(Parent1-Parent2);
    
    %% Mutation
    rm    = 0.25*(2*rand(N,D)-1).*(1-rand(N,D).^(-(1-Problem.FE/Problem.maxFE).^0.7));
    Site  = rand(N,D) < 1/D;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    OffDec(Site) = OffDec(Site) + rm(Site).*(Upper(Site)-Lower(Site));
                     
	%% Set the infeasible decision variables to feasible values
    temp1 = OffDec < Lower;
    temp2 = OffDec > Upper;
    rnd   = rand(N,D);
    OffDec(temp1) = Lower(temp1) + 0.5*rnd(temp1).*(Parent1(temp1)-Lower(temp1));
    OffDec(temp2) = Upper(temp2) - 0.5*rnd(temp2).*(Upper(temp2)-Parent1(temp2));
    Offspring     = Problem.Evaluation(OffDec);
end
```
