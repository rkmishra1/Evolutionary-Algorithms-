# MFO-SPEA2

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Multiform optimization framework based on SPEA2

## Reference
R. Jiao, B. Xue, and M. Zhang. A multiform optimization framework for constrained multiobjective optimization. IEEE Transactions on Cybernetics, 2023, 53(8):5165-5177.

## Source Code

### `CalFitness.m`
```matlab
function [Fitness, rank] = CalFitness(PopObj, PopCon)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0, PopCon), 2);
    end
    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate, 2);
    
    %% Calculate R(i)
    R = zeros(1, N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:, i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj, PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance, 2);
    D = 1./(Distance(:, floor(sqrt(N))) + 2);
    
    %% Calculate the fitnesses
    Fitness  = R + D';
    [~,rank] = sortrows(Fitness');
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population, rankP] = EnvironmentalSelection(Population, N, PopCon)
% The environmental selection of MFO-SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    %% Calculate the fitness of each solution
    [Fitness, rankP] = CalFitness(Population.objs, PopCon);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~, Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs, sum(Next) - N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    rankP      = rankP(Next);
end

function Del = Truncation(PopObj, K)
% Select part of the solutions by truncation
    %% Truncation
    Distance = pdist2(PopObj, PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1, size(PopObj, 1));
    while sum(Del) < K
        Remain    = find(~Del);
        Temp      = sort(Distance(Remain, Remain), 2);
        [~, Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `MFOSPEA2.m`
```matlab
classdef MFOSPEA2 < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <constrained>
% Multiform optimization framework based on SPEA2

%------------------------------- Reference --------------------------------
% R. Jiao, B. Xue, and M. Zhang. A multiform optimization framework for
% constrained multiobjective optimization. IEEE Transactions on
% Cybernetics, 2023, 53(8):5165-5177.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    methods
        function main(Algorithm, Problem)
            %% Generate random population for target task
            TargetPop  = Problem.Initialization();
            %% Generate random population for source task
            SourcePop  = TargetPop;   
            initialE   = max(max(0,TargetPop.cons), [], 1);
            initialE(initialE<1) = 1;
            [~, rankTP] = CalFitness(TargetPop.objs, TargetPop.cons);
            [~, rankSP] = CalFitness(SourcePop.objs, SourcePop.cons-initialE);

            %% Optimization
            while Algorithm.NotTerminated(TargetPop)
                epsn       = ReduceBoundary(initialE, ceil(Problem.FE/Problem.N), ceil(Problem.maxFE/Problem.N)-1);
                MatingPool = TournamentSelection(2, Problem.N, [rankTP', rankSP']);
                P          = [TargetPop, SourcePop];
                Offspring  = OperatorGA(Problem, P(MatingPool));
                %% Environmental selection for target task
                [TargetPop, rankTP] = EnvironmentalSelection([TargetPop, Offspring], Problem.N, [TargetPop.cons; Offspring.cons]);
                %% Environmental selection for source task
                [SourcePop, rankSP] = EnvironmentalSelection([SourcePop, Offspring], Problem.N, [SourcePop.cons-epsn; Offspring.cons-epsn]);
            end
        end
    end
end

function epsn = ReduceBoundary(eF, k, MaxK)
    %% Reduce the epsilon constraint boundary for source task
    z        = 1e-8;
    Nearzero = 1e-15;
    B        = MaxK./power(log((eF + z)./z), 1.0./10);
    B(B==0)  = B(B==0) + Nearzero;
    f        = eF.* exp( -(k./B).^10 );
    tmp      = find(abs(f-z) < Nearzero);
    f(tmp)   = f(tmp).*0 + z;
    epsn     = f - z;
    epsn(epsn<=0) = 0;
end
```
