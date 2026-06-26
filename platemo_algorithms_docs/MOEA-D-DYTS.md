# MOEA-D-DYTS

**Tags**: <2020> <multi/many> <real/integer>

## Description
MOEA/D with dynamic Thompson sampling

## Reference
L. Sun and K. Li. Adaptive operator selection based on dynamic Thompson sampling for MOEA/D. Proceedings of the International Conference on Parallel Problem Solving from Nature, 2020, 271-284.

## Source Code

### `BetaSample.m`
```matlab
function op = BetaSample(pds)
% Thompson Sampling-based operator selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    n   = size(pds,2);
    ops = ones(1,n);
    for i = 1 : size(pds,2)
        ops(i) = random(pds(i));
    end
    [~,op] = max(ops);
end
```

### `CreditAssignment.m`
```matlab
function FRR = CreditAssignment(SW,D)
% Credit assignment

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    K      = 4;  % Number of operators
    Reward = zeros(1,K);
    for i = 1 : K
        Reward(i) = sum(SW(2,SW(1,:)==i));
    end
    [~,Rank] = sort(Reward,'descend');
    [~,Rank] = sort(Rank);
    Decay    = D.^Rank.*Reward;
    FRR      = Decay./sum(Decay);
end
```

### `FiveOps.m`
```matlab
function Offspring = FiveOps(Problem,op,x,x1,x2,x3,x4,x5)
% Four different DE operators

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [CR,F,proM,disM,K] = deal(1,0.5,1,20,0.5);
    D     = length(x.dec);
    Lower = Problem.lower;
    Upper = Problem.upper;

    %% Differental evolution
    switch op
        case 1
            % DE/rand/1
            v = x.dec + F*(x1.dec-x2.dec);
        case 2
            % DE/rand/2
            v = x.dec + F*(x1.dec-x2.dec) + F*(x3.dec-x4.dec);
        case 3
            % DE/current-to-rand/2
            v = x.dec + K*(x.dec-x1.dec) + F*(x2.dec-x3.dec) + F*(x4.dec-x5.dec);
        case 4
            % DE/current-to-rand/1
            v = x.dec + K*(x.dec-x1.dec) + F*(x2.dec-x3.dec);
        case 5
            uni = rand(1,D);
            v = x.dec + uni.*(Upper-Lower);

    end
    Offspring = x.dec;
    Site      = rand(1,D) < (CR+(op>2));
    Offspring(Site) = v(Site);

    %% Polynomial mutation
    Site  = rand(1,D) < proM/D;
    mu    = rand(1,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(Offspring);
end
```

### `MOEADDYTS.m`
```matlab
classdef MOEADDYTS < ALGORITHM
% <2020> <multi/many> <real/integer>
% MOEA/D with dynamic Thompson sampling

%------------------------------- Reference --------------------------------
% L. Sun and K. Li. Adaptive operator selection based on dynamic Thompson
% sampling for MOEA/D. Proceedings of the International Conference on
% Parallel Problem Solving from Nature, 2020, 271-284.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Generate the weight vectors
            [Weight, Problem.N] = UniformPoint(Problem.N, Problem.M);
            % Size of neighborhood
            T = 20;
            % Maximum number of solutions replaced by each offspring
            nr = 2;

            %% Detect the neighbours of each solution
            B = pdist2(Weight, Weight);
            [~, B] = sort(B, 2);
            B = B(:, 1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs, [], 1);
            % Utility for each subproblem
            Pi = ones(Problem.N, 1);
            % Old Tchebycheff function value of each solution on its subproblem
            oldObj = max(abs((Population.objs - repmat(Z, Problem.N, 1)) .* Weight), [], 2);

            %% Optimization
            C   = 100;
            pds = repelem(makedist('Beta'), 5);

            for i = 1 : 5
                pds(i) = makedist('Beta', 'a', 1, 'b', 1);
            end
            countOps = zeros(1, 5);

            while Algorithm.NotTerminated(Population)
                for subgeneration = 1 : 5
                    % Choose I
                    Bounday = find(sum(Weight < 1e-3, 2) == Problem.M - 1)';
                    I       = [Bounday, TournamentSelection(10, floor(Problem.N / 5) - length(Bounday), -Pi)];

                    % For each solution in I
                    for i = I
                        % Bandit-based operator selection
                        op = BetaSample(pds);
                        countOps(op) = countOps(op) + 1;

                        % Choose the parents
                        if rand < 0.8
                            P = B(i, randperm(end));
                        else
                            P = randperm(Problem.N);
                        end

                        % Generate an offspring
                        Offspring = FiveOps(Problem, op, Population(i), Population(P(1)), Population(P(2)), Population(P(3)), Population(P(4)), Population(P(5)));

                        % Update the ideal point
                        Z = min(Z, Offspring.obj);

                        % Update the solutions in P by Tchebycheff approach
                        g_old   = max(abs(Population(P).objs - repmat(Z, length(P), 1)) .* Weight(P, :), [], 2);
                        g_new   = max(repmat(abs(Offspring.obj - Z), length(P), 1) .* Weight(P, :), [], 2);
                        replace = find(g_old >= g_new, nr);
                        Population(P(replace)) = Offspring;
                        r = any(replace);
                        a = pds(op).a; b = pds(op).b;
                        if a + b < C
                            a = a + r;
                            b = b + 1 - r;
                        else
                            a = (a + r) * C / (C + 1);
                            b = (b + 1 - r) * C / (C + 1);
                        end
                        pds(op).a = a;
                        pds(op).b = b;
                    end
                end
                if ~mod(ceil(Problem.FE / Problem.N), 10)
                    % Update Pi for each solution
                    newObj    = max(abs((Population.objs - repmat(Z, Problem.N, 1)) .* Weight), [], 2);
                    DELTA     = (oldObj - newObj) ./ oldObj;
                    Temp      = DELTA < 0.001;
                    Pi(~Temp) = 1;
                    Pi(Temp)  = (0.95 + 0.05 * DELTA(Temp) / 0.001) .* Pi(Temp);
                    oldObj    = newObj;
                end
            end
        end
    end
end
```
