# MTDE-MKTA

**Tags**: <2025> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>

## Description
Multitasking differential evolution with multiple knowledge types and transfer adaptation

## Reference
Y. Li and W. Gong. Multiobjective multitask optimization with multiple knowledge types and transfer adaptation. IEEE Transactions on Evolutionary Computation, 2025, 29(1): 205-216.

## Source Code

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `Generation.m`
```matlab
function offspring = Generation(Problem,population, rank, model, t,Tau1,Tau2)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for i = 1 : length(population{t})
        offspring(i) = population{t}(i);
        
        % Parameter disturbance
        F  = normrnd(population{t}(i).add(1), 0.1);
        F  = min(max(F, 0.2), 1.2);
        CR = normrnd(population{t}(i).add(2), 0.1);
        CR = min(max(CR, 0), 1);
        TR = normrnd(population{t}(i).add(3), 0.1);
        TR = min(max(TR, 0), 1);
        KP = normrnd(population{t}(i).add(4), 0.1);
        if KP < 0
            KP = 1 + KP;
        elseif KP > 1
            KP = KP - 1;
        end
    
        % Parameter mutation
        if rand() < Tau1
            F = 0.2 + rand();
        end
        if rand() < Tau1
            CR = rand();
        end
        if rand() < Tau2
            TR = rand();
        end
        if rand() < Tau2
            KP = rand();
        end
    
        % Select individuals (rank-DE)
        Np = length(population{t});
        x1 = randi(Np);
        while rand() > (Np - rank{t}(x1)) / Np || x1 == i
            x1 = randi(Np);
        end
        x2 = randi(Np);
        while rand() > (Np - rank{t}(x2)) / Np || x2 == i || x2 == x1
            x2 = randi(Np);
        end
        x3 = randi(Np);
        while x3 == i || x3 == x1 || x3 == x2
            x3 = randi(Np);
        end
        xDeci = population{t}(i).dec;
        xDec1 = population{t}(x1).dec;
        xDec2 = population{t}(x2).dec;
        xDec3 = population{t}(x3).dec;
    
        % Knowledge transfer
        if rand() < TR
            k = randi(length(population)); % help task
            while (k == t), k = randi(length(population)); end
            Np = length(population{k});
    
            rnd = KP;
            if rnd > 1/2
                xDeck = population{k}(randi(Np)).dec;
            else
                xDeck = population{k}(randi(Np)).dec;
                xDeck = (xDeck -model{k}.mean) ./ model{k}.std;
                xDeck = model{t}.mean + model{t}.std .* xDeck;
            end
            xDec2 = xDeck;
        end
    
        offspringdec = xDec1 + F * (xDec2 - xDec3);
        offspringdec = DE_Crossover(offspringdec, xDeci, CR);
        offspringdec = min(max(offspringdec, 0), 1);
        offspringdec(end) = t;
        offspring(i) = Problem.Evaluation(offspringdec,[F,CR,TR,KP]);
    end
end

function OffDec = DE_Crossover(OffDec, ParDec, CR)
    replace = rand(1, size(OffDec, 2)) > CR;
    replace(randi(size(OffDec, 2))) = false;
    OffDec(:, replace) = ParDec(:, replace);
end
```

### `MTDEMKTA.m`
```matlab
classdef MTDEMKTA < ALGORITHM
% <2025> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>
% Multitasking differential evolution with multiple knowledge types and transfer adaptation

%------------------------------- Reference --------------------------------
% Y. Li and W. Gong. Multiobjective multitask optimization with multiple
% knowledge types and transfer adaptation. IEEE Transactions on
% Evolutionary Computation, 2025, 29(1): 205-216.
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
            Tau1  = 0.2;
            Tau2  = 0.1;
            ProbT = size(Problem.SubM,2);
            ProbN = Problem.N/2;

            %% Population initialization
            for t = 1:ProbT
                SubPopulation{t} = [];
                for i = 1 : ProbN
                    Dec = [rand(1, max(Problem.SubD)),t];
                    F   = 0.2 + rand();
                    CR  = rand();
                    TR  = rand(); % Knowledge transfer rate
                    KP  = rand(); % Knowledge type proportion
                    Slo = Problem.Evaluation(Dec,[F,CR,TR,KP]);
                    SubPopulation{t} = [SubPopulation{t},Slo];
                end
                [SubPopulation{t},Fitness{t}] = spea2EnvironmentalSelection(SubPopulation{t},ProbN);
                decs = SubPopulation{t}.decs;
                model{t}.mean = mean(decs);
                model{t}.std  = std(decs) +1e-100;
            end
            
            %% Optimization
            while Algorithm.NotTerminated([SubPopulation{:}])
                for t = 1 : ProbT
                    [~,rank{t}] = sort(Fitness{t});
                    [~,rank{t}] = sort(rank{t});
                    decs  = SubPopulation{t}.decs;
                    alpha = 0.5;
                    model{t}.mean = alpha * model{t}.mean + (1 - alpha) * mean(decs);
                    model{t}.std  = alpha * model{t}.std + (1 - alpha) * (std(decs)) +1e-100;
                end
                for t = 1 : ProbT
                    offspring{t} = Generation(Problem,SubPopulation, rank, model, t,Tau1,Tau2);
                end
                for t = 1 : ProbT
                    SubPopulation{t} = [SubPopulation{t}, offspring{t}];
                    [SubPopulation{t},Fitness{t}] = spea2EnvironmentalSelection(SubPopulation{t},ProbN);
                end
            end
        end
    end
end
```

### `spea2EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = spea2EnvironmentalSelection(Population,N)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    Fitness = CalFitness(Population.objs);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```
