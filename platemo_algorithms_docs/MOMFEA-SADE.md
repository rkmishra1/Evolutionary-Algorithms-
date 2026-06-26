# MOMFEA-SADE

**Tags**: <2022> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>

## Description
Multi-objective multifactorial evolutionary algorithm with subspace alignment and adaptive differential evolution

## Reference
Z. Liang, H. Dong, C. Liu, W. Liang, and Z. Zhu. Evolutionary multitasking for multiobjective optimization with subspace alignment and adaptive differential evolution. IEEE Transactions on Cybernetics, 2022, 52(4): 2096-2109.

## Source Code

### `Generation.m`
```matlab
function offspring = Generation(Problem,population, Best, DE_Pool, t,RMP,LP,F1,F2,LCR,UCR)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    other_task = 1:length(population);
    other_task(other_task == t) = [];
    for i = 1 : length(population)
        pop_Dec{i} = population{i}.decs;
        pop_Dec{i} = pop_Dec{i}(:,1:end-1);
    end

    x1_task      = other_task(randi(length(other_task)));
    x1_Dec_other = domainAdaption(pop_Dec{t}, pop_Dec{x1_task}, randi(length(population{t}), 1, length(population{t})));
    x2_task      = other_task(randi(length(other_task)));
    x2_Dec_other = domainAdaption(pop_Dec{t}, pop_Dec{x2_task}, randi(length(population{t}), 1, length(population{t})));
    x3_task      = other_task(randi(length(other_task)));
    x3_Dec_other = domainAdaption(pop_Dec{t}, pop_Dec{x3_task}, randi(length(population{t}), 1, length(population{t})));

    x1_Dec_other = [x1_Dec_other,t*ones(size(x1_Dec_other,1),1)];
    x2_Dec_other = [x2_Dec_other,t*ones(size(x2_Dec_other,1),1)];
    x3_Dec_other = [x3_Dec_other,t*ones(size(x3_Dec_other,1),1)];
    for i = 1 : length(population{t})
        offspring(i) = population{t}(i);
        A = randperm(length(population{t}), 4);
        A(A == i) = []; x1 = A(1); x2 = A(2); x3 = A(3);
        CR = LCR + rand() .* (UCR - LCR);

        if rand() < RMP % Random Mating
            x1_Dec = x1_Dec_other(i, :);
            x2_Dec = x2_Dec_other(i, :);
            x3_Dec = x3_Dec_other(i, :);
            switch DE_Pool(i)
                case 1 % DE/best/1/bin
                    offspringDec = population{t}(Best{t}(randi(length(Best{t})))).dec + F1 * (x1_Dec - x2_Dec);
                    offspringDec = DE_Crossover(offspringDec, population{t}(i).dec, CR);
                case 2 % DE/rand/1/bin
                    offspringDec = x1_Dec + F1 * (x2_Dec - x3_Dec);
                    offspringDec = DE_Crossover(offspringDec, population{t}(i).dec, CR);
                case 3 % DE/current-to-rand/1
                    offspringDec = population{t}(i).dec + F2 * (x1_Dec - population{t}(i).dec) + F1 * (x2_Dec - x3_Dec);
            end
        else
            switch DE_Pool(i)
                case 1 % DE/best/1/bin
                    offspringDec = population{t}(Best{t}(randi(length(Best{t})))).dec + F1 * (population{t}(x1).dec - population{t}(x2).dec);
                    offspringDec = DE_Crossover(offspringDec, population{t}(i).dec, CR);
                case 2 % DE/rand/1/bin
                    offspringDec = population{t}(x1).dec + F1 * (population{t}(x2).dec - population{t}(x3).dec);
                    offspringDec = DE_Crossover(offspringDec, population{t}(i).dec, CR);
                case 3 % DE/current-to-rand/1
                    offspringDec = population{t}(i).dec + F2 * (population{t}(x1).dec - population{t}(i).dec) + F1 * (population{t}(x2).dec - population{t}(x3).dec);
            end
        end

        offspringDec(offspringDec > 1) = 1;
        offspringDec(offspringDec < 0) = 0;
        offspringDec(:,end) = t;
        offspring(i) = Problem.Evaluation(offspringDec,[true,DE_Pool(i)]);
    end
end

function OffDec = DE_Crossover(OffDec, ParDec, CR)
    replace = rand(1, size(OffDec, 2)) > CR;
    replace(randi(size(OffDec, 2))) = false;
    OffDec(:, replace) = ParDec(:, replace);
end
```

### `MOMFEASADE.m`
```matlab
classdef MOMFEASADE < ALGORITHM
% <2022> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>
% Multi-objective multifactorial evolutionary algorithm with subspace alignment and adaptive differential evolution

%------------------------------- Reference --------------------------------
% Z. Liang, H. Dong, C. Liu, W. Liang, and Z. Zhu. Evolutionary
% multitasking for multiobjective optimization with subspace alignment and
% adaptive differential evolution. IEEE Transactions on Cybernetics, 2022,
% 52(4): 2096-2109.
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
            RMP   = 1;
            LP    = 30;
            F1    = 0.6;
            F2    = 0.5;
            LCR   = 0.3;
            UCR   = 0.9;
            ProbT = size(Problem.SubM,2);
            for t = 1 : ProbT
                Dec = [rand(Problem.N/2, max(Problem.SubD)),t*ones(Problem.N/2,1)];
                SubPopulation{t} = Problem.Evaluation(Dec,[false*ones(Problem.N/2,1),0*ones(Problem.N/2,1)]);
            end
            STNum  = 3;
            R_Used = [];
            R_Succ = [];
            ProbT  = size(Problem.SubM,2);
            ProbN  = Problem.N/2;          

            %% Optimization
            Gen = 0;
            while Algorithm.NotTerminated([SubPopulation{:}])
                Gen = Gen+1;
                if Gen <= LP
                    DE_Pro = ones(1, STNum);
                else
                    DE_Pro = sum(R_Succ(Gen - LP:Gen - 1, :) + 1, 1) ./ sum(R_Used(Gen - LP:Gen - 1, :) + 1, 1);
                end                
                Best = getBest(SubPopulation);
                for t = 1 : ProbT
                    % DE Strategies
                    DE_Pool{t} = getDEPool(DE_Pro, length(SubPopulation{t}));
                    % Generation
                    for i = 1 : length(SubPopulation{t})
                        SubPopulation{t}(i).add(1) = false;
                    end
                    offspring = Generation(Problem,SubPopulation, Best, DE_Pool{t}, t,RMP,LP,F1,F2,LCR,UCR);
                    % Selection
                    SubPopulation{t} = [SubPopulation{t}, offspring];
                    rank             = NSGA2Sort(SubPopulation{t});
                    SubPopulation{t} = SubPopulation{t}(rank(1:ProbN));
                end
                % DE Strategies Probabilities Updation
                R_Used(Gen, :) = hist([DE_Pool{:}], 1:STNum);
                pop_all = [SubPopulation{:}];
                Adds1   = [];
                Adds2   = [];
                for i = 1 : length(pop_all)
                    Adds1 = [Adds1,pop_all(i).add(1)];                    
                end
                child_idx  = Adds1 == true;
                DE_Succpop = pop_all(child_idx);
                for i = 1 : length(DE_Succpop)                    
                    Adds2 = [Adds2,DE_Succpop(i).add(2)];
                end
                DE_Succ = Adds2;
                R_Succ(Gen, :) = hist(DE_Succ, 1:STNum);
            end
        end
    end
end
```

### `NSGA2Sort.m`
```matlab
function [rank, FrontNo, CrowdDis] = NSGA2Sort(population)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    FrontNo  = NDSort(population.objs, inf);
    CrowdDis = CrowdingDistance(population.objs, FrontNo);
    [~,rank] = sortrows([FrontNo', -CrowdDis']);
end
```

### `domainAdaption.m`
```matlab
function TransferredDec = domainAdaption(TDecs, ODecs, x)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    dim     = size(ODecs, 2);     
    coeff_t = pca(TDecs, 'NumComponents', floor(dim * 0.5));
    coeff_o = pca(ODecs, 'NumComponents', floor(dim * 0.5));

    orth_coeff_t = orth(coeff_t);
    orth_coeff_o = orth(coeff_o);

    Xb = orth_coeff_t * orth_coeff_o' * orth_coeff_t;

    o_pca_sa    = ODecs * Xb;
    o_pca_sa_re = o_pca_sa * coeff_t';
    max_v       = max(max(o_pca_sa_re));
    min_v       = min(min(o_pca_sa_re));

    TransferredDec = (o_pca_sa_re(x, :) - min_v) ./ (max_v - min_v);
end
```

### `getBest.m`
```matlab
function Best = getBest(population)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Best = {};
    for t = 1 : length(population)
        FrontNo = NDSort(population{t}.objs, inf);
        Best{t} = find(FrontNo == 1);
    end
end
```

### `getDEPool.m`
```matlab
function DE_Pool = getDEPool(DE_Pro, N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    DE_Pool  = [];
    roulette = DE_Pro / sum(DE_Pro);
    for i = 1 : N
        r = rand();
        for k = 1 : length(DE_Pro)
            if r <= sum(roulette(1:k))
                DE_Pool(i) = k;
                break;
            end
        end
    end
end
```
