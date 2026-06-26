# CMOEA-CD

**Tags**: <2025> <multi/many> <real/binary/permutation><constrained/none>

## Description
Constraint-Pareto dominance and diversity enhancement strategy based constrained MOEA

## Reference
Z. Liu, F. Han, Q. Ling, H. Han, and J. Jiang. Constraint-Pareto dominance and diversity enhancement strategy based evolutionary algorithm for solving constrained multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2771-2784.

## Source Code

### `CMOEACD.m`
```matlab
classdef CMOEACD < ALGORITHM
% <2025> <multi/many> <real/binary/permutation><constrained/none>
% Constraint-Pareto dominance and diversity enhancement strategy based constrained MOEA
% e1 --- 1 --- Type of environmental selection for forward exploration(1. SPEA2 2. NSGA-II 3. modified NSGA-III)
% e2 --- 1 --- Type of environmental selection for feasible exploitation(1. SPEA2 2. NSGA-II 3. modified NSGA-III)

%------------------------------- Reference --------------------------------
% Z. Liu, F. Han, Q. Ling, H. Han, and J. Jiang. Constraint-Pareto
% dominance and diversity enhancement strategy based evolutionary algorithm
% for solving constrained multiobjective optimization problems. IEEE
% Transactions on Evolutionary Computation, 2025, 29(6): 2771-2784.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhe Liu

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [e1,e2] = Algorithm.ParameterSet(1,1);
            Ns = floor(Problem.N/3);

            %% Generate random population
            Population = Problem.Initialization();
            FA  = [];
            DA  = [];
            FEA = Population;
            Offspring = Population;
            zmin      = min(Population.objs,[],1) - 1e-6;

            %% Optimization
            while Algorithm.NotTerminated(FEA)
                zmin = min(zmin, min(Offspring.objs, [], 1) - 1e-6);
                FA   = ForwardExplorationArchive(FA, Offspring, zmin, Ns, e1);                   
                DA   = DiversityEnhancementArchive(DA, Offspring, zmin, Ns);
                FEA  = FeasibilityExploitationArchive(FEA, Offspring, Problem.N, e2);
                Pop1 = FA;
                Pop2 = DA;  
                Pop3 = FEA(unidrnd(length(FEA), [1, floor(Problem.N/3)]));      
                MatingPool_Pop1_1 = randperm(length(Pop1));
                MatingPool_Pop1_2 = randperm(length(Pop1));
                MatingPool_Pop2_1 = randperm(length(Pop2));
                MatingPool_Pop2_2 = randperm(length(Pop2));
                MatingPool_Pop3_1 = randperm(length(Pop3));
                MatingPool_Pop3_2 = randperm(length(Pop3));
                if rand() < 0.5
                    Offspring1 = OperatorDE(Problem, Pop1, Pop1(MatingPool_Pop1_1), Pop1(MatingPool_Pop1_2),{1,0.5,1,1});
                    Offspring2 = OperatorDE(Problem, Pop2, Pop2(MatingPool_Pop2_1), Pop2(MatingPool_Pop2_2),{1,0.5,1,1});
                    Offspring3 = OperatorDE(Problem, Pop3, Pop3(MatingPool_Pop3_1), Pop3(MatingPool_Pop3_2),{1,0.5,1,1});
                else
                    Offspring1 = OperatorGA(Problem, Pop1(MatingPool_Pop1_1), {1,20,1,1});
                    Offspring2 = OperatorGA(Problem, Pop2(MatingPool_Pop2_1), {1,20,1,1});
                    Offspring3 = OperatorGA(Problem, Pop3(MatingPool_Pop3_1), {1,20,1,1});
                end
                Offspring = [Offspring1, Offspring2, Offspring3];
            end
        end
    end
end
```

### `DiversityEnhancementArchive.m`
```matlab
function DA = DiversityEnhancementArchive(DA, Offspring, zmin, Ns)
% The Diversity Enhancement Archive of CMOEA-CD

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhe Liu

    %% Dominance relation calculation
    S  = [DA, Offspring];
    DA = [];
    NonDominated = DominationCal(S, 1);
    S   = S(NonDominated);
    Obj = S.objs;
    Con = S.cons;
    CV  = sum(max(0,Con),2);
    %% Minimum angular distance calculation 
    [~, M]  = size(Obj);
    [W, Ns] = UniformPoint(Ns, M); 
    Angle_W_to_W = acos(1 - pdist2(W,W,'cosine'));
    Angle_W_to_W(eye(Ns) == 1) = inf;
    h = mean(min(Angle_W_to_W));

    %% Enviornmental selection
    if sum(CV <= 0) > 0
        zmax = max(Obj(CV <= 0, :), [], 1);
    else
        [~,index] = min(CV);
        zmax = Obj(index,:);
    end                    
    Obj          = (Obj - zmin) ./ (zmax - zmin);
    Angle_S_to_W = sin(acos(1 - pdist2(W,Obj,'cosine')));
    for i = 1 : Ns
        Angle = Angle_S_to_W(i,:);
        list  = Angle <= h;
        if sum(list) == 0
            [~,index]   = min(Angle_S_to_W(i,:));
            list(index) = true;
        end
        T        = inf(length(S), 1);
        T(list)  = CV(list);
        feasible = T <= 0;
        if sum(feasible) <= 0
            [~,index] = min(T);
        else
            T = inf(size(Angle));
            Fitness     = Angle_S_to_W(i,:);
            T(feasible) = Fitness(feasible);
            [~,index]   = min(T);
        end
        DA = [DA,S(index)];
    end        
end
```

### `DominationCal.m`
```matlab
function NonDominated = DominationCal(Population, Add)
% The dominance relation calculation of CMOEA-CD
% add = 0 --- Pareto dominance
% add = 1 --- contraint-Pareto dominance

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhe Liu

    PopObj = Population.objs;
    PopObj = roundn(PopObj, -10);
    [N, M] = size(PopObj);
    Dominated = false(1, N);
    PopCon    = Population.cons;
    Cons      = sum(max(0,PopCon),2);
    for i = 1 : N-1
        err = PopObj(i,:) - PopObj;
        eq  = zeros(N, 1);
        max_err = max(err, [],  2);
        min_err = min(err, [],  2);
        for j = i + 1: N
            for k = 1 : M
                if err(j, k) ~= 0
                    break;
                end
                if k == M
                    eq(j) = 1;
                end
            end
            if Add == 0
                if eq(j) == 1
                    Dominated(j) = true;
                elseif min_err(j) >= 0 
                    Dominated(i) = true;
                elseif max_err(j) <= 0
                    Dominated(j) = true;
                end
            else
                if eq(j) == 1
                    if Cons(i) <= Cons(j)
                        Dominated(j) = true;
                    else
                        Dominated(i) = true;
                    end
                elseif min_err(j) >= 0 
                    if Cons(j) <= 0 || Cons(j) <= Cons(i)
                        Dominated(i) = true;
                    end
                elseif max_err(j) <= 0
                    if Cons(i) <= 0 || Cons(i) <= Cons(j)
                        Dominated(j) = true;
                    end
                end
            end
        end
    end
    NonDominated = ~Dominated;
end
```

### `FeasibilityExploitationArchive.m`
```matlab
function FEA = FeasibilityExploitationArchive(FEA, Offspring, N, Add)
% The Feasible Exploitation Archive of CMOEA-CD
% Add = 1 --- environmental selection of SPEA2
% Add = 2 --- environmental selection of NSGA-II
% Add = 3 --- environmental selection of modified NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhe Liu

    %% Dominance relation calculation
    FEA = [FEA, Offspring];
    NonDominated = DominationCal(FEA, 1);
    FEA = FEA(NonDominated);
    PopObj = FEA.objs;
    [P, M] = size(PopObj);  
    if P <= N
        FEA = FEA([1:P, unidrnd(P,[1, N - P])]);
    else
        PopCon = FEA.cons;
        Cons   = sum(max(0,PopCon),2);   
        if sum(Cons <= 0) <= N
            [~, index] = sort(Cons);
            remained_solution = index(1: N);
            FEA = FEA(remained_solution);
        %% Enviornmental selection
        elseif Add == 1
            FEA    = FEA(Cons <= 0); 
            PopObj = FEA.objs;
            [P, ~] = size(PopObj); 
            zmin   = min(PopObj, [], 1) - 1e-6;
            zmax   = max(PopObj, [], 1);        
            PopObj = (PopObj - zmin) ./ (zmax - zmin);
            Del    = Truncation(PopObj, P - N);
            remained_solution = 1 : P;
            remained_solution(Del) = [];
            FEA = FEA(remained_solution);
        elseif Add == 2
            FEA      = FEA(Cons <= 0); 
            PopObj   = FEA.objs;
            zmin     = min(PopObj, [], 1) - 1e-6;
            zmax     = max(PopObj, [], 1);        
            PopObj   = (PopObj - zmin) ./ (zmax - zmin);
            CrowdDis = CrowdingDistance(PopObj);
            [~,Rank] = sort(CrowdDis,'descend');  
            FEA = FEA(Rank(1:N));       
        elseif Add == 3
            FEA      = FEA(Cons <= 0); 
            PopObj   = FEA.objs; 
            [W, N]   = UniformPoint(N, M); 
            zmin     = min(PopObj, [], 1) - 1e-6;
            zmax     = max(PopObj, [], 1); 
            PopObj   = (PopObj - zmin) ./ (zmax - zmin);
            Distance = sqrt(sum(PopObj.^2, 2));
            Angle_Pop_to_W = sin(acos(1 - pdist2(W, PopObj, 'cosine')));
            S   = FEA;
            FEA = [];
            for i = 1 : N
                Fitness   = Distance' .* Angle_Pop_to_W(i,:);
                [~,index] = min(Fitness);
                FEA = [FEA,S(index)];
            end
        end
    end   
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

### `ForwardExplorationArchive.m`
```matlab
function FA = ForwardExplorationArchive(FA, Offspring, zmin, Ns, Add)
% The Forward Exploration Archive of CMOEA-CD
% Add = 1 --- environmental selection of SPEA2
% Add = 2 --- environmental selection of NSGA-II
% Add = 3 --- environmental selection of modified NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhe Liu

    %% Dominance relation calculation
    FA = [FA, Offspring];
    NonDominated = DominationCal(FA, 0);
    FA = FA(NonDominated);
    PopObj = FA.objs;
    [P, M] = size(PopObj);  
    if P <= Ns
        FA = FA([1:P, unidrnd(P,[1, Ns - P])]);
    elseif Add == 1
        zmax   = max(PopObj, [], 1);        
        PopObj = (PopObj - zmin) ./ (zmax - zmin);
        Del    = Truncation(PopObj, P - Ns);
        remained_solution = 1: P;
        remained_solution(Del) = [];
        FA = FA(remained_solution);
    elseif Add == 2
        zmax     = max(PopObj, [], 1);        
        PopObj   = (PopObj - zmin) ./ (zmax - zmin);
        CrowdDis = CrowdingDistance(PopObj);
        [~,Rank] = sort(CrowdDis,'descend');  
        FA = FA(Rank(1:Ns));       
    elseif Add == 3     
        [W, Ns]  = UniformPoint(Ns, M);     
        zmax     = max(PopObj, [], 1); 
        PopObj   = (PopObj - zmin) ./ (zmax - zmin);
        Distance = sqrt(sum(PopObj.^2, 2));
        Angle_Pop_to_W = sin(acos(1 - pdist2(W, PopObj, 'cosine')));
        S  = FA;
        FA = [];
        for i = 1 : Ns
            Fitness   = Distance' .* Angle_Pop_to_W(i,:);
            [~,index] = min(Fitness);
            FA = [FA,S(index)];
        end
    end   
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
