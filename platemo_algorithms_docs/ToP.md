# ToP

**Tags**: <2019> <multi> <real/integer> <constrained>

## Description
Two-phase framework with NSGA-II

## Reference
Z. Liu and Y. Wang. Handling constrained multiobjective optimization problems with constraints in both the decision and objective spaces. IEEE Transactions on Evolutionary Computation, 2019, 23(5): 870-884.

## Source Code

### `DE_cr.m`
```matlab
function Offspring = DE_cr(Problem,Parent1,Parent2,Parent3,Parent4)
% DE operator used in TOP

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [CR,F,proM,disM] = deal(1,0.5,1,20);
    
    if isa(Parent1(1),'SOLUTION')
        evaluated = true;
        Parent_i  = Parent1.decs;
        Parent_r1 = Parent2.decs;
        Parent_r2 = Parent3.decs;
        Parent_r3 = Parent4.decs;
    else
        evaluated = false;
    end
    [N,D] = size(Parent1);

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent_i;
    Offspring(Site) = Offspring(Site) + F*(Parent_r1(Site)-Parent_i(Site))+ F*(Parent_r2(Site)-Parent_r3(Site));

    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `DE_rb.m`
```matlab
function Offspring = DE_rb(Problem,Parent1,Parent2,Parent3,Parent4)
% DE operator used in TOP

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [CR,F,proM,disM] = deal(1,0.5,1,20);
    
    if isa(Parent1(1),'SOLUTION')
        evaluated   = true;
        Parent_best = Parent1.decs;
        Parent_r1   = Parent2.decs;
        Parent_r2   = Parent3.decs;
        Parent_r3   = Parent4.decs;
    else
        evaluated = false;
    end
    [N,D] = size(Parent1);

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent_r1;
    Offspring(Site) = Offspring(Site) + F*(Parent_best(Site)-Parent_r1(Site))+ F*(Parent_r2(Site)-Parent_r3(Site));

    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `ToP.m`
```matlab
classdef ToP < ALGORITHM
% <2019> <multi> <real/integer> <constrained>
% Two-phase framework with NSGA-II

%------------------------------- Reference --------------------------------
% Z. Liu and Y. Wang. Handling constrained multiobjective optimization
% problems with constraints in both the decision and objective spaces. IEEE
% Transactions on Evolutionary Computation, 2019, 23(5): 870-884.
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
            CVP        = sum(max(0,Population.cons),2);
            Pf         = sum(CVP==0)/size(Population,2);	% Pf is the ratio of feasible solutions of current population
            Delta      = 1;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if (Delta>=0.2 || Pf<=1/3) && Problem.FE<=0.9*Problem.maxFE
                    %% Phase I
                    fp_transformed = sum(Population.objs,2)/Problem.M; % weighted objective of parent population
                    % generate offspring
                    Offspring = [];
                    for i = 1 : Problem.N
                        if rand < 0.5               % Perform DE/current-to-rand/1
                            MatingPool  = [i,randi(Problem.N,1,3)];
                            Offspring_i = DE_cr(Problem,Population(MatingPool(1)),Population(MatingPool(2)),Population(MatingPool(3)),Population(MatingPool(4)));
                            Offspring   = cat(2,Offspring,Offspring_i);
                        else                        % Perform DE/rand-to-best/1/bin
                            [~,bestindex] = min(fp_transformed);
                            MatingPool    = [bestindex,randi(Problem.N,1,3)];
                            Offspring_i   = DE_rb(Problem,Population(MatingPool(1)),Population(MatingPool(2)),Population(MatingPool(3)),Population(MatingPool(4)));
                            Offspring     = cat(2,Offspring,Offspring_i);
                        end
                    end
                    % environment selection based on CDP
                    CVO = sum(max(0,Offspring.cons),2);
                    fo_transformed = sum(Offspring.objs,2)/Problem.M; % weighete objective of offspring

                    betterindex = find(CVO<CVP);
                    Population(betterindex) = Offspring(betterindex);

                    betterindex = find(CVO==CVP & fo_transformed<fp_transformed);
                    Population(betterindex) = Offspring(betterindex);
                    % update Delta
                    CVP   = sum(max(0,Population.cons),2);
                    Pf    = sum(CVP==0)/size(Population,2);
                    feasible_solutions = Population(CVP==0);
                    f_max = max(feasible_solutions.objs);
                    f_min = min(feasible_solutions.objs);
                    f_normalized = (feasible_solutions.objs-repmat(f_min,size(feasible_solutions,2),1))./(repmat(f_max,size(feasible_solutions,2),1)-repmat(f_min,size(feasible_solutions,2),1));
                    f = sum(f_normalized,2);
                    f_sorted = sort(f,'ascend');
                    if size(f_sorted,1) >= 3
                        Delta = f_sorted(floor(size(f_sorted,1)/3))-f_sorted(1);
                    end
                else
                    %% Phase II
                    [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
                    Offspring  = OperatorDE(Problem,Population(TournamentSelection(2,Problem.N,FrontNo,-CrowdDis)),Population(randi(Problem.N,1,Problem.N)),Population(randi(Problem.N,1,Problem.N)));
                    Population = EnvironmentalSelection([Population,Offspring],Problem.N);
                end
            end
        end
    end
end
```
