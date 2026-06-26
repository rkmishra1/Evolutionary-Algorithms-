# CMOES

**Tags**: <2024> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Constrained multi-objective optimization based on even search

## Reference
F. Ming, W. Gong, and Y. Jin. Even search in a promising region for constrained multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2024, 11(2): 474-486.

## Source Code

### `CMOES.m`
```matlab
classdef CMOES < ALGORITHM
% <2024> <multi> <real/integer/label/binary/permutation> <constrained>
% Constrained multi-objective optimization based on even search
% type --- 1 --- Type of operator (1. GA 2. DE)

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, and Y. Jin. Even search in a promising region for
% constrained multi-objective optimization. IEEE/CAA Journal of Automatica
% Sinica, 2024, 11(2): 474-486.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            type = Algorithm.ParameterSet(1);
            
            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection_NSGA2(Population1,Problem.N);
            Fitness2    = CalFitness(Population2.objs);
            Population3 = Population1;
            stage_changed = 0;
            Population = [Population1,Population2];
            CV = sum(max(0,Population.cons),2);
            max_cv = max(CV);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                gen        = ceil(Problem.FE/Problem.N);
                CV1 = sum(max(0,Population1.cons),2);
                num_fea = sum(CV1==0);
                if num_fea <= 0 || gen <= 0.2 * ceil(Problem.maxFE/Problem.N)
                    MatingPool = TournamentSelection(2,2*Problem.N,FrontNo,-CrowdDis);
                    if type == 1
                        Offspring  = OperatorGA(Problem,Population1(MatingPool));
                    else
                        Offspring  = OperatorDE(Problem,Population1,Population1(MatingPool(1:end/2)),Population1(MatingPool(end/2+1:end)));
                    end
                    MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);
                    if type == 1
                        Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                    else
                        Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                    end
                    [Population1,FrontNo,CrowdDis] = EnvironmentalSelection_NSGA2([Population1,Offspring,Offspring2],Problem.N);
                    [Population2,Fitness2] = EnvironmentalSelection([Population2,Offspring,Offspring2],Problem.N,false);
                    Population3 = Population2;
                    Population = [Population1,Population2];
                    CV = sum(max(0,Population.cons),2);
                    max_cv = max(max_cv,max(CV));
                else
                    tau = gen/ceil(Problem.maxFE/Problem.N);
                    Population = [Population1,Population2,Population3];
                    CV = sum(max(0,Population.cons),2);
                    max_cv = max(CV);
                    if stage_changed == 0
                        [~,Fitness2,~,Fitness3] = EvenSearch(Population,Population1(CV1==0),Problem.N,tau,max_cv);
                        [~,Fitness1] = EnvironmentalSelection(Population1,Problem.N,true);
                        stage_changed = 1;
                    end
                    
                    if ~isempty(Population2)
                        MatingPool2 = TournamentSelection(2,2*length(Population2),Fitness2);
                        if type == 1
                            Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                        else
                            Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                        end
                    else
                        Offspring2 = [];
                    end
                    if ~isempty(Population3)
                        MatingPool3 = TournamentSelection(2,2*length(Population3),Fitness3);
                        if type == 1
                            Offspring3  = OperatorGAhalf(Problem,Population3(MatingPool3));
                        else
                            Offspring3  = OperatorDE(Problem,Population3,Population3(MatingPool3(1:end/2)),Population3(MatingPool3(end/2+1:end)));
                        end
                    else
                        Offspring3 = [];
                    end
                    MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                    if type == 1
                        Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                    else
                        Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                    end
                    
                    Offspring = [Offspring1,Offspring2,Offspring3];
                    
                    [Population1,Fitness1] = EnvironmentalSelection([Population,Offspring],Problem.N,true);
                    [Population2,Fitness2,Population3,Fitness3] = EvenSearch([Population,Offspring],Population1(CV1==0),Problem.N,tau,max_cv);
                end
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon)
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
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
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

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection(Population,N,isOrigin)
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
    if isOrigin
        Fitness = CalFitness(Population.objs,Population.cons);
    else
        Fitness = CalFitness(Population.objs);
    end

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
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population = Population(rank);
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

### `EnvironmentalSelection_NSGA2.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection_NSGA2(Population,N)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
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

### `EvenSearch.m`
```matlab
function [Population2,Fitness,Population3,CV] = EvenSearch(Population,non_dom,N,tau,max_cv)
% Even search for helper population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    %% Determine if in promising region (1 is in)
    lables = GetLable(Population.objs,non_dom.objs);
    Next1 = lables == 1;
    First_Population = Population(Next1);
    
    % first choose non-dominated solutions near UPF
    Fitness = CalFitness(First_Population.objs);
    Next2 = Fitness < 1;
    if sum(Next2) > N
        Del  = Truncation(First_Population(Next2).objs,sum(Next2)-N);
        Temp = find(Next2);
        Next2(Temp(Del)) = false;
    end
    % store non-dominated solutions in former N
    Population2 = First_Population(Next2);
    Fitness = Fitness(Next2);
    
    % then choose CV better solutions near or in feasible regions
    % first limit the position in the objective space
    Second_Population = First_Population;
    CV = sum(max(0,Second_Population.cons),2);
    epsilon = (max_cv)*(1-tau)^2;
    Next3 = CV <= epsilon;
    if sum(Next3) > N
        Del  = Truncation(Second_Population(Next3).objs,sum(Next3)-N);
        Temp = find(Next3);
        Next3(Temp(Del)) = false;
    end
    % store more feasible solutions in former N
    Population3 = Second_Population(Next3);
    CV = CV(Next3);
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

### `GetLable.m`
```matlab
function lable = GetLable(Solution,non_dom)
% Get the label of each solution, 1 represents is

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    N = size(Solution,1);
    M = size(non_dom,1);
    Lables = zeros(N,M);

    %% Detect the dominance relation between each solution in Data and fns
    for i = 1 : N
        for j = 1 : M
            k = any(Solution(i,:)<non_dom(j,:)) - any(Solution(i,:)>non_dom(j,:));
            if k == 1 || k == 0
                Lables(i,j) = 1;
            end
        end
    end
    lable = zeros(1,N);
    lable(sum(Lables,2)==M) = 1;
end
```
