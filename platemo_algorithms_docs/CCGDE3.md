# CCGDE3

**Tags**: <2013> <multi> <real/integer> <large/none>

## Description
Cooperative coevolution generalized differential evolution 3

## Reference
L. M. Antonio and C. A. Coello Coello. Use of cooperative coevolution for solving large scale multiobjective optimization problems. Proceedings of the IEEE Congress on Evolutionary Computation, 2013, 2758-2765.

## Source Code

### `CCDE.m`
```matlab
function Offspring = CCDE(Parent1,Parent2,Parent3,lower,upper,Parameter)
% Differential evolution operator used in CCGDE3

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    if nargin > 5
        [CR,F,proM,disM] = deal(Parameter{:});
    else
        [CR,F,proM,disM] = deal(0.5,0.5,1,20);
    end
    [N,D] = size(Parent1);

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    Lower = repmat(lower,N,1);
    Upper = repmat(upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end
```

### `CCGDE3.m`
```matlab
classdef CCGDE3 < ALGORITHM
% <2013> <multi> <real/integer> <large/none>
% Cooperative coevolution generalized differential evolution 3

%------------------------------- Reference --------------------------------
% L. M. Antonio and C. A. Coello Coello. Use of cooperative coevolution for
% solving large scale multiobjective optimization problems. Proceedings of
% the IEEE Congress on Evolutionary Computation, 2013, 2758-2765.
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
            %% Random grouping
            NumEsp = 2; % the number of subPopulations
            varsPerGroup = floor(Problem.D/NumEsp);
            Index = [];
            numSPop = ceil(0.8*Problem.N);  % number of individuals in each subPopulation
            for i = 1:NumEsp-1
                Index = [Index,ones(1,varsPerGroup).*i];
            end
            Index = [Index, ones(1,Problem.D-size(Index,2)).*NumEsp];
            Index = Index(randperm(length(Index))); % randomly grouping
            
            %% Generate random population
            Gmax        = 1;	% the times of each subpopulation
            Dec         = unifrnd(repmat(Problem.lower,numSPop,1),repmat(Problem.upper,numSPop,1));
            subDec1     = Dec(:,Index==1);
            subDec2     = Dec(:,Index==2);
            Population1 = GetInd(Problem,subDec1,subDec2,Index,numSPop,1);
            Population2 = GetInd(Problem,subDec2,subDec1,Index,numSPop,2);
            Population  = EnvironmentalSelection([Population1,Population2],Problem.N);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                FrontNo1  = NDSort(Population1.objs,1);
                FrontNo2  = NDSort(Population2.objs,1);
                NDsubDec1 = subDec1(FrontNo1==1,:);
                NDsubDec2 = subDec2(FrontNo2==1,:);
                for j = 1 : NumEsp
                    for k = 1 : Gmax
                        if j == 1
                        	OffDec1     = CCDE(subDec1,subDec1(randi(numSPop,1,numSPop),:),subDec1(randi(numSPop,1,numSPop),:),Problem.lower(Index==j),Problem.upper(Index==j));
                        	Offspring   = GetInd(Problem,OffDec1,NDsubDec2,Index,numSPop,j);
                        	Population1 = GDE3_EnvironmentalSelection(Population1,Offspring,numSPop);
                        	Dec1        = Population1.decs;
                        	subDec1     = Dec1(:,Index==1);
                        else
                        	OffDec2     = CCDE(subDec2,subDec2(randi(numSPop,1,numSPop),:),subDec2(randi(numSPop,1,numSPop),:),Problem.lower(Index==j),Problem.upper(Index==j));
                        	Offspring   = GetInd(Problem,OffDec2,NDsubDec1,Index,numSPop,j);
                        	Population2 = GDE3_EnvironmentalSelection(Population2,Offspring,numSPop);
                        	Dec2        = Population1.decs;
                        	subDec2     = Dec2(:,Index==2);
                        end
                        Population = EnvironmentalSelection([Population1,Population2],Problem.N);
                    end
                end
            end
        end
    end
end

function Population = GetInd(Problem,subDec1,subDec2,Index,numSPop,j)
    Dec = zeros(numSPop,Problem.D);
    if j == 1
       Dec(:,Index==1) = subDec1;
       Dec(:,Index==2) = subDec2(randi(end,numSPop,1),:);
    else
       Dec(:,Index==2) = subDec1;
       Dec(:,Index==1) = subDec2(randi(end,numSPop,1),:);           
    end
    Population = Problem.Evaluation(Dec);
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N)
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
end
```

### `GDE3_EnvironmentalSelection.m`
```matlab
function Population = GDE3_EnvironmentalSelection(Population,Offspring,N)
% The environmental selection of GDE3

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    %% Select by constraint-domination
    PopObj    = Population.objs;
    PopCon    = Population.cons;
    feasibleP = all(PopCon<=0,2);
    OffObj    = Offspring.objs;
    OffCon    = Offspring.cons;
    feasibleO = all(OffCon<=0,2);
    % The offsprings which can replace its parent
    updated = ~feasibleP&feasibleO  | ...
              ~feasibleP&~feasibleO & all(PopCon>=OffCon,2) | ...
              feasibleP&feasibleO   & all(PopObj>=OffObj,2);
    % The offsprings which can add to the population
    selected = feasibleP&feasibleO & any(PopObj<OffObj,2) & any(PopObj>OffObj,2);
    % Update the population
    Population(updated) = Offspring(updated);
    Population          = [Population,Offspring(selected)];
    
    %% Select by non-dominated sorting and crowding distance
    PopObj   = Population.objs;
    PopCon   = Population.cons;
    feasible = all(PopCon<=0,2);
    % Non-dominated sorting based on constraint-domination
    FrontNo = inf(1,length(Population));
    [FrontNo(feasible),MaxFNo] = NDSort(PopObj(feasible,:),inf);
    FrontNo(~feasible) = NDSort(PopCon(~feasible,:),inf) + MaxFNo;
    % Determine the last front
    MaxFNo    = find(cumsum(hist(FrontNo,1:max(FrontNo)))>=N,1);
    lastFront = find(FrontNo==MaxFNo);
    % Eliminate solutions in the last front one by one
    while length(lastFront) > N - sum(FrontNo<MaxFNo)
        [~,worst] = min(CrowdingDistance(PopObj(lastFront,:)));
        lastFront(worst) = [];
    end
    Population = Population([find(FrontNo<MaxFNo),lastFront]);
end
```
