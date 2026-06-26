# NBLEA

**Tags**: <2014> <multi> <real> <constrained/none> <bilevel>

## Description
Nested bilevel evolutionary algorithm

## Reference
A. Sinha, P. Malo, and K. Deb. Test problem construction for single-objective bilevel optimization. Evolutionary Computation, 2014, 22(3): 439-477.

## Source Code

### `CalFitness.m`
```matlab
function Fitness = CalFitness(C,Population)
% Calculate the fitness of each solution in terms of a single level

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = Population.objs;
    PopCon = Population.cons;
    if any(isnan(PopObj(:,1)))  % Lower level
        PopObj = PopObj(:,2);
        PopCon = PopCon(:,C+1:end);
    else                        % Upper level
        PopObj = PopObj(:,1);
        PopCon = PopCon(:,1:C);
    end
    if isempty(PopCon)
        PopCon = zeros(size(PopObj,1),1);
    else
        PopCon = sum(max(0,PopCon),2);
    end
    Feasible = PopCon <= 0;
    Fitness  = Feasible.*PopObj + ~Feasible.*(PopCon+1e10);
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Problem,Population,Offspring)
% The environmental selection of NBLEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    selected = randperm(length(Population),2);
    Pool     = [Population(selected),Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool));
    Population(selected) = Pool(rank(1:length(selected)));
end
```

### `NBLEA.m`
```matlab
classdef NBLEA < ALGORITHM
% <2014> <multi> <real> <constrained/none> <bilevel>
% Nested bilevel evolutionary algorithm

%------------------------------- Reference --------------------------------
% A. Sinha, P. Malo, and K. Deb. Test problem construction for
% single-objective bilevel optimization. Evolutionary Computation, 2014,
% 22(3): 439-477.
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
            ulPopDec = unifrnd(repmat(Problem.lower(1:Problem.DU),Problem.N,1),repmat(Problem.upper(1:Problem.DU),Problem.N,1));
            for i = 1 : size(ulPopDec,1)
               llPopDec(i,:) = llSearch(Problem,ulPopDec(i,:),[]);
            end
            Population = Problem.Evaluation([ulPopDec,llPopDec]); 
            
            %% UL Optimization
            while Algorithm.NotTerminated(Population)
                % Upper level optimization
                MatingPool = TournamentSelection(2,3,CalFitness(Problem.C,Population));
                ParentDec  = Population(MatingPool).decs;
                ulOffDec   = OperatorPCX(ParentDec(:,1:Problem.DU),Problem.lower(1:Problem.DU),Problem.upper(1:Problem.DU));
                % Lower level optimization
                [~,closest] = min(pdist2(ulOffDec,ParentDec(:,1:Problem.DU)),[],2);
                for i = 1 : size(ulOffDec,1)
                    llOffDec(i,:) = llSearch(Problem,ulOffDec(i,:),ParentDec(closest(i),Problem.DU+1:end));  
                end
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]); 
                % Environment selection for Upper Population
                Population = EnvironmentalSelection(Problem,Population,Offspring);
            end
        end
	end
end
```

### `OperatorPCX.m`
```matlab
function OffDec = OperatorPCX(Parent,Lower,Upper)
% Offspring generation based on PCX and polynomial mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [proC,proM,disM] = deal(0.9,1,20);
    [N,D] = size(Parent);

    %% PCX
    W      = mean(Parent,1);
    p1     = [2:N,1];
    p2     = [3:N,1,2];
    OffDec = Parent + 0.1*(Parent-W) + D./mean(abs(Parent-W),2).*(Parent(p2,:)-Parent(p1,:))/2;
    Site   = repmat(rand(N,1)>proC,1,D);
    OffDec(Site) = Parent(Site);
    
    %% Polynomial mutation
    Lower = repmat(Lower,N,1);
    Upper = repmat(Upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    OffDec       = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end
```

### `llSearch.m`
```matlab
function eliteIndiv = llSearch(Problem,ulPopDec,llPopDec)
% Obtain the upper member corresponds to the best lower member

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Lower level population initialization
    llPopDec     = [llPopDec;unifrnd(repmat(Problem.lower(Problem.DU+1:end),Problem.N-size(llPopDec,1),1),repmat(Problem.upper(Problem.DU+1:end),Problem.N-size(llPopDec,1),1))];
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec,Problem.N,1),llPopDec]);
    FElower      = 0;
    
    %% Optimization
    while FElower < Problem.maxFElower
        % Select parents and generate offspring
        MatingPool  = TournamentSelection(2,3,CalFitness(Problem.C,llPopulation));
        ParentDec   = llPopulation(MatingPool).decs;
        llOffDec    = OperatorPCX(ParentDec(:,Problem.DU+1:end),Problem.lower(Problem.DU+1:end),Problem.upper(Problem.DU+1:end));
        llOffspring = Problem.EvaluationLower([repmat(ulPopDec,size(llOffDec,1),1),llOffDec]);
        FElower     = FElower + length(llOffspring);
        % Select r members with better adaptability
        llPopulation = EnvironmentalSelection(Problem,llPopulation,llOffspring);
    end
    [~,best]   = min(CalFitness(Problem.C,llPopulation));
    eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
end
```
