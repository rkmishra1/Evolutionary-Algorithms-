# IM-MOEA

**Tags**: <2015> <multi> <real/integer> <large/none>

## Description
Inverse modeling based multiobjective evolutionary algorithm

## Reference
R. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff. A multiobjective evolutionary algorithm using Gaussian process-based inverse modeling. IEEE Transactions on Evolutionary Computation, 2015, 19(6): 838-856.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function Del = EnvironmentalSelection(Population,nSub)
% The environmental selection of IM-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,nSub);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front by crowding distance
    Last     = find(FrontNo==MaxFNo);
    CrowdDis = CrowdingDistance(Population(Last).objs);
    [~,rank] = sort(CrowdDis,'descend');
    Next(Last(rank(1:nSub-sum(Next)))) = true;
    % The index of deleted solutions
    Del = ~Next;
end
```

### `IMMOEA.m`
```matlab
classdef IMMOEA < ALGORITHM
% <2015> <multi> <real/integer> <large/none>
% Inverse modeling based multiobjective evolutionary algorithm
% K --- 10 --- Number of reference vectors

%------------------------------- Reference --------------------------------
% R. Cheng, Y. Jin, K. Narukawa, and B. Sendhoff. A multiobjective
% evolutionary algorithm using Gaussian process-based inverse modeling.
% IEEE Transactions on Evolutionary Computation, 2015, 19(6): 838-856.
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
            [W,K] = UniformPoint(K,Problem.M);
            W     = fliplr(sortrows(fliplr(W)));
            Problem.N     = ceil(Problem.N/K)*K;
            Population    = Problem.Initialization();
            [~,partition] = max(1-pdist2(Population.objs,W,'cosine'),[],2);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Modeling and reproduction
                for k = unique(partition)'
                    Population = [Population,Operator(Problem,Population(partition==k))];
                end
                % Environmental selection
                [~,partition] = max(1-pdist2(Population.objs,W,'cosine'),[],2);
                for k = unique(partition)'
                    current = find(partition==k);
                    if length(current) > Problem.N/K
                        Del = EnvironmentalSelection(Population(current),Problem.N/K);
                        Population(current(Del)) = [];
                        partition(current(Del))  = [];
                    end
                end
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Population,L)
% The Gaussian process based reproduction

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://www.soft-computing.de/jin-pub_year.html

    %% Parameter setting
    if nargin < 3
        L = 3;
    end
    PopDec = Population.decs;
	PopObj = Population.objs;
    [N,D]  = size(PopDec);
    
    %% Gaussian process based reproduction
    if length(Population) < 2*Problem.M
        OffDec = PopDec;
    else
        OffDec = [];
        fmin   = 1.5*min(PopObj,[],1) - 0.5*max(PopObj,[],1);
        fmax   = 1.5*max(PopObj,[],1) - 0.5*min(PopObj,[],1);
        % Train one groups of GP models for each objective
        for m = 1 : Problem.M
            parents = randperm(N,floor(N/Problem.M));
            offDec  = PopDec(parents,:);
            for d = randperm(D,L)
                % Gaussian Process
                try
                    [ymu,ys2] = gp(struct('mean',[],'cov',[],'lik',log(0.01)),...
                                   @infExact,@meanZero,@covLIN,@likGauss,...
                                   PopObj(parents,m),PopDec(parents,d),...
                                   linspace(fmin(m),fmax(m),size(offDec,1))');
                    offDec(:,d) = ymu + rand*sqrt(ys2).*randn(size(ys2));
                catch
                end
            end
            OffDec = [OffDec;offDec];
        end
    end
    
    %% Convert invalid values to random values
    [N,D]   = size(OffDec);
    Lower   = repmat(Problem.lower,N,1);
    Upper   = repmat(Problem.upper,N,1);
    randDec = unifrnd(Lower,Upper);
    invalid = OffDec<Lower | OffDec>Upper;
    OffDec(invalid) = randDec(invalid);

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Site   = rand(N,D) < proM/D;
    mu     = rand(N,D);
    temp   = Site & mu<=0.5;
    OffDec = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec);
end
```
