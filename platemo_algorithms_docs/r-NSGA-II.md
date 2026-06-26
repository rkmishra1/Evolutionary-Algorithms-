# r-NSGA-II

**Tags**: <2010> <multi> <real/integer/label/binary/permutation>

## Description
r-dominance based NSGA-II

## Reference
L. B. Said, S. Bechikh, and K. Ghedira. The r-dominance: A new dominance relation for interactive evolutionary multicriteria decision making. IEEE Transactions on Evolutionary Computation, 2010, 14(5): 801-818.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N,Points,W,delta)
% The environmental selection of r-NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-r-dominated sorting
    [FrontNo,MaxFNo] = NrDSort(Population.objs,N,Points,W,delta);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front by their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `NrDSort.m`
```matlab
function [FrontNo,MaxFNo] = NrDSort(PopObj,nSort,Points,W,delta)
% Do non-r-dominated sorting

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    FrontNo = inf(1,size(PopObj,1));
    for i = 1 : size(Points,1)
        FrontNo = min(FrontNo,nrdsort(PopObj,Points(i,:),W(i,:),delta));
    end
    MaxFNo = find(cumsum(hist(FrontNo,1:max(FrontNo)))>=min(nSort,length(FrontNo)),1);
    FrontNo(FrontNo>MaxFNo) = inf;
end

function FrontNo = nrdsort(PopObj,g,w,delta)
% Sort the population according to one preferred point

    [PopObj,~,Loc] = unique(PopObj,'rows');
    % Calculate the weighted Euclidean distance of each solution
    Dist       = sqrt(((PopObj-repmat(g,size(PopObj,1),1))./repmat(max(PopObj,[],1)-min(PopObj,[],1),size(PopObj,1),1)).^2*(w/sum(w))');
    DistExtent = max(Dist) - min(Dist);
    % Sort the population based on their Dist values, so that a solution
    % cannot r-dominate the solutions having smaller Dist values than it
    [Dist,rank] = sort(Dist);
    PopObj      = PopObj(rank,:);
    % Non-r-dominated sorting
    [N,M]   = size(PopObj);
    FrontNo = inf(1,N);
    MaxFNo  = 0;
    while any(FrontNo==inf)
        MaxFNo = MaxFNo + 1;
        for i = 1 : N
            if FrontNo(i) == inf
                Dominated = false;
                for j = 1 : N
                    if FrontNo(j) >= MaxFNo&&j ~= i
                        m = 1;
                        % First check the Pareto dominance relationship
                        while m <= M && PopObj(i,m) >= PopObj(j,m)
                            m = m + 1;
                        end
                        Dominated = m > M;                       
                        if Dominated
                           break;
                        end
                    end
                end
                % If the current solution is non-dominated one, then check the r-dominance relationship
                if ~Dominated
                    for j = i-1 : -1 : 1
                        if FrontNo(j) == MaxFNo
                            Dominated = (Dist(j)-Dist(i))./DistExtent < -delta;
                            if Dominated
                                break;
                            end
                        end
                    end
                end
                if ~Dominated
                    FrontNo(i) = MaxFNo;
                end
            end
        end
    end
    FrontNo(rank) = FrontNo;
    FrontNo = FrontNo(Loc);
end
```

### `rNSGAII.m`
```matlab
classdef rNSGAII < ALGORITHM
% <2010> <multi> <real/integer/label/binary/permutation>
% r-dominance based NSGA-II
% Points ---     --- Set of preferred points
% W      ---     --- Set of weight vectors for preferred points
% delta  --- 0.1 --- Non-r-dominance threshold

%------------------------------- Reference --------------------------------
% L. B. Said, S. Bechikh, and K. Ghedira. The r-dominance: A new dominance
% relation for interactive evolutionary multicriteria decision making. IEEE
% Transactions on Evolutionary Computation, 2010, 14(5): 801-818.
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
            [Points,W,delta] = Algorithm.ParameterSet(zeros(1,Problem.M)+0.5,ones(1,Problem.M),0.1);

            %% Generate random population
            Population = Problem.Initialization();
            FrontNo    = NrDSort(Population.objs,inf,Points,W,1-(1-delta)*Problem.FE/Problem.maxFE);
            CrowdDis   = CrowdingDistance(Population.objs,FrontNo);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N,Points,W,1-(1-delta)*Problem.FE/Problem.maxFE);
            end
        end
    end
end
```
