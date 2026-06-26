# RPEA

**Tags**: <2017> <many> <real/integer/label/binary/permutation>

## Description
research purposes. All publications which use this platform or any code

## Reference
alpha --- 0.4 --- Ratio of individuals being used to generate reference points delta --- 0.1 --- Parameter determining the difference between the reference point and the individuals Y. Liu, D. Gong, X. Sun, and Y. Zhang. Many-objective evolutionary optimization based on reference points. Applied Soft Computing, 2017, 50: 344-355.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,R,N)
% The environmental selection of RPEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the Tchebychev distances
    Distance = TchebychevDistance(Population.objs,R);
    
    %% Environmental selection
    RemainP = 1 : length(Population);
    RemainR = 1 : size(R,1);
    while length(RemainP) > length(Population)-N
        if isempty(RemainR)
            RemainR = 1 : size(R,1);
        end
        [temp,imin]   = min(Distance(RemainP,RemainR),[],1);
        [~,jmin]      = min(temp);
        imin          = imin(jmin);
        RemainP(imin) = [];
        RemainR(jmin) = [];
    end
    Population = Population(setdiff(1:length(Population),RemainP));
end
```

### `GenerateRefPoints.m`
```matlab
function R = GenerateRefPoints(Q,diff,alpha,N)
% Generate the reference points according to the combined population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Q    = Q(NDSort(Q.objs,1)==1);
    R    = [];
    subN = min(length(Q),ceil(alpha*N));
    CrowdDis = CrowdingDistanceInEachObj(Q.objs);
    [~,rank] = sort(CrowdDis,1,'descend');
    for m = 1 : length(Q(1).obj)
        Rm      = Q(rank(1:subN,m)).objs;
        Rm(:,m) = Rm(:,m) - diff(m);
        R       = [R;Rm];
    end
    R = R(NDSort(R,1)==1,:);
    if size(R,1) > N
        CrowdDis = CrowdingDistanceInEachObj(R);
        [~,rank] = sort(sum(CrowdDis,2),'descend');
        R        = R(rank(1:N),:);
    end
end

function CrowdDis = CrowdingDistanceInEachObj(PopObj)
% Calculate the crowding distance of each solution in each objective

    [N,M]    = size(PopObj);
    CrowdDis = zeros(N,M);
    Fmax     = max(PopObj,[],1);
    Fmin     = min(PopObj,[],1);
    for i = 1 : M
        [~,rank] = sortrows(PopObj(:,i));
        CrowdDis(rank(1),i)   = inf;
        CrowdDis(rank(end),i) = inf;
        for j = 2 : N-1
            CrowdDis(rank(j),i) = (PopObj(rank(j+1),i)-PopObj(rank(j-1),i))/(Fmax(i)-Fmin(i));
        end
    end
end
```

### `RPEA.m`
```matlab
classdef RPEA < ALGORITHM
% <2017> <many> <real/integer/label/binary/permutation>
% Reference points-based evolutionary algorithm
% alpha --- 0.4 --- Ratio of individuals being used to generate reference points
% delta --- 0.1 --- Parameter determining the difference between the reference point and the individuals

%------------------------------- Reference --------------------------------
% Y. Liu, D. Gong, X. Sun, and Y. Zhang. Many-objective evolutionary
% optimization based on reference points. Applied Soft Computing, 2017,
% 50: 344-355.
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
            [alpha,delta] = Algorithm.ParameterSet(0.4,0.1);

            %% Generate random population
            Population = Problem.Initialization();
            R          = GenerateRefPoints(Population,delta*(max(Population.objs,[],1)-min(Population.objs,[],1)),alpha,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,min(TchebychevDistance(Population.objs,R),[],2));
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                R          = GenerateRefPoints([Population,Offspring],delta*(max(Population.objs,[],1)-min(Population.objs,[],1)),alpha,Problem.N);
                Population = EnvironmentalSelection([Population,Offspring],R,Problem.N);
            end
        end
    end
end
```

### `TchebychevDistance.m`
```matlab
function Distance = TchebychevDistance(PopObj,R)
% Calculate the Tchebychev distance between each individual and reference
% point

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    fmax     = max(PopObj,[],1);
    fmin     = min(PopObj,[],1);
    Distance = zeros(size(PopObj,1),size(R,1));
    for i = 1 : size(PopObj,1)
        Distance(i,:) = max((repmat(PopObj(i,:),size(R,1),1)-R)./repmat(fmax-fmin,size(R,1),1)./size(PopObj,2),[],2)';
    end
end
```
