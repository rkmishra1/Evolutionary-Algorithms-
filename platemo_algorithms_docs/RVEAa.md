# RVEAa

**Tags**: <2016> <multi/many> <real/integer/label/binary/permutation> <constrained/none>

## Description
RVEA embedded with the reference vector regeneration strategy

## Reference
R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff. A reference vector guided evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 773-791.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,V,theta)
% The environmental selection of RVEA*

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select only the non-dominated solutions
    Population = Population(NDSort(Population.objs,1)==1);
    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    NV     = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the degree of violation of each solution
    CV = sum(max(0,Population.cons),2);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current1 = find(associate==i & CV==0);
        current2 = find(associate==i & CV~=0);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD = (1+M*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
            % Select the one with the minimum APD value
            [~,best] = min(APD);
            Next(i)  = current1(best);
        elseif ~isempty(current2)
            % Select the one with the minimum CV value
            [~,best] = min(CV(current2));
            Next(i)  = current2(best);
        end
    end
    % Population for next generation
    Population = Population(Next(Next~=0));
end
```

### `RVEAa.m`
```matlab
classdef RVEAa < ALGORITHM
% <2016> <multi/many> <real/integer/label/binary/permutation> <constrained/none>
% RVEA embedded with the reference vector regeneration strategy
% alpha ---   2 --- The parameter controlling the rate of change of penalty
% fr    --- 0.1 --- The frequency of employing reference vector adaptation

%------------------------------- Reference --------------------------------
% R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff. A reference vector guided
% evolutionary algorithm for many-objective optimization. IEEE Transactions
% on Evolutionary Computation, 2016, 20(5): 773-791.
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
            [alpha,fr] = Algorithm.ParameterSet(2,0.1);

            %% Generate the reference points and random population
            [V0,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population     = Problem.Initialization();
            V              = [V0;rand(Problem.N,Problem.M)];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = randi(length(Population),1,Problem.N);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Population = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^alpha);
                if ~mod(ceil(Problem.FE/Problem.N),ceil(fr*Problem.maxFE/Problem.N))
                    V(1:Problem.N,:) = ReferenceVectorAdaptation(Population.objs,V0);
                end
                V(Problem.N+1:end,:) = ReferenceVectorRegeneration(Population.objs,V(Problem.N+1:end,:));
                if Problem.FE >= Problem.maxFE
                    Population = Truncation(Population,Problem.N);
                end
            end
        end
    end
end
```

### `ReferenceVectorAdaptation.m`
```matlab
function V = ReferenceVectorAdaptation(PopObj,V)
% Reference vector adaption strategy

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    V = V.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V,1),1);
end
```

### `ReferenceVectorRegeneration.m`
```matlab
function V = ReferenceVectorRegeneration(PopObj,V)
% Reference vector regeneration

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj        = PopObj - repmat(min(PopObj,[],1),size(PopObj,1),1);
    [~,associate] = max(1-pdist2(PopObj,V,'cosine'),[],2);
    inValid       = setdiff(1:size(V,1),associate);
    V(inValid,:)  = rand(length(inValid),size(V,2)).*repmat(max(PopObj,[],1),length(inValid),1);
end
```

### `Truncation.m`
```matlab
function Population = Truncation(Population,N)
% Limit the size of final popualtion in RVEA*

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Choose = true(1,length(Population));
    Cosine = 1 - pdist2(Population.objs,Population.objs,'cosine');
    Cosine(logical(eye(length(Cosine)))) = 0;
    while sum(Choose) > N
        Remain   = find(Choose);
        Temp     = sort(-Cosine(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Choose(Remain(Rank(1))) = false;
    end
    Population = Population(Choose);
end
```
