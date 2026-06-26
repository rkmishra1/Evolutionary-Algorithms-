# PeEA

**Tags**: <2021> <multi/many> <real/integer/label/binary/permutation>

## Description
Pareto front shape estimation based evolutionary algorithm

## Reference
L. Li, G. G. Yen, A. Sahoo, L. Chang, and T. Gu. On the estimation of pareto front and dimensional similarity in many-objective evolutionary algorithm. Information Sciences, 2021, 563: 375-400.

## Source Code

### `CalFitness.m`
```matlab
function [Fitness, extreme] = CalFitness(PopObj)
% Calculate the fitness of each solution
% F(x) = f(x) + d(x) Algorithm 3

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Li Li

    [N,M]  = size(PopObj); %N  No. of solutions, M No. of Objectives
   
    %% q
    Zmin = min(PopObj,[],1); % Identify the ideal point
    % Identify the extreme points
    W = zeros(M) + 1e-6;
    W(logical(eye(M))) = 1;
    ASF = zeros(N,M);
    for i = 1 : M
        ASF(:,i) = max((PopObj-repmat(Zmin,N,1))./repmat(W(i,:),N,1),[],2);
    end
    [~,extreme] = min(ASF,[],1); % 'extreme' is the extreme solutions
    % Calculate the intercepts
    Hyperplane = PopObj(extreme,:)\ones(M,1); % linear equation X=A\B???A*X=B???hyperplane is the solution of linear equation PopObj(extreme,:)x Hyperplane = ones(M,1)
    a = (1./Hyperplane)';
    if any(isnan(a))
        a = max(PopObj,[],1);
    end
    % Normalization
    PopObj = (PopObj-repmat(Zmin,N,1))./repmat(a-Zmin,N,1);
    
    %---------------w_nad
    w_nad = a; % set a as the nadir point

    Zmin = min(PopObj,[],1); % Identify the ideal point
    for i = 1 : M
        ASF_q(:,i) = max((PopObj-repmat(Zmin,N,1))./repmat(w_nad,N,1),[],2);
    end
    [~,key_point] = min(ASF_q,[],1);
    q = sqrt(sum(PopObj(key_point).^2,2)) * sqrt(M); 
    if q <= 1
        fx = sum(PopObj-repmat(Zmin,N,1),2);
    else
        fx = max(PopObj-repmat(Zmin,N,1),[],2);
    end
  
    %% Calculate the DMD between each two solutions  
    Distance = inf(N);
    for i = 1 : N
        for j = [1:i-1,i+1:N]
            denominator   = 1 + min(PopObj(i,:),PopObj(j,:)); 
            numerator     = max(PopObj(i,:),PopObj(j,:)) - min(PopObj(i,:),PopObj(j,:));  
            dis           = numerator./ denominator;
            Distance(i,j) = sum(dis);
        end
    end
      
    %% Calculate D(i)
    Distance = sort(Distance,2);
    k = floor(sqrt(N)); % k-NN
    d = 1./(Distance(:,k)+2); % dimensionality margin distance
    
    %% Calculate the fitnesses
    Fitness = fx + d;
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,T)
% The environmental selection of PeEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Li Li

    [Fitness,extreme] = CalFitness(Population.objs);
    extreme = unique(extreme); % eliminate the identical solutions
    [~,mm]  = size(extreme);  
   
	%% Calculate the angle between each two solutions
    Angle = acos(1-pdist2(Population.objs,Population.objs,'cosine'));
    Angle(logical(eye(length(Population)))) = inf;

    %% Angle-based tournament selection
    Remain = 1 : length(Population);
    Remain(extreme) = []; % eliminate extremes first
    while length(Remain) > (T-mm) % mins mm first
        % Identify the two solutions A and B with the minimum angle
        [sortA,rank1] = sort(Angle(Remain,Remain),2); % sortA is the sorted angle, rank1 NXN index of matric angle
        [~,rank2]     = sortrows(sortA);              % rank2 Nx1 index of sorted angle
        A = rank2(1);                                 % A, with minimum angle
        B = rank1(A,1);                               % B corrponding to A with minimum angle
        % Eliminate one of A and B
        if  Fitness(Remain(A)) > Fitness(Remain(B))
            Remain(A) = [];
        else
            Remain(B) = [];
        end
    end
    % Population for next generation
    Population = Population([Remain,extreme]); % add extremes into pop
end
```

### `MatingSelection.m`
```matlab
function MatingPool = MatingSelection(PopObj)
% The mating selection of PeEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Li Li

    N  = size(PopObj,1);
    [Fitness,extreme] = CalFitness(PopObj);
    Parents1   = randi(N,1,N);
    Parents2   = randi(N,1,N);
    Dominate   = any(PopObj(Parents1,:)<PopObj(Parents2,:),2) - any(PopObj(Parents1,:)>PopObj(Parents2,:),2);
    MatingPool = [Parents1(Dominate==1),...
                  Parents2(Dominate==-1),...
                  Parents1(Dominate==0 & Fitness(Parents1)<=Fitness(Parents2)),...
                  Parents2(Dominate==0 & Fitness(Parents1)>Fitness(Parents2))];
end
```

### `PeEA.m`
```matlab
classdef PeEA < ALGORITHM
% <2021> <multi/many> <real/integer/label/binary/permutation>
% Pareto front shape estimation based evolutionary algorithm

%------------------------------- Reference --------------------------------
% L. Li, G. G. Yen, A. Sahoo, L. Chang, and T. Gu. On the estimation of
% pareto front and dimensional similarity in many-objective evolutionary
% algorithm. Information Sciences, 2021, 563: 375-400.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Li Li

	methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = MatingSelection(Population.objs);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Population = EnvironmentalSelection([Population,Offspring],Problem.N); 
            end
        end
    end
end
```
