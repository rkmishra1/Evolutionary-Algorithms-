# CMOSMA

**Tags**: <2022> <multi/many> <real/integer> <constrained>

## Description
Constrained multi-objective evolutionary algorithm with self-organizing map

## Reference
C. He, M. Li, C. Zhang, H. Chen, P. Zhong, Z. Li, and J. Li. A self-organizing map approach for constrained multi-objective optimization problems. Complex & Intelligent Systems, 2022, 8: 5355-5375.

## Source Code

### `Associate.m`
```matlab
function [XU] = Associate(Population,W,N)
% Associate each solution with a neuron

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Chao He

     A  = 1 : N;
     U  = 1 : N;
     XU = zeros(1,N);
     for i = 1 : N
         x = randi(length(A));
         [~,u] = min(pdist2(Population(A(x)).dec,W(U,:)));
         XU(U(u)) = A(x);
         A(x)     = [];
         U(u)     = [];
     end
end
```

### `CMOSMA.m`
```matlab
classdef CMOSMA < ALGORITHM
% <2022> <multi/many> <real/integer> <constrained>
% Constrained multi-objective evolutionary algorithm with self-organizing map

%------------------------------- Reference --------------------------------
% C. He, M. Li, C. Zhang, H. Chen, P. Zhong, Z. Li, and J. Li. A
% self-organizing map approach for constrained multi-objective optimization
% problems. Complex & Intelligent Systems, 2022, 8: 5355-5375.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Chao He

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [D,tau0,H] = Algorithm.ParameterSet(repmat(ceil(Problem.N.^(1/(Problem.M-1))),1,Problem.M-1),0.7,5);
            D1         = D;
            Problem.N  = prod(D);
            sigma0     = sqrt(sum(D.^2)/(Problem.M-1))/2;

            %% Generate random population
            FP = Problem.Initialization();
            AP = Problem.Initialization();

            %% Initialize the SOM
            % Training set
            S = FP.decs;
            % Weight vector of each neuron
            W = S;
            [LDis,B] = Initialize_SOM(S,D,H);
            % Training set
            S2 = AP.decs;
            % Weight vector of each neuron
            W2 = S2;
            [LDis2,B2] = Initialize_SOM(S2,D1,H);
            
            %% Optimization
            while Algorithm.NotTerminated(FP)
                % Update SOM1
                W = UpdateSOM(S,W,Problem.FE,Problem.maxFE,LDis,sigma0,tau0); 
                % Update SOM2
                W2 = UpdateSOM(S2,W2,Problem.FE,Problem.maxFE,LDis2,sigma0,tau0);

                % Associate each solution with a neuron
                XU  = Associate(FP,W,Problem.N);
                XU2 = Associate(AP,W2,Problem.N); 

                %Construct  matingPool  
                [MatingPool1] = MatingPool(XU,Problem.N,B);
                [MatingPool2] = MatingPool(XU2,Problem.N,B2);

                % Evolution
                A1 = FP.decs;
                Offspring1 = OperatorGA(Problem,[FP(XU),FP(MatingPool1)]);%GA
                A2         = AP.decs;
                Offspring2 = OperatorGA(Problem,[AP(XU2),AP(MatingPool2)]);%GA
                %EnvironmentalSelection
                FP = EnvironmentalSelection([FP,Offspring1,Offspring2],Problem.N,true);
                AP = EnvironmentalSelection([AP,Offspring1,Offspring2],Problem.N,false);
                % Update the training set
                S  = setdiff(FP.decs,A1,'rows');
                S2 = setdiff(AP.decs,A2,'rows');
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

% This function is written by Chao He

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
function [Population] = EnvironmentalSelection(Population,N,isOrigin)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Chao He

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

### `Initialize_SOM.m`
```matlab
function [LDis,B] = Initialize_SOM(S,D,H)
% Initialize the SOM

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Chao He

   % Position of each neuron
    D = arrayfun(@(S)1:S,D,'UniformOutput',false);
    eval(sprintf('[%s]=ndgrid(D{:});',sprintf('c%d,',1:length(D))))
    eval(sprintf('Z=[%s];',sprintf('c%d(:),',1:length(D))))
    % Distance between each two neurons in latent space
    LDis = pdist2(Z,Z);
    % H nearest neurons of each neuron in latent space
    [~,B] = sort(LDis,2);
    B     = B(:,2:min(H+1,end));
end
```

### `MatingPool.m`
```matlab
function [MatingPool] = MatingPool(XU,N,B)
% Construct the MatingPool

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Chao He

    MatingPool = 1 : N;
    for u = 1 :N  
        if rand < 0.9
            Q = XU(B(u,:));
        else
            Q = 1 : N;
        end
        MatingPool(u)= Q(randperm(end,1));
    end
end
```

### `UpdateSOM.m`
```matlab
function W = UpdateSOM(S,W,FE,maxFE,LDis,sigma0,tau0)
% Update SOM

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Chao He

    for s = 1 : size(S,1)
        sigma  = sigma0*(1-(FE+s)/maxFE);
        tau    = tau0*(1-(FE+s)/maxFE);
        [~,u1] = min(pdist2(S(s,:),W));
        U      = LDis(u1,:) < sigma;
        W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)).*(repmat(S(s,:),sum(U),1)-W(U,:));
    end
end
```
