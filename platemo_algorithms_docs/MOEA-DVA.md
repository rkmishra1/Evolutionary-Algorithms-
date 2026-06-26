# MOEA-DVA

**Tags**: <2016> <multi> <real/integer> <large>

## Description
Multi-objective evolutionary algorithm based on decision variable

## Reference
X. Ma, F. Liu, Y. Qi, X. Wang, L. Li, L. Jiao, M. Yin, and M. Gong. A multiobjective evolutionary algorithm based on decision variable analyses for multiobjective optimization problems with large-scale variables. IEEE Transactions Evolutionary Computation, 2016, 20(2): 275-298.

## Source Code

### `ControlVariableAnalysis.m`
```matlab
function [DiverIndexes,ConverIndexes] = ControlVariableAnalysis(Problem,NCA)
% Control variable analysis

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    DiverIndexes  = false(1,Problem.D);
    ConverIndexes = false(1,Problem.D);
    for i = 1 : Problem.D
        x      = unifrnd(Problem.lower,Problem.upper);
        S      = repmat(x,NCA,1);
        S(:,i) = ((1:NCA)'-1+rand(NCA,1))/NCA*(Problem.upper(i)-Problem.lower(i)) + Problem.lower(i);
        S      = Problem.Evaluation(S);
        [~,MaxFNo] = NDSort(S.objs,inf);
        if MaxFNo == length(S)
            ConverIndexes(i) = true;
        else
            DiverIndexes(i) = true;
        end
    end
end
```

### `DividingDistanceVariables.m`
```matlab
function [Subcomponents,Population] = DividingDistanceVariables(Problem,NIA,DiverIndexes,ConverIndexes)
% Dividing distance variables

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    %% Generate the initial population
    PopDec = zeros(Problem.N,Problem.D);
    % Generate the diverse variables by uniformly sampling method
    if sum(DiverIndexes) == 1
        PopDec(:,DiverIndexes) = (0:Problem.N-1)/(Problem.N-1);
    elseif sum(DiverIndexes) > 4
        PopDec(:,DiverIndexes) = rand(Problem.N,sum(DiverIndexes));
    else
        PopDec(:,DiverIndexes) = UDall(Problem.N,sum(DiverIndexes));
    end
    % Randomly generate the distance variables
    PopDec(:,ConverIndexes) = rand(Problem.N,sum(ConverIndexes));
    % Generate the initial population
    PopDec     = PopDec.*repmat(Problem.upper-Problem.lower,Problem.N,1) + repmat(Problem.lower,Problem.N,1);
    Population = Problem.Evaluation(PopDec);
    
    %% Interdependence analysis
    interaction = false(Problem.D);
    interaction(logical(eye(Problem.D))) = true;
    for i = 1 : Problem.D-1
        for j = i+1 : Problem.D
            drawnow('limitrate');
            for time2try = 1 : NIA
                % Detect whether the i-th and j-th decision variables are
                % interacting
                x    = randi(Problem.N);
                a2   = rand*(Problem.upper(i)-Problem.lower(i)) + Problem.lower(i);
                b2   = rand*(Problem.upper(j)-Problem.lower(j)) + Problem.lower(j);
                Decs = repmat(Population(x).dec,3,1);
                Decs(1,i) = a2;
                Decs(2,j) = b2;
                Decs(3,[i,j]) = [a2,b2];
                F = Problem.Evaluation(Decs);
                delta1 = F(1).obj - Population(x).obj;
                delta2 = F(3).obj - F(2).obj;
                interaction(i,j) = interaction(i,j) | any(delta1.*delta2<0);
                interaction(j,i) = interaction(i,j);
                % Update the solution
                if ConverIndexes(j) && all(F(2).obj <=Population(x).obj)
                    Population(x) = F(2);
                end
                if ConverIndexes(i) && all(F(1).obj <=Population(x).obj)
                    Population(x) = F(1);
                end
                if all(ConverIndexes([i,j])) && all(F(3).obj <=Population(x).obj)
                    Population(x) = F(3);
                end
            end
        end
    end

    %% Dividing distance variables
    Subcomponents = {};
    divided = false(1,Problem.D);
    while ~all(divided(ConverIndexes))
        x = find(~divided & ConverIndexes,1);
        while sum(any(interaction(x,ConverIndexes),1)) > length(x)
            x = find(any(interaction(x,:),1) & ConverIndexes);
        end
        Subcomponents = [Subcomponents,x];
        divided(x) = true;
    end
end
```

### `MOEADVA.m`
```matlab
classdef MOEADVA < ALGORITHM
% <2016> <multi> <real/integer> <large>
% Multi-objective evolutionary algorithm based on decision variable
% analyses
% NCA --- 20 --- The number of sampling solutions in control variable analysis
% NIA ---  6 --- The maximum number of tries required to judge the interaction

%------------------------------- Reference --------------------------------
% X. Ma, F. Liu, Y. Qi, X. Wang, L. Li, L. Jiao, M. Yin, and M. Gong. A
% multiobjective evolutionary algorithm based on decision variable analyses
% for multiobjective optimization problems with large-scale variables. IEEE
% Transactions Evolutionary Computation, 2016, 20(2): 275-298.
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
            [NCA,NIA] = Algorithm.ParameterSet(20,6);

            %% Control variable analysis
            [DiverIndexes,ConverIndexes] = ControlVariableAnalysis(Problem,NCA);

            %% Dividing distance variables based on two variable analyses
            [Subcomponents,Population] = DividingDistanceVariables(Problem,NIA,DiverIndexes,ConverIndexes);

            %% Calculate the neighbours of each individual
            PopDec = Population.decs;
            Dis    = pdist2(PopDec(:,DiverIndexes),PopDec(:,DiverIndexes));
            Dis(logical(eye(length(Dis)))) = inf;
            [~,Neighbour] = sort(Dis,2);
            Neighbour     = Neighbour(:,1:ceil(Problem.N/10));

            %% Subcomponent optimization
            if Problem.M == 2; threshold = 0.01; else threshold = 0.03; end
            while Algorithm.NotTerminated(Population)
                for i = 1 : length(Subcomponents)
                    drawnow('limitrate');
                    Population = SubcomponentOptimizer(Problem,Population,Neighbour,Subcomponents{i});
                end
            end
        end
    end
end
```

### `SubcomponentOptimizer.m`
```matlab
function Population = SubcomponentOptimizer(Problem,Population,Neighbour,indices)
% Subcomponent optimizer

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for i = 1 : length(Population)
        if rand < 0.9
            P = Neighbour(i,randperm(size(Neighbour,2),2));
        else
            P = randperm(length(Population),2);
        end
        OffDec          = Population(i).dec;
        NewDec          = OperatorDE(Problem,OffDec,Population(P(1)).dec,Population(P(2)).dec,{1,0.5,length(OffDec)/length(indices)/2,20});
        OffDec(indices) = NewDec(indices);
        Offspring       = Problem.Evaluation(OffDec);
        if sum(Offspring.obj) < sum(Population(i).obj)
            Population(i) = Offspring;
        end
    end
end
```

### `UDall.m`
```matlab
function Data = UDall(N,M)
% Generate N M-dimensional points by the uniform design

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://web.xidian.edu.cn/fliu/lunwen.html

    %% All the possible values of each points
    hm  = find(gcd(1:N,N)==1);
    udt = mod((1:N)'*hm,N); 
    udt(udt==0) = N;
    
    %% Choose M columns among hm as the output, which have the minimum CD2 value
    nCombination = nchoosek(length(hm),M);
    if nCombination < 1e4
        Combination = nchoosek(1:length(hm),M);
        CD2 = zeros(nCombination,1);
        for i = 1 : nCombination
            UT     = udt(:,Combination(i,:));
            CD2(i) = calCD2(UT);
        end
        [~,minIndex] = min(CD2);
        Data = udt(:,Combination(minIndex,:));
    else
        CD2 = zeros(N,1);
        for i = 1 : N
            UT     = mod((1:N)'*i.^(0:M-1),N);
            CD2(i) = calCD2(UT);
        end
        [~,minIndex] = min(CD2);
        Data = mod((1:N)'*minIndex.^(0:M-1),N);
        Data(Data==0) = N;
    end
    Data = (Data-1)/(N-1);
end

function CD2 = calCD2(UT)
% Calculate the CD2 (centered L2-discrepancy) value of the point set, to
% measure the uniformity of the points

    [N,S] = size(UT);
    X     = (2*UT-1)/(2*N);
    
    CS1 = sum(prod(2+abs(X-1/2)-(X-1/2).^2,2));
    CS2 = zeros(N,1);
    for i = 1 : N    
        CS2(i) = sum(prod((1+1/2*abs(repmat(X(i,:),N,1)-1/2)+1/2*abs(X-1/2)-1/2*abs(repmat(X(i,:),N,1)-X)),2));
    end
    CS2 = sum(CS2);
    CD2 = (13/12)^S-2^(1-S)/N*CS1+1/(N^2)*CS2;
end
```
