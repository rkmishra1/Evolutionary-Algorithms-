# A-NSGA-III

**Tags**: <2014> <multi/many> <real/integer/label/binary/permutation> <constrained/none>

## Description
Adaptive NSGA-III

## Reference
H. Jain and K. Deb. An evolutionary many-objective optimization algorithm using reference-point based non-dominated sorting approach, part II: Handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation, 2014, 18(4): 602-622.

## Source Code

### `ANSGAIII.m`
```matlab
classdef ANSGAIII < ALGORITHM
% <2014> <multi/many> <real/integer/label/binary/permutation> <constrained/none>
% Adaptive NSGA-III

%------------------------------- Reference --------------------------------
% H. Jain and K. Deb. An evolutionary many-objective optimization algorithm
% using reference-point based non-dominated sorting approach, part II:
% Handling constraints and extending to an adaptive approach. IEEE
% Transactions on Evolutionary Computation, 2014, 18(4): 602-622.
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
            %% Generate the reference points and random population
            % All the reference points
            [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Z = sortrows(Z);
            % Distance between two consecutive reference points for the adaption
            interval = Z(1,end) - Z(2,end);
            % Initial population
            Population = Problem.Initialization();
            % Ideal point
            Zmin = min(Population(all(Population.cons<=0,2)).objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,Z,Zmin);
                Z          = Adaptive(Population.objs,Z,Problem.N,interval);
            end
        end
    end
end
```

### `Adaptive.m`
```matlab
function Z = Adaptive(PopObj,Z,N,interval)
% Addition and deletion of reference points

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    M = size(PopObj,2);
    
    %% Addition of reference points
    rho   = Associate(PopObj,Z);
    old_Z = [];
    while any(rho>=2) && ~isequal(old_Z,Z)
        old_Z = Z;
        for i = find(rho>=2)
            p = repmat(Z(i,:),M,1) - interval/M;
            p(logical(eye(M))) = p(logical(eye(M))) + interval;
            Z = [Z;p];
        end
        Z(any(Z<0,2),:) = [];
        [~,index]       = unique(round(Z*1e4)/1e4,'rows','stable');
        Z               = Z(index,:);
        rho = Associate(PopObj,Z);
    end
    
    %% Deletion of reference points
    Z(intersect(N+1:size(Z,1),find(~rho)),:) = [];
end

function rho = Associate(PopObj,Z)
% Associate each solution with one reference point

    %% Calculate the distance of each solution to each reference vector
    NormP    = sqrt(sum(PopObj.^2,2));
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    Distance = repmat(NormP,1,size(Z,1)).*sqrt(1-Cosine.^2);
    
    %% Associate each solution with its nearest reference point
    [~,pi] = min(Distance',[],1);
    
    %% Calculate the number of associated solutions of each reference point
    rho = hist(pi,1:size(Z,1));
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,Z,Zmin)
% The environmental selection of NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,N-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Choose = LastSelection(PopObj1,PopObj2,K,Z,Zmin)
% Select part of the solutions in the last front

    PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
    [N,M]  = size(PopObj);
    N1     = size(PopObj1,1);
    N2     = size(PopObj2,1);
    NZ     = size(Z,1);

    %% Normalization
    % Detect the extreme points
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));
    end
    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes
    Hyperplane = PopObj(Extreme,:)\ones(M,1);
    a = 1./Hyperplane;
    if any(isnan(a))
        a = max(PopObj,[],1)';
    end
    % Normalization
    PopObj = PopObj./repmat(a',N,1);
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    % Associate each solution with its nearest reference point
    [d,pi] = min(Distance',[],1);

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = hist(pi(1:N1),1:NZ);
    
    %% Environmental selection
    Choose  = false(1,N2);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);
        Jmin = find(rho(Temp)==min(rho(Temp)));
        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose==0 & pi(N1+1:end)==j);
        % Then select one solution associated with this reference point
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(d(N1+I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end
```
