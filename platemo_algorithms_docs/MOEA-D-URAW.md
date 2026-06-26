# MOEA-D-URAW

**Tags**: <2019> <multi/many> <real/integer/label/binary/permutation>

## Description
MOEA/D with uniform randomly adaptive weights

## Reference
L. R. C. Farias and A. F. R. Araujo. Many-objective evolutionary algorithm based on decomposition with random and adaptive weights. Proceedings of the IEEE International Conference on Systems, Mans and Cybernetics, 2019.

## Source Code

### `MOEADURAW.m`
```matlab
classdef MOEADURAW < ALGORITHM
% <2019> <multi/many> <real/integer/label/binary/permutation>
% MOEA/D with uniform randomly adaptive weights
% delta --- 0.9 --- The probability of choosing parents locally
% nr    ---   2 --- Maximum number of solutions replaced by each offspring

%------------------------------- Reference --------------------------------
% L. R. C. Farias and A. F. R. Araujo. Many-objective evolutionary
% algorithm based on decomposition with random and adaptive weights.
% Proceedings of the IEEE International Conference on Systems, Mans and
% Cybernetics, 2019.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [delta,nr] = Algorithm.ParameterSet(0.9,2);

            %% Generate the weight vectors
            [W,Problem.N] = UniformlyRandomlyPoint(Problem.N,Problem.M);
            % Transformation on W
            W = 1./W./repmat(sum(1./W,2),1,size(W,2));
            % Size of neighborhood
            T = ceil(Problem.N/10);
            % Size of external elite
            nEP = ceil(Problem.N*2);
            % Ratio of updated weight vectors
            nus = 0.05;

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z          = min(Population.objs,[],1);

            %% Optimization
            EP = [];
            adaptation_moment = round(ceil(Problem.maxFE/Problem.N)*0.05);
            while Algorithm.NotTerminated(Population)
                % For each solution	
                Offsprings(1:Problem.N) = SOLUTION();
                for i = 1 : Problem.N
                    % Choose the parents
                    if rand < delta
                        P = B(i,randperm(size(B,2)));
                    else
                        P = randperm(Problem.N);
                    end

                    % Generate an offspring
                    Offsprings(i) = OperatorGAhalf(Problem,Population(P(1:2)));

                    % Update the ideal point
                    Z = min(Z,Offsprings(i).obj);

                    % Update the solutions in P by Tchebycheff approach
                    g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                    g_new = max(repmat(abs(Offsprings(i).obj-Z),length(P),1).*W(P,:),[],2);
                    Population(P(find(g_old>=g_new,nr))) = Offsprings(i);
                end
                if Problem.FE/Problem.maxFE <= 0.9
                    if isempty(EP)
                        EP = updateEP(Population,Offsprings,nEP);
                    else
                        EP = updateEP(EP,Offsprings,nEP);
                    end
                    if mod(ceil(Problem.FE/Problem.N),adaptation_moment) == 0
                        % Adaptive weight adjustment          
                        [Population,W] = updateWeight(Population,W,Z,EP,nus*Problem.N); 
                        B = pdist2(W,W);
                        [~,B] = sort(B,2);
                        B = B(:,1:T);
                    end
                end
            end
        end
    end
end
```

### `UniformlyRandomlyPoint.m`
```matlab
function [W1,N] = UniformlyRandomlyPoint(N,M)
%UniformlyRandomlyPoint - Generate a set of uniform randomly distributed points on
%the unit hyperplane
%
%   [W,N] = UniformlyRandomlyPoint(N,M) returns N uniform randomly distributed
%   points with M objectives.
%
%   Example:
%       [W,N] = UniformlyRandomlyPoint(275,10)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias
	
    W1 = eye(M,M);
	W1 = [W1;ones(1,M)/M];
    
	W2 = rand(5000,M);
	W2 = W2./repmat(sum(W2,2),1,size(W2,2));
	
	while size(W1,1) < N
		index = find_index_with_largest_distance (W1,W2);
		W1(size(W1,1)+1,:) = W2(index,:);
		W2(index,:) = [];
	end	

    W1 = max(W1,1e-6);
    N  = size(W1,1);

end

function index = find_index_with_largest_distance (W1,W2)
    Distance = pdist2(W2,W1);
    Temp     = sort(Distance,2);
    [~,Rank] = sortrows(Temp);
    index    = Rank(length(Rank));
end
```

### `updateEP.m`
```matlab
function EP = updateEP(EP,Offsprings,nEP)
% Update the external population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias

    %% Select the non-dominated solutions
    EP    = [EP,Offsprings];
    EP    = EP(NDSort(EP.objs,1)==1);
    [N,M] = size(EP.objs);
	
    %% Delete the overcrowded solutions
    Dis = pdist2(EP.objs,EP.objs);
    Dis(logical(eye(length(Dis)))) = inf;
    Del = false(1,N);
    while sum(Del) < N-nEP
        Remain    = find(~Del);
        subDis    = sort(Dis(Remain,Remain),2);
        [~,worst] = min(prod(subDis(:,1:min(M,length(Remain))),2));
        Del(Remain(worst)) = true;
    end   
    EP = EP(~Del);
end
```

### `updateWeight.m`
```matlab
function [Population,W] = updateWeight(Population,W,Z,EP,nus)
% Delete overcrowded subproblems and add new subproblems

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias
    
	%% Parameter setting
    [N,M] = size(Population.objs);
    
    %% Delete the overcrowded subproblems
    Dis = pdist2(Population.objs,Population.objs);
    Dis(logical(eye(length(Dis)))) = inf;
    Del = false(1,length(Population));
    while sum(Del) < min(nus,length(EP))
        Remain    = find(~Del);
        subDis    = sort(Dis(Remain,Remain),2);
        [~,worst] = min(prod(subDis(:,1:min(M,length(Remain))),2));
        Del(Remain(worst)) = true;
    end
    Population = Population(~Del);
    W = W(~Del,:);
    
    %% Add new subproblems
    % Determine the new solutions be added
    Combine  = [Population,EP];
    Selected = false(1,length(Combine));
    Selected(1:length(Population)) = true;
    Dis = pdist2(Combine.objs,Combine.objs);
    Dis(logical(eye(length(Dis)))) = inf;
    while sum(Selected) < min(N,length(Selected))
        subDis   = sort(Dis(~Selected,Selected),2);
        [~,best] = max(prod(subDis(:,1:min(M,size(subDis,2))),2));
        Remain   = find(~Selected);
        Selected(Remain(best)) = true;
    end
    % Add new subproblems
    newObjs = EP(Selected(length(Population)+1:end)).objs;
    temp    = 1./(newObjs-repmat(Z,size(newObjs,1),1));
    temp(temp==inf) = 0.999999; % when (temp == Z) then 0
    W = [W;temp./repmat(sum(temp,2),1,size(temp,2))];
    % Add new solutions
    Population = Combine(Selected);
end
```
