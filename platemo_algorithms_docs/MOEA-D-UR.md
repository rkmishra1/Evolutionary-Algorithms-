# MOEA-D-UR

**Tags**: <2022> <multi/many> <real/integer/label/binary/permutation>

## Description
MOEA/D with update when required

## Reference
L. R. de Farias and A. F. Araujo. A decomposition-based many-objective evolutionary algorithm updating weights when required. Swarm and Evolutionary Computation, 2022, 68: 100980.

## Source Code

### `MOEADUR.m`
```matlab
classdef MOEADUR < ALGORITHM
% <2022> <multi/many> <real/integer/label/binary/permutation>
% MOEA/D with update when required
% start  ---  0.2 --- Start adaptation
% finish --- 0.93 --- Finish adaptation
% K      ---   10 --- Number of divisions of the objective space

%------------------------------- Reference --------------------------------
% L. R. de Farias and A. F. Araujo. A decomposition-based many-objective
% evolutionary algorithm updating weights when required. Swarm and
% Evolutionary Computation, 2022, 68: 100980.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
	
% This function is written by Lucas Farias

	methods
		function main(Algorithm,Problem)
            %% Parameter setting
            [start, finish, K] = Algorithm.ParameterSet(0.2, 0.93, 10);

            %% Parameter setting
            delta = 0.9;                % The probability of choosing parents locally
            nr    = 2;                  % Maximum number of solutions replaced by each offspring
            T     = ceil(Problem.N/10); % Size of neighborhood
            mini_generation = 1;        % The number of generations carried out within the objective space division method

            %% Generate the weight vectors
            [W,Problem.N] = UniformlyRandomlyPoint(Problem.N,Problem.M);    
            W     = 1./W./repmat(sum(1./W,2),1,size(W,2)); % WS-Transformation on W
            W_URP = W;

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z          = min(Population.objs,[],1);  
            EP = Population(NDSort(Population.objs,1)==1); % external population

            WhenDoesItStart	= floor(start*(Problem.maxFE/Problem.N));
            whenItEnds		= floor(finish*(Problem.maxFE/Problem.N));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % For each solution	
                Offsprings(1:Problem.N) = SOLUTION();       
                chosenNeighborhood      = rand(Problem.N,1) < delta;		
                for i = 1 : Problem.N
                    % Choose the parents
                    if chosenNeighborhood(i)==1
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

                if Problem.FE/Problem.N == WhenDoesItStart			
                    X = unique(Population.objs,'rows');
                    X = X(NDSort(X,1)==1,:);
                    X = normalize(X,'norm');        % Normalizada da Pop via segunda norma
                    spreading_index = norm(X)/4;	% Norma L2 na Pop Normalizada

                    fun_threshold=[-1.989e-05;0.0002034;0.03376;0.2373];
                    threshold = polyval(fun_threshold,Problem.M);

                    if spreading_index<=threshold   % Regular MOP
                        period = 12;                % period between adaptation
                        nus    = 0.25;              % number of updated subproblems
                    else % Irregular MOP
                        period = 28;                % period between adaptation
                        nus    = 0.075;             % number of updated subproblems
                    end

                    rate_update_weight = round(nus*Problem.N);	% Ratio of updated weight vectors

                    fun_rho = [-0.4707,0.8644,-0.1508,0.05745];
                    rho     = polyval(fun_rho,spreading_index);	% convergence threshold           

                    I_old = max(abs((Population.objs-repmat(Z,Problem.N,1)).*W),[],2);
                end

                % EP update
                if Problem.FE/Problem.N <= whenItEnds    
                    EP = [EP,Offsprings];
                end

                if  Problem.FE/Problem.N >= WhenDoesItStart && Problem.FE/Problem.N <= whenItEnds           
                    if ~mod(Problem.FE/Problem.N,period)
                        % Improvement Metric
                        I_new = max(abs((Population.objs-repmat(Z,Problem.N,1)).*W),[],2);
                        improvement_Metric = mean(1-(I_new./I_old));
                        I_old = I_new;

                        if abs(improvement_Metric) <= rho   
                            % EP update
                            EP = unique(EP);
                            EP = EP(NDSort(EP.objs,1)==1);    

                            % adaptive weight adjustment				
                            [Population,W,B] = updateWeight(Population,W,Z,T,EP,rate_update_weight);
                            I_old = max(abs((Population.objs-repmat(Z,Problem.N,1)).*W),[],2);

                            % division of objective space    
                            [Population,Z] = space_divide(Problem,EP,Population,Z,W_URP,W,K,mini_generation);
                            EP = [];
                        end
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
%UniformlyRandomlyPoint - Generate a set of uniform randomly distributed
%points on the unit hyperplane
%
%   [W,N] = UniformlyRandomlyPoint(N,M) returns N uniform randomly
%   distributed points with M objectives.
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

### `space_divide.m`
```matlab
function [Population,Z]=space_divide(Problem,EP,Population,Z,W_URP,W,K,mini_generation)
% Division of the Objective Space

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias

	[idx,~] = kmeans(Population.objs,K); 
	allInd  = [Population EP];

	%% For each group
	for current_group_index=1:max(idx)		
		%% Define the current group
		currentGroup = Population(idx(:)==current_group_index);
		group_size   = length(currentGroup);
        if length(unique(currentGroup)) > 2
            %% Generate the normalized weight vectors
            W_normalized = W_URP(idx(:)==current_group_index,:);
            w_maximum    = max(W_normalized);
            w_minimums   = min(W_normalized);
            W_normalized = (W_URP.*repmat((w_maximum-w_minimums),length(W_URP),1)) + repmat(w_minimums,length(W),1); %normalization in variable range(x,y)

            %% Associate individuals to weights
            currentGroup = associate_newW2allInd(allInd,W_normalized,Z);

            [candidates,Z] = mini_gen(Problem,currentGroup,Z,group_size,mini_generation);
            allInd         = [allInd candidates];
        end
	end
	Population = combine_allInd2Population(Problem,Population,allInd,W,Z);
end

function Population = associate_newW2allInd(candidates,W,Z)   
	Combine    = candidates;
    CombineObj = abs(Combine.objs-repmat(Z,length(Combine),1));
    g = zeros(length(Combine),size(W,1));
    for i = 1 : size(W,1)
        g(:,i) = max(CombineObj.*repmat(W(i,:),length(Combine),1),[],2);
    end
    % Choose the best solution for each subproblem
    [~,best]   = min(g,[],1);
    Population = Combine(best);
end

function [currentGroup,Z] = mini_gen(Problem,currentGroup,Z,group_size,mini_generation)
	generation = 1;
	while generation <= mini_generation
		
		% Choose the parents
		parents    = unique(currentGroup);
		MatingPool = randi(length(parents),1,group_size);
		% Generate offsprings
        Offsprings = OperatorGA(Problem,parents(MatingPool));
        
		% Update the ideal point
		Z = min([Z;Offsprings.objs],[],1);
		
		currentGroup = [currentGroup Offsprings];
		
        generation = generation + 1;
	end
end

function Population = combine_allInd2Population(Problem,Population,candidates,W,Z)
   
    Combine = unique(candidates);
    if length(Combine) < Problem.N
        Combine = candidates;
    end
    CombineObj = abs(Combine.objs-repmat(Z,length(Combine),1));
    g = zeros(length(Combine),size(W,1));
    for i = 1 : size(W,1)
        g(:,i) = max(CombineObj.*repmat(W(i,:),length(Combine),1),[],2);
    end
    % Choose the best solution for each subproblem
    [~,best]   = min(g,[],1);
    candidates = Combine(best);
    candidates = unique(candidates);
	
    for i = 1 : length(candidates)        
        % Global Replacement
        all_g_TCH  = max(abs((candidates(i).obj-repmat(Z,Problem.N,1)).*W),[],2);
        best_g_TCH = min(all_g_TCH);
        Chosen_one = find(all_g_TCH(:,1)==best_g_TCH);            
        % Update the solutions in P by Tchebycheff approach        
        if Population(Chosen_one) ~= candidates(i)
            g_old = max(abs(Population(Chosen_one).objs-repmat(Z,length(Chosen_one),1)).*W(Chosen_one,:),[],2);
            g_new = max(repmat(abs(candidates(i).obj-Z),length(Chosen_one),1).*W(Chosen_one,:),[],2);
            Population(Chosen_one(g_old>=g_new)) = candidates(i);
        end
    end
end
```

### `updateWeight.m`
```matlab
function [Population,W,B] = updateWeight(Population,W,Z,T,EP,nus)
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
        Remain = find(~Del);
        subDis = sort(Dis(Remain,Remain),2);
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
        subDis = sort(Dis(~Selected,Selected),2);
        [~,best] = max(prod(subDis(:,1:min(M,size(subDis,2))),2));
        Remain = find(~Selected);
        Selected(Remain(best)) = true;
    end
    % Add new subproblems
    newObjs = EP(Selected(length(Population)+1:end)).objs;
    temp    = 1./(newObjs-(repmat(Z,size(newObjs,1),1) - 0.001));
    temp(temp==inf) = 0.999999; %1e-6; % when (temp == Z) then 0
    W = [W;temp./repmat(sum(temp,2),1,size(temp,2))];
	
	% Redetect the neighbours of each solution
	B = pdist2(W,W);
	[~,B] = sort(B,2);
	B = B(:,1:T);
	
    % Add new solutions
    Population = Combine(Selected);
end
```
