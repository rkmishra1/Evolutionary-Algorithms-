# MOEA-D-AWA

**Tags**: <2014> <multi/many> <real/integer/label/binary/permutation>

## Description
MOEA/D with adaptive weight adjustment

## Reference
Y. Qi, X. Ma, F. Liu, L. Jiao, J. Sun, and J. Wu. MOEA/D with adaptive weight adjustment. Evolutionary Computation, 2014, 22(2): 231-264.

## Source Code

### `MOEADAWA.m`
```matlab
classdef MOEADAWA < ALGORITHM
% <2014> <multi/many> <real/integer/label/binary/permutation>
% MOEA/D with adaptive weight adjustment
% rate_update_weight --- 0.05 --- Ratio of updated weight vectors
% rate_evol          ---  0.8 --- Ratio of iterations to evolve with only MOEA/D
% wag                ---  100 --- Iteration interval of utilizing AWA

%------------------------------- Reference --------------------------------
% Y. Qi, X. Ma, F. Liu, L. Jiao, J. Sun, and J. Wu. MOEA/D with adaptive
% weight adjustment. Evolutionary Computation, 2014, 22(2): 231-264.
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
            [rate_update_weight,rate_evol,wag] = Algorithm.ParameterSet(0.05,0.8,100);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            % Transformation on W
            W = 1./W./repmat(sum(1./W,2),1,size(W,2));
            % Size of neighborhood
            T = ceil(Problem.N/10);
            % Maximum number of solutions replaced by each offspring
            nr = ceil(Problem.N/100);
            % Size of external elite
            nEP = ceil(Problem.N*1.5);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z          = min(Population.objs,[],1);
            Pi         = ones(Problem.N,1);
            oldObj     = max(abs((Population.objs-repmat(Z,Problem.N,1)).*W),[],2);

            %% Optimization
            EP = [];
            while Algorithm.NotTerminated(Population)
                if ~mod(ceil(Problem.FE/Problem.N),10)
                    % Allocation of computing resources
                    newObj    = max(abs((Population.objs-repmat(Z,Problem.N,1)).*W),[],2);
                    DELTA     = (oldObj-newObj)./oldObj;
                    Temp      = DELTA <= 0.001;
                    Pi(~Temp) = 1;
                    Pi(Temp)  = (0.95+0.05*DELTA(Temp)/0.001).*Pi(Temp);
                    oldObj    = newObj;
                end
                for subgeneration = 1 : 5
                    % Choose I
                    Bounday = find(sum(W<1e-3,2)==1)';
                    I = [Bounday,TournamentSelection(10,floor(Problem.N/5)-length(Bounday),-Pi)];

                    % Evolve each solution in I
                    Offspring(1:length(I)) = SOLUTION();
                    for i = 1 : length(I)
                        % Choose the parents
                        if rand < 0.9
                            P = B(I(i),randperm(size(B,2)));
                        else
                            P = randperm(Problem.N);
                        end

                        % Generate an offspring
                        Offspring(i) = OperatorGAhalf(Problem,Population(P(1:2)));

                        % Update the ideal point
                        Z = min(Z,Offspring(i).obj);

                        % Update the solutions in P by Tchebycheff approach
                        g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                        g_new = max(repmat(abs(Offspring(i).obj-Z),length(P),1).*W(P,:),[],2);
                        Population(P(find(g_old>=g_new,nr))) = Offspring(i);
                    end
                end
                if Problem.FE >= rate_evol*Problem.maxFE
                    % Adaptive weight adjustment
                    if isempty(EP)
                        EP = updateEP(Population,Offspring,nEP);
                    else
                        EP = updateEP(EP,Offspring,nEP);
                    end
                    if ~mod(ceil(Problem.FE/Problem.N),wag/5)
                        [Population,W] = updateWeight(Population,W,Z,EP,rate_update_weight*Problem.N);
                    end
                end
            end
        end
    end
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

    %% Select the non-dominated solutions
    EP    = [EP,Offsprings];
    EP    = EP(NDSort(EP.objs,1)==1);
    [N,M] = size(EP.objs);
    
    %% Delete the overcrowded solutions
    Dis = pdist2(EP.objs,EP.objs);
    Dis(logical(eye(length(Dis)))) = inf;
    Del = false(1,N);
    while sum(Del) < N-nEP
        Remain = find(~Del);
        subDis = sort(Dis(Remain,Remain),2);
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

    [N,M] = size(Population.objs);

    %% Update the current population by EP
    % Calculate the function value of each solution in Population or EP on
    % each subproblem in W
    Combine    = [Population,EP];
    CombineObj = abs(Combine.objs-repmat(Z,length(Combine),1));
    g = zeros(length(Combine),size(W,1));
    for i = 1 : size(W,1)
        g(:,i) = max(CombineObj.*repmat(W(i,:),length(Combine),1),[],2);
    end
    % Choose the best solution for each subproblem
    [~,best]   = min(g,[],1);
    Population = Combine(best);
    
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
    temp    = 1./(newObjs-repmat(Z,size(newObjs,1),1));
    W       = [W;temp./repmat(sum(temp,2),1,size(temp,2))];
    % Add new solutions
    Population = Combine(Selected);
end
```
