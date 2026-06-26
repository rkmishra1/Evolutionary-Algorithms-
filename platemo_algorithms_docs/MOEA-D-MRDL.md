# MOEA-D-MRDL

**Tags**: <2015> <multi> <real/integer>

## Description
MOEA/D with maximum relative diversity loss

## Reference
S. B. Gee, K. C. Tan, V. A. Shim, and N. R. Pal. Online diversity assessment in evolutionary multiobjective optimization: A geometrical perspective. IEEE Transactions on Evolutionary Computation, 2015, 19(4): 542-559.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,Egamma] = EnvironmentalSelection(Population,Offspring,W,B,Z,gamma)
% The environmental selection of MOEA/D-MRDL

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Egamma = [];
    convergenceDirection = [];
    for i = randperm(length(Population))
        % Find the neighboring parents which have larger Tchebycheff scalar
        % function values than the offspring
        g_old = max(abs(Population(B(i,:)).objs-repmat(Z,size(B,2),1)).*W(B(i,:),:),[],2);
        g_new = max(abs(repmat(Offspring(i).obj-Z,size(B,2),1)).*W(B(i,:),:),[],2);
        PM    = B(i,g_old>g_new);
        % Replacement
        if ~isempty(PM)
            [~,nearest] = min(pdist2(Offspring(i).obj,Population(PM).objs));
            if isempty(convergenceDirection)
                % Record the convergence direction
                convergenceDirection = [convergenceDirection;Population(PM(nearest)).obj-Offspring(i).obj];
                % Replace the nearest parent with the offspring
                Population(PM(nearest)) = Offspring(i);
            else
                % Compute MRDL of each parent in PM to the offspring
                Sine1 = sqrt(1-(1-pdist2(Population(PM).objs,convergenceDirection,'cosine')).^2);
                Sine2 = sqrt(1-(1-pdist2(Offspring(i).obj,convergenceDirection,'cosine')).^2);
                RDL   = repmat(sqrt(sum(Population(PM).objs.^2,2)),1,size(Sine1,2)).*Sine1./repmat(norm(Offspring(i).obj).*Sine2,size(Sine1,1),1);
                MRDL  = max(RDL,[],2);
                if all(MRDL<=gamma)
                    % Record the convergence direction
                    convergenceDirection = [convergenceDirection;Population(PM(nearest)).obj-Offspring(i).obj];
                    % Replace the nearest parent with the offspring
                    Population(PM(nearest)) = Offspring(i);
                    % Record the MRDL for calculating the average MRDL
                    Egamma = [Egamma;MRDL(nearest)];
                end
            end
        end
    end
    % The average MRDL over the whole population
    Egamma = mean(Egamma);
end
```

### `MOEADMRDL.m`
```matlab
classdef MOEADMRDL < ALGORITHM
% <2015> <multi> <real/integer>
% MOEA/D with maximum relative diversity loss
% gamma --- 20 --- Maximum allowable maximum relative diversity loss
% nmov  --- 10 --- Size of moving average

%------------------------------- Reference --------------------------------
% S. B. Gee, K. C. Tan, V. A. Shim, and N. R. Pal. Online diversity
% assessment in evolutionary multiobjective optimization: A geometrical
% perspective. IEEE Transactions on Evolutionary Computation, 2015, 19(4):
% 542-559.
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
            [gamma,nmov] = Algorithm.ParameterSet(20,10);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);	% Weight vectors
            T = ceil(Problem.N/10);                             % Neighbourhood size

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T)';                          % The neighbours of each weight vector

            %% Generate random population
            Population = Problem.Initialization();	% The initial population
            Z = min(Population.objs,[],1);          % The ideal point

            %% Optimization
            disC      = 20;     % The distribution index of SBX
            Pn        = 0;      % The standard deviation of Gaussian permutation
            allEgamma = [];     % The MRDL array storing all the past MRDL values
            while Algorithm.NotTerminated(Population)
                % Select parents
                MatingPool = [B(randi(T,1,Problem.N)+(0:Problem.N-1)*T),B(randi(T,1,Problem.N)+(0:Problem.N-1)*T)];
                % Generate offsprings
                OffDec    = OperatorGAhalf(Problem,Population(MatingPool).decs,{1,disC,1,20});
                OffDec    = OffDec + randn(size(OffDec))*Pn;
                Offspring = Problem.Evaluation(OffDec);
                % Update the ideal point
                Z          = min([Z;Offspring.objs],[],1);
                % Environmental selection
                [Population,Egamma] = EnvironmentalSelection(Population,Offspring,W,B',Z,gamma);
                % Operator adaption
                [disC,Pn,allEgamma] = OperatorAdaption(disC,Pn,allEgamma,Egamma,nmov);
            end
        end
    end
end
```

### `OperatorAdaption.m`
```matlab
function [etaC,Pn,allEgamma]  = OperatorAdaption(etaC,Pn,allEgamma,Egamma,nmov)
% Adapt the parameters of operators according to MRDL

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Add the current MRDL to the MRDL array
    if isempty(allEgamma)
        if ~isnan(Egamma)
            allEgamma = Egamma;
        end
        return;
    else
        if ~isnan(Egamma)
            allEgamma = [allEgamma,Egamma];
        else
            allEgamma = [allEgamma,allEgamma(end)];
        end
    end

    %% Calculate the moving average of MRDL of each generation
    MAEgamma = zeros(size(allEgamma));
    for i = 1 : length(allEgamma)
        MAEgamma(i) = mean(allEgamma(max(1,i-nmov+1):i));
    end
    
    %% Calculate the predicted current MRDL
    Y      = log(MAEgamma(1:end-1))';
    Phi    = [ones(1,length(Y));1:length(Y)]';
    Lambda = Phi\Y;
    predictEgamma = exp(Lambda(1)+Lambda(2)*length(allEgamma));
    
    %% Update the parameters of operators
    if allEgamma(end) > predictEgamma
        etaC = max(etaC-2,2);
        Pn   = 0.5*(allEgamma(end)-predictEgamma);
    else
        etaC = etaC + 2;
        Pn   = 0;
    end
end
```
