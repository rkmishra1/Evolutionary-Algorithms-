# MOEA-D-DCWV

**Tags**: <2019> <multi/many> <real/integer/label/binary/permutation>

## Description
MOEA/D with distribution control of weight vector set

## Reference
T. Takagi, K. Takadama, and H. Sato. A distribution control of weight vector set for multi-objective evolutionary algorithms. Proceedings of the Bio-inspired Information and Communication Technologies, Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, 2019, 70-80.

## Source Code

### `MOEADDCWV.m`
```matlab
classdef MOEADDCWV < ALGORITHM
% <2019> <multi/many> <real/integer/label/binary/permutation>
% MOEA/D with distribution control of weight vector set
% p --- -1 --- The intermediate objective value

%------------------------------- Reference --------------------------------
% T. Takagi, K. Takadama, and H. Sato. A distribution control of weight 
% vector set for multi-objective evolutionary algorithms. Proceedings of 
% the Bio-inspired Information and Communication Technologies, Lecture 
% Notes of the Institute for Computer Sciences, Social Informatics and 
% Telecommunications Engineering, 2019, 70-80.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Tomoaki Takagi

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            p = Algorithm.ParameterSet(-1);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Set the weight vectors
            [W,W0] = setWeight(W,p);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if ~isempty(W0)
                    W = updateWeight(Population.objs, W0);
                end
                % For each solution
                for i = 1 : Problem.N
                    % Choose the parents
                    P = B(i,randperm(size(B,2)));

                    % Generate an offspring
                    Offspring = OperatorGAhalf(Problem,Population(P(1:2)));

                    % Update the ideal and nadir point
                    Z = min(Z,Offspring.obj);
                    Zmax  = max(Population.objs,[],1);

                    % Update the solutions in P by Modified Tchebycheff approach with normalization
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1)./W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1)./W(P,:),[],2);
                    Population(P(g_old>=g_new)) = Offspring;
                end
            end
        end
    end
end
```

### `setWeight.m`
```matlab
function [W,W0] = setWeight(W,p)
% Set the weight vectors

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Tomoaki Takagi

    if 0 <= p && p <= 1
        %% Static approach
        W0 = [];
        M  = size(W,2);
        % Distribution control of weight vector set
        TF     = W < 1.0 / M;
        W(TF)  = W(TF) * p * M;
        W(~TF) = 1.0 - (1.0 - W(~TF)) * (1.0 - p) * M / (M - 1);
    else
        %% Dynamic approach
        W0 = W;
    end
end
```

### `updateWeight.m`
```matlab
function W = updateWeight(objs,W)
% Update the weight vectors

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Tomoaki Takagi

    % Calculate the intermediate objective value p
    M       = size(objs,2);
    objs    = normalize(objs,'range');
    normP   = sqrt(sum(objs.^2,2));
    CosineP = sum(objs./M,2).*sqrt(M)./normP;
    [~,I]   = min(normP.*sqrt(1-CosineP.^2));
    p       = normP(I)*CosineP(I) / sqrt(M);
    % Distribution control of weight vector set
    TF     = W < 1.0 / M;
    W(TF)  = W(TF) * p * M;
    W(~TF) = 1.0 - (1.0 - W(~TF)) * (1.0 - p) * M / (M - 1);
end
```
