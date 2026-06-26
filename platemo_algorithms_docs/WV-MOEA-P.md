# WV-MOEA-P

**Tags**: <2016> <multi> <real/integer>

## Description
Weight vector based multi-objective optimization algorithm with preference

## Reference
X. Zhang, X. Jiang, and L. Zhang. A weight vector based multi-objective optimization algorithm with preference. Acta Electronica Sinica (Chinese), 2016, 44(11): 2639-2645.

## Source Code

### `STM.m`
```matlab
function Population = STM(Population,W,z,znad)
% Selection based on STM model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N  = length(Population);
    NW = size(W,1);

    %% The modified Tchebycheff value of each solution on each subproblem
    g = zeros(N,NW);
    for i = 1 : N
        g(i,:) = max(repmat(abs(Population(i).obj-z),NW,1)./W,[],2)';
    end

    %% The perpendicular distance of each solution on each subproblem
    PopObj   = (Population.objs-repmat(z,N,1))./repmat(znad-z,N,1);
    Cosine   = 1 - pdist2(PopObj,W,'cosine');
    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NW).*sqrt(1-Cosine.^2);
    
    %% STM selection
    Fp  = zeros(1,NW);
    FX  = zeros(1,N);
    Phi = false(NW,N);
    while any(Fp==0)
        RemainW  = find(Fp==0);
        i        = RemainW(randi(length(RemainW)));
        RemainX  = find(~Phi(i,:));
        [~,best] = min(g(RemainX,i));
        j        = RemainX(best);
        Phi(i,j) = true;
        if FX(j) == 0
            Fp(i) = j;
            FX(j) = i;
        elseif Distance(j,i) < Distance(j,FX(j))
            Fp(i)     = j;
            Fp(FX(j)) = 0;
            FX(j)     = i;
        end
    end
    Population = Population(Fp);
end
```

### `WVMOEAP.m`
```matlab
classdef WVMOEAP < ALGORITHM
% <2016> <multi> <real/integer>
% Weight vector based multi-objective optimization algorithm with preference
% Points ---      --- Set of preferred points
% b      --- 0.05 --- Range of preferred region

%------------------------------- Reference --------------------------------
% X. Zhang, X. Jiang, and L. Zhang. A weight vector based multi-objective
% optimization algorithm with preference. Acta Electronica Sinica
% (Chinese), 2016, 44(11): 2639-2645.
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
            [Points,b] = Algorithm.ParameterSet(ones(1,Problem.M),0.05);

            %% Generate the weight vectors and random population
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Map the weight vectors
            Dis   = pdist2(W,W);
            B     = zeros(Problem.N,T);
            Group = ceil((1:Problem.N)/Problem.N*size(Points,1));
            for i = unique(Group)
                % The weight vectors around the i-th preferred point
                Current = find(Group==i);
                % Map the weight vectors
                W(Current,:) = 2*b.*W(Current,:) + Points(i,:) + b;
                % Detect the neighbours of each vector
                [~,rank]     = sort(Dis(Current,Current),2);
                B(Current,:) = Current(rank(:,1:T));
            end

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % For each group
                for i = unique(Group)
                    Current = find(Group==i);
                    % Generate an offspring for each solution
                    P = zeros(length(Current),2);
                    for j = 1 : size(P,1)
                        if rand < 0.9
                            P(j,:) = B(j,randperm(size(B,2),2));
                        else
                            P(j,:) = Current(randperm(length(Current),2));
                        end
                    end
                    Offspring = OperatorDE(Problem,Population(Current),Population(P(:,1)),Population(P(:,2)));
                    % Environmental selection
                    Z = min(Z,min(Offspring.objs,[],1));
                    Population(Current) = STM([Population(Current),Offspring],W(Current,:),Z,Z+ones(1,Problem.M));
                end
            end
        end
    end
end
```
