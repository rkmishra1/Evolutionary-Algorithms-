# LDS-AF

**Tags**: <2025> <multi> <real/integer> <large/none> <expensive>

## Description
Low-dimensional surrogate aggregation function

## Reference
H. Gu, H. Wang, C. He, B. Yuan, and Y. Jin. Large-scale multiobjective evolutionary algorithm guided by low-dimensional surrogates of scalarization functions. Evolutionary Computation, 2025, 33(3): 309-334.

## Source Code

### `LDSAF.m`
```matlab
classdef LDSAF < ALGORITHM
% <2025> <multi> <real/integer> <large/none> <expensive>
% Low-dimensional surrogate aggregation function

%------------------------------- Reference --------------------------------
% H. Gu, H. Wang, C. He, B. Yuan, and Y. Jin. Large-scale multiobjective
% evolutionary algorithm guided by low-dimensional surrogates of
% scalarization functions. Evolutionary Computation, 2025, 33(3): 309-334.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Haoran Gu

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [delta,N_s] = Algorithm.ParameterSet(0.9, 20);
            
            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            
            %% Detect the neighbours of each solution
            T      = ceil(Problem.N / 10);
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1 : T);
            
            %% Initialize population
            PopDec     = UniformPoint(Problem.N, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            A          = Population;
            Z          = min(Population.objs, [], 1);
            
            %% Define RBFN
            RBFN_list1 = RBFN_PP(Problem);
            RBFN_list2 = RBFN_P(Problem);
            P_center   = B;
            
            %% Optimization
            while Algorithm.NotTerminated(A)
                % For each sub-problem
                for i = 1 : Problem.N
                    %% Model-construction
                    if Problem.FE >= 350
                        pdb  = pdist2(W, W);
                        pds2 = pdb(i,B(i,:));
                        farv = find(pds2>=(max(pds2)-0.00001));
                        a    = randperm(length(farv));
                        
                        center = P_center(i,farv(a(1)));
                        
                        RBFN_list2(i) = RBFN_list2(i).ModelConstruction(A, B(i, :), W, Z,center);
                        model = RBFN_list2(i);
                    else
                        RBFN_list1(i) = RBFN_list1(i).ModelConstruction(A, B(i, :), W, Z);
                        model = RBFN_list1(i);
                    end
                    
                    %% Choose the parents
                    if rand < delta
                        P = B(i, randperm(end));
                    else
                        P = randperm(Problem.N);
                    end
                    
                    %% Solution-generation
                    y_i = SolutionSelection(Problem, Population, P, model, N_s, i);
                    
                    %% Evaluate offspring
                    y_i = Problem.Evaluation(y_i);
                    
                    %% Update the reference point
                    Z = min(Z, y_i.obj);
                    if Problem.FE <= 2*Problem.N
                        nr = 2;
                    else
                        nr = 4;
                    end
                    
                    %% Update population and archive
                    g_old = max(abs(Population(P).objs - repmat(Z, length(P), 1)) .* W(P, :), [], 2);
                    g_new = max(repmat(abs(y_i.obj - Z), length(P), 1) .* W(P, :), [], 2);
                    Population(P(find(g_old >= g_new, nr))) = y_i;
                    A = [A, y_i];
                    
                    %% Check termination criteria
                    Algorithm.NotTerminated(A);
                end
            end
        end
    end
end
```

### `RBFN_P.m`
```matlab
classdef RBFN_P
    
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Haoran Gu

    properties
        Problem;
        index;
        x;
        trainsample;
        vec;
        d;
        lam;
    end
    methods
        %% Constructor
        function obj = RBFN_P(Problem)
            if nargin == 1
                obj(1, Problem.N) = RBFN_P;
                for i = 1 : length(obj)
                    obj(i).index   = i;
                    obj(i).Problem = Problem;
                    obj(i).x       = [];
                    obj(i).trainsample;
                    obj(i).vec;
                    obj(i).d;
                    obj(i).lam;
                end
            end
        end
        %% Model-construction
        function obj = ModelConstruction(obj, A, B_i, W, Z,center)
            indices = 1 : length(A);
            for i = 1 : length(A)
                obj.x(i, :) = A(i).dec;
            end
            g_data1 = max(abs(A(indices).objs - repmat(Z, length(indices), 1)) .* W(center, :), [], 2);
            uniformed_xdata = zeros(length(A), obj.Problem.D);
            for i = 1 : length(A)
                uniformed_xdata(i, :) = obj.UniformInput(obj.x(i, :));
            end
            covx            = cov(uniformed_xdata);
            [vec_pop,~,~]   = pcacov(covx);
            uniformed_xdata = uniformed_xdata* vec_pop;
            obj.vec         = vec_pop;
            
            obj.d = obj.Problem.D/2;
            obj.trainsample = uniformed_xdata(:,1:obj.d);
            pair            = pdist2(obj.trainsample, obj.trainsample);
            D_max           = max(max(pair, [], 2));
            spread          = D_max * (obj.Problem.D * obj.Problem.N) ^ (-1 / obj.Problem.D);
            RBFModel        = newrbe(transpose(obj.trainsample), transpose(g_data1), spread);
            obj.lam         = RBFModel;
        end
        function predicted_value = PredictClass(obj, x)
            uniformed_xdata = zeros(size(obj.x,1), obj.Problem.D);
            uniformed_x     = zeros(size(x,1), obj.Problem.D);
            for i = 1 : size(obj.x,1)
                uniformed_xdata(i, :) = obj.UniformInput(obj.x(i, :));
            end
            for j = 1 : size(x,1)
                uniformed_x(j,:) = obj.UniformInput(x(j,:));
            end
            uniformed_x     = uniformed_x * obj.vec;
            test_x          = uniformed_x(:,1:obj.d);
            predicted_value = transpose(sim(obj.lam, transpose(test_x)));
        end
        function uniformed_x = UniformInput(obj, x)
            uniformed_x = ones(1, obj.Problem.D);
            for i = 1 : obj.Problem.D
                x_min = obj.Problem.lower(i);
                x_max = obj.Problem.upper(i);
                uniformed_x(i) = (x(i) - x_min) / (x_max - x_min);
            end
        end
    end
end
```

### `RBFN_PP.m`
```matlab
classdef RBFN_PP
    
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Haoran Gu

    properties
        Problem;
        index;
        x;
        trainsample;
        vec;
        d;
        lam;
    end
    methods
        %% Constructor
        function obj = RBFN_PP(Problem)
            if nargin == 1
                obj(1, Problem.N) = RBFN_PP;
                for i = 1 : length(obj)
                    obj(i).index   = i;
                    obj(i).Problem = Problem;
                    obj(i).x       = [];
                    obj(i).trainsample;
                    obj(i).vec;
                    obj(i).d;
                    obj(i).lam;
                end
            end
        end
        function obj = ModelConstruction(obj, A, B_i, W, Z)
            indices = 1 : length(A);
            for i = 1 : length(A)
                obj.x(i, :) = A(i).dec;
            end
            g_data1 = max(abs(A(indices).objs - repmat(Z, length(indices), 1)) .* W(B_i(1), :), [], 2);
            uniformed_xdata = zeros(length(A), obj.Problem.D);
            for i = 1 : length(A)
                uniformed_xdata(i, :) = obj.UniformInput(obj.x(i, :));
            end
            covx            = cov(uniformed_xdata);
            [vec_pop,~,~]   = pcacov(covx);
            uniformed_xdata = uniformed_xdata* vec_pop;
            obj.vec         = vec_pop;
            
            obj.d = obj.Problem.D/2;
            obj.trainsample = uniformed_xdata(:,1:obj.d);
            pair            = pdist2(obj.trainsample, obj.trainsample);
            D_max           = max(max(pair, [], 2));
            spread          = D_max * (obj.Problem.D * obj.Problem.N) ^ (-1 / obj.Problem.D);
            RBFModel        = newrbe(transpose( obj.trainsample), transpose(g_data1), spread);
            obj.lam         = RBFModel;
        end
        function predicted_value = PredictClass(obj, x)
            uniformed_xdata = zeros(size(obj.x,1), obj.Problem.D);
            uniformed_x     = zeros(size(x,1), obj.Problem.D);
            for i = 1 : size(obj.x,1)
                uniformed_xdata(i, :) = obj.UniformInput(obj.x(i, :));
            end
            for j = 1 : size(x,1)
                uniformed_x(j,:) = obj.UniformInput(x(j,:));
            end
            uniformed_x     = uniformed_x * obj.vec;
            test_x          = uniformed_x(:,1:obj.d);
            predicted_value = transpose(sim(obj.lam, transpose(test_x)));
        end
        function uniformed_x = UniformInput(obj, x)
            uniformed_x = ones(1, obj.Problem.D);
            for i = 1 : obj.Problem.D
                x_min = obj.Problem.lower(i);
                x_max = obj.Problem.upper(i);
                uniformed_x(i) = (x(i) - x_min) / (x_max - x_min);
            end
        end
    end
end
```

### `SolutionSelection.m`
```matlab
function y_i = SolutionSelection(Problem, Population, P, model, N_s, i)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Haoran Gu

    Prnd = nchoosek(P,2);
    Prnd = [Prnd;Prnd];
    Prnd(N_s/2+1:N_s,[1,2]) = Prnd(N_s/2+1:N_s,[2,1]);
    for r = 1 : N_s
        candidate(r,:) = OperatorDE(Problem, Population(i).dec, Population(Prnd(r,1)).dec, Population(Prnd(r,2)).dec);
    end
    c       = model.PredictClass(candidate);
    [~,pos] = min(c);
    y_i     = candidate(pos,:);
end
```
