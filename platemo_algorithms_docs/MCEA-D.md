# MCEA-D

**Tags**: <2022> <multi/many> <real/integer> <expensive>

## Description
Multiple classifiers-assisted evolutionary algorithm based on decomposition

## Reference
T. Sonoda and M. Nakata. Multiple classifiers-assisted evolutionary algorithm based on decomposition for high-dimensional multi-objective problems. IEEE Transactions on Evolutionary Computation, 2022, 26(6): 1581-1595.

## Source Code

### `MCEAD.m`
```matlab
classdef MCEAD < ALGORITHM
% <2022> <multi/many> <real/integer> <expensive>
% Multiple classifiers-assisted evolutionary algorithm based on decomposition
% delta  --- 0.9 --- The probability of choosing parents locally
% nr     ---   2 --- Maximum number of solutions replaced by each offspring
% Rmax   ---  10 --- Maximum repeat time of offspring generation

%------------------------------- Reference --------------------------------
% T. Sonoda and M. Nakata. Multiple classifiers-assisted evolutionary
% algorithm based on decomposition for high-dimensional multi-objective
% problems. IEEE Transactions on Evolutionary Computation, 2022, 26(6):
% 1581-1595.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Masaya Nakata

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [delta, nr, R_max] = Algorithm.ParameterSet(0.9, 2, 10);
            
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
         
            %% Define SVM
            svm_list = SVM(Problem);
            
            %% Optimization
            while Algorithm.NotTerminated(A)
                % For each sub-problem
                for i = 1 : Problem.N
                    %% Model-construction
                    svm_list(i) = svm_list(i).ModelConstruction(A, B(i, :), W, Z);
                    
                    %% Choose the parents
                    if rand < delta
                        P = B(i, randperm(end));
                    else
                        P = randperm(Problem.N);
                    end
        
                    %% Solution-generation
                    y_i = SolutionGeneration(Problem, Population, P, svm_list(i), R_max, i);
        
                    %% Evaluate offspring
                    y_i = Problem.Evaluation(y_i);
        
                    %% Update the reference point
                    Z = min(Z, y_i.obj);
        
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

### `SVM.m`
```matlab
classdef SVM
% Definition of SVM model and fuctions used in MCEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Masaya Nakata

    properties 
        Problem           % Problem instance defined by PlatEMO
        index             % Index of sub-problem
        x                 % SVM input
        label             % Class of training input
        mdl               % SVM model
        C                 % SVM parameter C
        gamma             % SVM parameter gamma
    end

    methods
        %% Constructor
        function obj = SVM(Problem)
            if nargin == 1
                obj(1, Problem.N) = SVM;
                for i = 1 : length(obj)
                    obj(i).index   = i;
                    obj(i).Problem = Problem;
                    obj(i).C       = 1.0;
                    obj(i).gamma   = 1.0;
                    obj(i).x       = [];
                    obj(i).label   = [];
                    obj(i).mdl;
                end
            end
        end

        %% Model-construction
        function obj = ModelConstruction(obj, A, B_i, W, Z)
            % Initialization
            indices = [1 : length(A)];
            for i = 1 : length(A)
                obj.x(i, :)     = A(i).dec;
                obj.label(i, 1) = -1;
            end
            
            % Get the set of current best solutions of neighbor sub-problems
            C_i = [];
            for i = 1 : length(B_i)
                % Calculate scalarization funcion values
                g_data = max(abs(A(indices).objs - repmat(Z, length(indices), 1)) .* W(B_i(i), :), [], 2);
                
                % Get the set of current best solution avoiding duplicative selection
                [~, sorted_index] = sort(g_data);
                for j = 1 : length(sorted_index)
                    if ~ismember(sorted_index(j), C_i)
                        C_i = [C_i, sorted_index(j)];
                        obj.label(sorted_index(j), 1) = 1;
                        break;
                    end
                end
            end

            % Train SVM 
            uniformed_xdata = zeros(length(obj.label), obj.Problem.D);
            for i = 1 : length(obj.label)
                uniformed_xdata(i, :) = obj.UniformInput(obj.x(i, :));
            end
            sigma   = sqrt(1 / (2 * obj.gamma));
            obj.mdl = fitcsvm(uniformed_xdata, obj.label, 'BoxConstraint', obj.C, 'KernelScale', sigma, 'KernelFunction', 'rbf');
        end

        %% Predict the class of input and get the decision score function value
        function [predicted_class, score] = PredictClass(obj, x)
            % Uniform the input
            uniformed_x = obj.UniformInput(x);
            
            % Predict the class of input
            [predicted_class, score_list] = obj.mdl.predict(uniformed_x);
            
            % Return the decision score function value
            score = score_list(2);
        end

        %% Uniform the input
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

### `SolutionGeneration.m`
```matlab
function y_i = SolutionGeneration(Problem, Population, P, c_i, R_max, i)
% Solution-generation in MCEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Masaya Nakata

    for r = 1 : R_max 
        % Generate candidate solution
        candidate = OperatorDE(Problem, Population(i).dec, Population(P(1)).dec, Population(P(2)).dec);
        
        % Shuffle the parents
        rnd = randperm(length(P));
        P   = P(rnd);
        
        % Input the candidate solution to SVM
        [c, d_i] = c_i.PredictClass(candidate);
  
        if c == 1
        % If predicted label of the candidate solution is positive class
            % Return the candidate solution and terminate the process 
            y_i = candidate;
            return;
        else
        % If predicted label of the candidate solution is negative class
            % Choose the candidate solution having the best decision score function value
            if r == 1
                d_i_max = d_i;
                y_i     = candidate;
            elseif d_i_max < d_i
                d_i_max = d_i;
                y_i     = candidate;
            end
        end
    end
end
```
