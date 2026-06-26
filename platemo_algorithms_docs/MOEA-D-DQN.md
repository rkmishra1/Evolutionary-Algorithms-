# MOEA-D-DQN

**Tags**: <2023> <multi/many> <real/integer>

## Description
MOEA/D based on deep Q-network (Enhanced with proper Target Network)

## Reference
Y. Tian, X. Li, H. Ma, X. Zhang, K. C. Tan, and Y. Jin, Deep reinforcement learning based adaptive operator selection for evolutionary multi-objective optimization, IEEE Transactions on Emerging Topics in Computational Intelligence, 2023, 7(4): 1051-1064.

## Source Code

### `DE_rand_1.m`
```matlab
classdef DE_rand_1
% Differential evolution type 1

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name    % operator name
        F       % scaling factor
    end
    methods
        function obj = DE_rand_1()
            obj.name = 'OP3';
            obj.F    = 0.5;
        end
        function v = do(obj, OldChrom, r0, neighbourVector)
            r1 = neighbourVector(1);
            r2 = neighbourVector(2);
            x  = OldChrom(r0, :);
            v  = x + obj.F * (OldChrom(r1, :) - OldChrom(r2, :));
        end
    end
end
```

### `DE_rand_2.m`
```matlab
classdef DE_rand_2
% Differential evolution type 2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name    % operator name
        F       % scaling factor
    end
    methods
        function obj = DE_rand_2()
            obj.name = 'OP4';
            obj.F    = 0.5;
        end
        function v = do(obj, OldChrom, r0, neighbourVector)
            x  = OldChrom(r0, :);
            r1 = neighbourVector(1);
            r2 = neighbourVector(2);
            r3 = neighbourVector(3);
            r4 = neighbourVector(4);
            v  = x + obj.F * (OldChrom(r1, :) - OldChrom(r2, :) + OldChrom(r3, :) - OldChrom(r4, :));
        end
    end
end
```

### `DQN.m`
```matlab
classdef DQN < handle
% DQN network with proper Target Network and TD-learning target: r + gamma*maxQ(s',a')

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        % Constants
        MEMORY_CAPACITY     = 512
        BATCH_SIZE          = 16
        LR                  = 0.01          % learning rate
        EPSILON             = 0.9         	% greedy policy
        GAMMA               = 0.95          % discount factor for future rewards
        TARGET_REPLACE_ITER = 7             % target update frequency
        % Variables
        net                     % Q-network (main network)
        net_target              % Target Q-network (for stable training)
        n_states              	% state number
        n_actions            	% action number
        learn_step_counter      % for target updating
        memory_counter          % for storing memory
        memory_scurr            % current state
        memory_a              	% action
        memory_r             	% reward
        memory_snext          	% next state
    end
    methods
        function obj = DQN(inDim, outDim)
        % Network definition
            layers = [featureInputLayer(inDim, 'Name', 'input')
                      fullyConnectedLayer(128, 'Name', 'fc_1')
                      reluLayer('Name', 'relu_1')
                      fullyConnectedLayer(256, 'Name', 'fc_2')
                      reluLayer('Name', 'relu_2')
                      fullyConnectedLayer(128, 'Name', 'fc_3')
                      reluLayer('Name', 'relu_3')
                      fullyConnectedLayer(64, 'Name', 'fc_4')
                      reluLayer('Name', 'relu_4')
                      fullyConnectedLayer(32, 'Name', 'fc_5')
                      reluLayer('Name', 'relu_5')
                      fullyConnectedLayer(outDim, 'Name', 'fc_6')];
            lgraph  = layerGraph(layers);
            obj.net = dlnetwork(lgraph);
            
            % Create target network with same architecture
            obj.net_target = dlnetwork(lgraph);
            % Initialize target network with same weights as main network
            obj.net_target.Learnables = obj.net.Learnables;
            
            % Init params
            obj.n_states           = inDim;
            obj.n_actions          = outDim;
            obj.learn_step_counter = 0;
            obj.memory_counter     = 0;
            % Init memory
            obj.memory_scurr = zeros(inDim, obj.MEMORY_CAPACITY);
            obj.memory_a     = zeros(1, obj.MEMORY_CAPACITY);
            obj.memory_r     = zeros(1, obj.MEMORY_CAPACITY);
            obj.memory_snext = zeros(inDim, obj.MEMORY_CAPACITY);
        end
        function a = choose_action(obj, scurr)
        % Forward propagation, and choose action according to probability
            q_eval_raw = extractdata(predict(obj.net, dlarray(scurr, 'CB')));
            [~, idxs]  = sort(q_eval_raw);
            idxs       = obj.n_actions + 1 - idxs;           
            prob = obj.idxs2prob(idxs);
            prob = prob ./ sum(prob);
            a    = zeros(1, size(scurr, 2));
            for i = 1 : size(scurr, 2)
                a(i) = randsrc(1, 1, [1:obj.n_actions; prob(i,:)]);
            end
        end
        function prob = idxs2prob(obj, idxs)
        % Map from sorting index to probability
            prob_table = [70, 28, 10, 8];
            if idxs < 5
                prob = prob_table(idxs);
            else
                prob = 5;
            end
        end

        function obj = store_transition(obj, scurr, a, r, snext)
        % Store state transition table
            idx = rem(obj.memory_counter, obj.MEMORY_CAPACITY) + 1;
            obj.memory_scurr(:, idx) = scurr;
            obj.memory_a(:, idx)     = a;
            obj.memory_r(:, idx)     = r;
            obj.memory_snext(:, idx) = snext;
            obj.memory_counter       = obj.memory_counter + 1;
        end
        function learn(obj)
        % Train one step
            obj.learn_step_counter = obj.learn_step_counter + 1;
            
            % Update target network periodically
            if mod(obj.learn_step_counter, obj.TARGET_REPLACE_ITER) == 0
                obj.net_target.Learnables = obj.net.Learnables;
            end
            
            idxs        = randi(min(obj.MEMORY_CAPACITY, obj.memory_counter), 1, obj.BATCH_SIZE);
            batch_scurr = obj.memory_scurr(:, idxs);
            batch_a     = obj.memory_a(:, idxs);
            batch_r     = obj.memory_r(:, idxs);
            batch_snext = obj.memory_snext(:, idxs);
            
            [~,grad] = dlfeval(@obj.modelLoss, obj.net, obj.net_target, batch_scurr, batch_a, batch_r, batch_snext);
            obj.net     = dlupdate(@obj.sgdFunction, obj.net, grad);
        end
        function [loss, grad] = modelLoss(obj, net, net_target, scurr, a, r, snext)
        % Calculate loss and gradient using TD-learning
            % Current Q-values from main network
            q_eval_raw = predict(net, dlarray(scurr, 'CB'));
            q_eval     = dlarray(zeros(1, obj.BATCH_SIZE), 'CB');
            for i = 1 : obj.BATCH_SIZE
                q_eval(i) = q_eval_raw(a(i), i);
            end                      
            q_next_raw = extractdata(predict(net_target, dlarray(snext, 'CB')));
            q_next_max = max(q_next_raw, [], 1);  % max over actions
            q_target   = r + obj.GAMMA * q_next_max;  % TD target                       
            loss = mse(q_eval, dlarray(q_target, 'CB'));
            grad = dlgradient(loss, net.Learnables);
        end
        function upd_param = sgdFunction(obj, param, grad)
        % Update parameters
            upd_param = param - obj.LR .* grad;
        end
    end
end
```

### `MOEADDQN.m`
```matlab
classdef MOEADDQN < ALGORITHM
% <2023> <multi/many> <real/integer>
% MOEA/D based on deep Q-network (Enhanced with proper Target Network)

%------------------------------- Reference --------------------------------
% Y. Tian, X. Li, H. Ma, X. Zhang, K. C. Tan, and Y. Jin, Deep
% reinforcement learning based adaptive operator selection for evolutionary
% multi-objective optimization, IEEE Transactions on Emerging Topics in
% Computational Intelligence, 2023, 7(4): 1051-1064.
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
            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Operators
            crossOp = RecRL(Problem, W, Problem.maxFE, Problem.N);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);
            % Utility for each subproblem
            Pi     = ones(Problem.N,1);
            oldObj = max(abs((Population.objs-Z).*W),[],2);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                for i = 1 : 5
                    % Search for candidates
                    Boundary  = find(sum(W<1e-3,2)==Problem.M-1)';
                    Candidate = [Boundary,TournamentSelection(10,floor(Problem.N/5)-length(Boundary),-Pi)];
                    Candidate = unique(Candidate);
                    
                    % For each candidate
                    for j = 1 : size(Candidate,2)
                        % Choose neighbour, or treat all as neighbour
                        % indices of neighbor is stored in P
                        if rand < 0.9
                            P = B(Candidate(j),randperm(end));
                        else
                            P = randperm(Problem.N);
                        end
                        % Choose offspring
                        [crossOp, Offspring] = crossOp.do(Population.decs, Candidate(j), P, Problem.FE);
                        % Mutate
                        Offspring = PolyMut(Offspring, Problem.lower, Problem.upper);
                        % Evaluate
                        Offspring = Problem.Evaluation(Offspring);
                        % Determine which to update
                        Z = min(Z,Offspring.obj);
                        g_old = max(abs((Population(P).objs-Z).*W(P,:)),[],2);
                        g_new = max(abs((Offspring.obj-Z).*W(P,:)),[],2);
                        update_idxs = (g_old>=g_new);
                        % Update and train cross operator
                        if sum(update_idxs) >= 1
                            FIR = (g_old(update_idxs) - g_new(update_idxs)) ./ g_old(update_idxs);
                            Population(P(g_old>=g_new)) = Offspring;
                            crossOp = crossOp.learn(sum(FIR));
                        end
                    end
                end
                if ~mod(ceil(Problem.FE/Problem.N),50)
                    % Update Pi for each solution
                    newObj    = max(abs((Population.objs-Z).*W),[],2);
                    DELTA     = (oldObj-newObj)./oldObj;
                    Temp      = DELTA < 0.001;
                    Pi(~Temp) = 1;
                    Pi(Temp)  = (0.95+0.05*DELTA(Temp)/0.001).*Pi(Temp);
                    oldObj    = newObj;
                end
            end
        end
    end
end
```

### `PolyMut.m`
```matlab
function Population = PolyMut(Population, lb, ub)
% Polynomial mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    Population = min(max(Population, lb), ub);
    isMut      = rand(size(Population)) < 1 / size(Population, 2);
    u          = rand(size(Population));
    delta1     = (Population - lb) ./ (ub - lb);
    delta2     = (ub - Population) ./ (ub - lb);
    delta      = ((2.*u+(1-2.*u).*(1-delta1).^21).^(1/21)-1).*(u<=0.5) + (1-(2.*(1-u)+(2.*u-1).*(1-delta2).^21).^(1/21)).*(u>0.5);
    Population = Population + isMut .* delta .* (ub - lb);
    Population = min(max(Population, lb), ub);
end
```

### `RecM2m.m`
```matlab
classdef RecM2m < handle
% Crossover operator in MOEA/D-M2M

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name    % operator name
        MaxGen  % current genetic generation
    end
    methods
        function obj = RecM2m(maxgen)
            obj.name   = 'OP2';
            obj.MaxGen = maxgen;
        end
        function OffDec = do(obj, OldChrom, r0, neighbourVector, currentGen)
            [N, D] = size(OldChrom);
            r2     = datasample(neighbourVector, 1, 'Replace', false);
            p1     = OldChrom(r0, :);
            p2     = OldChrom(r2, :);
            rc     = (2 * rand(1) - 1) * (1 - rand(1) ^ (-(1 - currentGen / (obj.MaxGen + N)) ^ 0.7));
            OffDec = p1 + rc * (p1 - p2);
        end
    end
end
```

### `RecRL.m`
```matlab
classdef RecRL < handle
% Choosing crossover operator via reinforcement learning

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name              % operator name 
        problem         
        lambda_           % weight vector
        Opers             % candidate operators
        n                 % candidate operators number
        dqn               % Deep Q-Network 
        SW	              % sliding window
        a                 % action
        state             % current state
        state_            % next state
        countOpers        % count the selection frequency of different operators
        gen               % current genetic generation
    end
    methods
        function obj = RecRL(problem, lambda_, maxgen, NIND)
            obj.name       = 'RecRL';
            obj.problem    = problem;
            obj.lambda_    = lambda_;
            obj.Opers      = {Recsbx(), RecM2m(maxgen), DE_rand_1(), DE_rand_2()};
            obj.n          = length(obj.Opers);
            obj.dqn        = DQN(problem.D + problem.M, obj.n);
            obj.SW         = zeros(2, NIND * 4);
            obj.a          = 0;
            obj.state      = [];
            obj.state_     = [];
            obj.countOpers = zeros(obj.n, 1);
        end
        function [obj, offChrom] = do(obj, OldChrom, r0, neighbourVector, currentGen)
      	% Choose operator and generate offspring
            obj.gen   = currentGen;
            obj.state = [OldChrom(r0, :), obj.lambda_(r0, :)];
            if obj.dqn.memory_counter > 300
                obj.a = obj.dqn.choose_action(obj.state');
                for i = 1 : obj.n
                    if sum(obj.SW(1, :) == i) == 0
                        obj.a = i;
                        break;
                    end
                end
            else
                obj.a = randi(obj.n);
            end
            obj.countOpers(obj.a) = obj.countOpers(obj.a) + 1;
            if obj.a == 1
                offChrom = obj.Opers{1}.do(OldChrom, r0, neighbourVector);
            elseif obj.a == 2
                offChrom = obj.Opers{2}.do(OldChrom, r0, neighbourVector, currentGen);
            else
                offChrom = obj.Opers{obj.a}.do(OldChrom, r0, neighbourVector);
            end
            obj.state_ = [offChrom(1, :), obj.lambda_(r0, :)];
        end
        function obj = learn(obj, r)
        % Store state transition and (if necessary) update parameters in DQN
            obj.SW = [obj.SW(:, 2:end), [obj.a; r]];
            reward = max(obj.SW(2, obj.SW(1, :) == obj.a));
            obj.dqn.store_transition(obj.state, obj.a, reward, obj.state_);
            if obj.dqn.memory_counter > 200 && rand() < 0.2
                obj.dqn.learn();
            end
        end
    end
end
```

### `Recsbx.m`
```matlab
classdef Recsbx
% Simulated binary crossover

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name        % operator name
        XOVR        % cross probability
        Half        % half uniform cross
        n           % population distribution
    end
    
    methods
        function obj = Recsbx()
            obj.name = 'OP1';
            obj.XOVR = 0.7;
            obj.Half = true;
            obj.n    = 20;
        end
        function off = do(obj, OldChrom, r0, neighbourVector)
            r1   = datasample(neighbourVector, 1);
            p0   = OldChrom(r0, :);
            p1   = OldChrom(r1, :);
            D    = size(p1, 2);
            mu   = rand(1, D);
            beta = zeros(1, D);
            idx  = mu <= 0.5;
            beta(idx)  = (2 * mu(idx)).^(1 / (obj.n + 1));
            beta(~idx) = (2 - 2 * mu(~idx)).^(-1 / (obj.n + 1));
            beta = beta .* randsample([-1, 1], 1, true, [0.5, 0.5]);
            idx  = rand(1, D) < 0.5;
            beta(idx) = 1;
            if rand < 0.5
                off = (p0 + p1) / 2 + beta .* (p0 - p1) / 2;
            else
                off = (p0 + p1) / 2 - beta .* (p0 - p1) / 2;
            end
        end
    end
end
```
