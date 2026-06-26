# CMODRL

**Tags**: <2025> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Constrained multiobjective optimization via deep reinforcement learning

## Reference
F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. Automated configuration of evolutionary algorithms via deep reinforcement learning for constrained multiobjective optimization. IEEE Transactions on Cybernetics, 2025, 55(12): 5877-5890.

## Source Code

### `CMODRL.m`
```matlab
classdef CMODRL < ALGORITHM
% <2025> <multi> <real/integer/label/binary/permutation> <constrained>
% Constrained multiobjective optimization via deep reinforcement learning
% reward_step --- 5 --- Reward step setting

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, B. Xue, M. Zhang, and Y. Jin. Automated configuration
% of evolutionary algorithms via deep reinforcement learning for
% constrained multiobjective optimization. IEEE Transactions on
% Cybernetics, 2025, 55(12): 5877-5890.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            reward_step = Algorithm.ParameterSet(5);
            g           = 0;
            Data        = [];
            op_Data     = [];
            PopData     = [];
            model_built = 0;
            greedy      = 0.95;
            gama        = 0.95;
            step        = 0;
            accumulate  = 0;
            num_op      = 2;
            
            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization(ceil(Problem.N/2));
            Archive     = EnvironmentalSelection([Population1,Population2],Problem.N,true);
            initial_cv  = sum(sum(max(0,Population1.cons),2))/Problem.N;
            F_e         = rand/5 * initial_cv;
            
            %% Evaluate solutions based on static environment at initial iteration
            Fitness1 = CalFitness(Population1.objs,Population1.cons);
            Fitness2 = CalFitness(Population2.objs);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                g = g + 1;
                if g <= 100
                    % Generate offspring
                    MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                    MatingPool2 = TournamentSelection(2,Problem.N/2,Fitness2);
                    Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                    Offspring2  = OperatorGA(Problem,Population2(MatingPool2));
                    
                    % Update archive
                    [Archive,~,~] = EnvironmentalSelection([Archive,Offspring1,Offspring2],Problem.N,true);
                    
                    % Select based on learned environment and static environment
                    [Population1,Fitness1]   = EnvironmentalSelectionFe([Population1,Offspring1,Offspring2],Problem.N,F_e);
                    [Population2,Fitness2,~] = EnvironmentalSelection([Population2,Offspring1,Offspring2],Problem.N/2,false); 
                    
                % Randomly sample in the early 600 generations for exploration
                elseif g <= 600
                    step = step + 1;
                    if step == 1
                        accumulate = 0;
                        
                        % Random F_e and operator
                        average_CV = sum(sum(max(0,Population1.cons),2))/Problem.N;
                        F_e = rand/5 * average_CV;
                        if F_e <= 0
                            F_e = rand/5 * initial_cv * (1-g/ceil(Problem.maxFE/2*Problem.N));
                        end
                        action_op = randi(num_op);
                        
                        % old state
                        % use archive solutions and population1 solutions
                        reward_f  = sum(sum(Archive.objs,2))/Problem.N;
                        reward_cv = sum(sum(max(0,Archive.cons),2))/length(Archive);
                        f_max     = max(Archive.objs,[],1);
                        f_min     = min(Archive.objs,[],1);
                        reward_d  = sum(f_max-f_min);
                        
                        % use population1 solutions
                        % build the DAE/RBM model to estimate the state
                        dae   = DAE(Problem.M,1,10,Problem.N,0.5,0.5,0.1);
                        dae.train(PopData);
                        state = dae.reduce(Population1.objs);
                        state = state';
                        
                    end
                    
                    % Generate offspring
                    if action_op == 1
                        % use ordinary DE
                        MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                        MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                        Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                        Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                    else
                        % use GA
                        MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                        MatingPool2 = TournamentSelection(2,Problem.N/2,Fitness2);
                        Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                        Offspring2  = OperatorGA(Problem,Population2(MatingPool2));
                    end
                    
                    % Update archive
                    [Archive,~,Next] = EnvironmentalSelection([Archive,Offspring1,Offspring2],Problem.N,true);
                    accumulate = accumulate + sum(Next(length(Archive)+1:length(Archive)+length(Offspring1)))/length(Archive);
                    
                    % Select based on learned environment and static environment
                    [Population1,Fitness1]   = EnvironmentalSelectionFe([Population1,Offspring1,Offspring2],Problem.N,F_e);
                    [Population2,Fitness2,~] = EnvironmentalSelection([Population2,Offspring1,Offspring2],Problem.N/2,false);
                    
                    if step >= reward_step
                        % new state
                        % use population1 solutions
                        state_new = dae.reduce(Population1.objs);
                        state_new = state_new';
                        % use improvement on archive
                        reward_f1  = sum(sum(Archive.objs,2))/Problem.N;
                        reward_cv1 = sum(sum(max(0,Archive.cons),2))/length(Archive);
                        f_max      = max(Archive.objs,[],1);
                        f_min      = min(Archive.objs,[],1);
                        reward_d1  = sum(f_max-f_min);
                        
                        y = 1000 * ((reward_f + reward_cv + reward_d1) - (reward_f1 + reward_cv1 + reward_d) + accumulate/reward_step);
                        
                        % Update records and replay buffer
                        current_record    = [state F_e y state_new];
                        Data              = [Data;current_record];
                        current_record_op = [state action_op y state_new];
                        op_Data           = [op_Data;current_record_op];
                        step              = 0;
                    end
                % use DRL model every 5 generations, means keep F_e and action_op unchanged in 5 generations
                else
                    step = step + 1;
                    if step == 1
                        accumulate = 0;
                        
                        % reward
                        reward_f  = sum(sum(Archive.objs,2))/Problem.N;
                        reward_cv = sum(sum(max(0,Archive.cons),2))/length(Archive);
                        f_max     = max(Archive.objs,[],1);
                        f_min     = min(Archive.objs,[],1);
                        reward_d  = sum(f_max-f_min);
                        
                        % state
                        % build the DAE/RBM model to estimate the state
                        dae   = DAE(Problem.M,1,10,Problem.N,0.5,0.5,0.1);
                        dae.train(PopData);
                        state = dae.reduce(Population1.objs);
                        state = state';
                    end
                    if ~model_built
                        if size(Data,1) < 120
                            use_data = 1:size(Data,1);
                        else
                            use_data = randperm(size(Data,1),120);
                        end
                        % build initial Q-net, also named critic net
                        tr_x       = Data(use_data,1:Problem.N+1);
                        tr_y       = Data(use_data,Problem.N+2);
                        critic_net = TrainCritic(tr_x,tr_y);
                        tr_x       = Data(use_data,1:Problem.N);
                        actor_net  = TrainActor(critic_net,tr_x);
                        
                        % build initial operator network
                        if size(Data,1) < 120
                            use_data = 1:size(op_Data,1);
                        else
                            use_data = randperm(size(op_Data,1),120);
                        end
                        tr_x = op_Data(use_data,1:Problem.N+1);
                        tr_y = op_Data(use_data,Problem.N+2);
                        operator_net = TrainDQN(tr_x,tr_y);
                        
                        % determine actions based on DRL
                        test_x = state;
                        % determine F_e
                        action_es = forward(actor_net,dlarray(test_x','CB'));
                        F_e       = double(extractdata(action_es));
                        
                        % determine operator
                        test_x   = state;
                        test_op1 = [test_x,1];
                        test_op1 = test_op1';
                        test_op2 = [test_x,2];
                        test_op2 = test_op2';
                        q_op1    = forward(operator_net,dlarray(test_op1,'CB'));
                        q_op2    = forward(operator_net,dlarray(test_op2,'CB'));
                        if q_op1 > q_op2
                            action_op = 1;
                        elseif q_op1 < q_op2
                            action_op = 2;
                        else
                            action_op = randi(num_op);
                        end
                        model_built = 1;
                        
                        % Generate offspring
                        if action_op == 1
                            % use ordinary DE
                            MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                            MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                            Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                            Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                        else
                            % use GA
                            MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                            MatingPool2 = TournamentSelection(2,Problem.N/2,Fitness2);
                            Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                            Offspring2  = OperatorGA(Problem,Population2(MatingPool2));
                        end
                        
                        % Update archive
                        [Archive,~,Next] = EnvironmentalSelection([Archive,Offspring1,Offspring2],Problem.N,true);
                        accumulate       = accumulate + sum(Next(length(Archive)+1:length(Archive)+length(Offspring1)))/length(Archive);
                        
                        % Select based on learned environment and static environment
                        [Population1,Fitness1]   = EnvironmentalSelectionFe([Population1,Offspring1,Offspring2],Problem.N,F_e);
                        [Population2,Fitness2,~] = EnvironmentalSelection([Population2,Offspring1,Offspring2],Problem.N/2,false);
                    else
                        % Generate offspring
                        if action_op == 1
                            % use ordinary DE
                            MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                            MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                            Offspring1  = OperatorDE(Problem,Population1,Population1(MatingPool1(1:end/2)),Population1(MatingPool1(end/2+1:end)));
                            Offspring2  = OperatorDE(Problem,Population2,Population2(MatingPool2(1:end/2)),Population2(MatingPool2(end/2+1:end)));
                        else
                            % use GA
                            MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                            MatingPool2 = TournamentSelection(2,Problem.N/2,Fitness2);
                            Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                            Offspring2  = OperatorGA(Problem,Population2(MatingPool2));
                        end
                        
                        % Update archive
                        [Archive,~,Next] = EnvironmentalSelection([Archive,Offspring1,Offspring2],Problem.N,true);
                        accumulate       = accumulate + sum(Next(length(Archive)+1:length(Archive)+length(Offspring1)))/length(Archive);
                        
                        % Select based on learned environment and static environment
                        [Population1,Fitness1]   = EnvironmentalSelectionFe([Population1,Offspring1,Offspring2],Problem.N,F_e);
                        [Population2,Fitness2,~] = EnvironmentalSelection([Population2,Offspring1,Offspring2],Problem.N/2,false);
                        
                        if step >= reward_step
                            % Update replay buffer
                            % get new state
                            state_new = dae.reduce(Population1.objs);
                            state_new = state_new';
                            
                            % real reward
                            reward_f1  = sum(sum(Archive.objs,2))/Problem.N;
                            reward_cv1 = sum(sum(max(0,Archive.cons),2))/length(Archive);
                            f_max      = max(Archive.objs,[],1);
                            f_min      = min(Archive.objs,[],1);
                            reward_d1  = sum(f_max-f_min);
                            reward     = 1000 * ((reward_f + reward_cv + reward_d1) - (reward_f1 + reward_cv1 + reward_d) + accumulate/reward_step);
                            
                            % estimated reward of actor-critic
                            test_x_actor   = state_new;
                            target_action  = forward(actor_net,dlarray(test_x_actor','CB'));
                            test_x_critic  = [state_new,double(extractdata(target_action))];
                            target_reward  = forward(critic_net,dlarray(test_x_critic','CB'));
                            target_reward  = double(extractdata(target_reward));
                            y              = reward + gama*target_reward;
                            current_record = [state F_e y state_new];
                            Data           = [Data;current_record];
                            if size(Data,1) > 200
                                Data = Data(2:size(Data,1),:);
                            end
                            
                            % estimated reward of operator-net
                            test_x_op1        = [state_new,action_op];
                            target_reward     = forward(operator_net,dlarray(test_x_op1','CB'));
                            target_reward     = double(extractdata(target_reward));
                            y                 = reward + gama*target_reward;
                            current_record_op = [state action_op y state_new];
                            op_Data           = [op_Data;current_record_op];
                            if size(op_Data,1) > 200
                                op_Data = op_Data(2:size(op_Data,1),:);
                            end
                            
                            % Determine new actions
                            if rand > greedy
                                average_CV = sum(sum(max(0,Population1.cons),2))/Problem.N;
                                F_e        = rand/5 * average_CV;
                                if F_e <= 0
                                    F_e = rand/5 * initial_cv * (1-g/ceil(Problem.maxFE/2*Problem.N));
                                end
                                action_op = randi(num_op);
                            else
                                % determine F_e
                                test_x    = state_new;
                                action_es = forward(actor_net,dlarray(test_x','CB'));
                                F_e       = double(extractdata(action_es));
                                
                                % determine operator
                                test_x   = state_new;
                                test_op1 = [test_x,1];
                                test_op1 = test_op1';
                                test_op2 = [test_x,2];
                                test_op2 = test_op2';
                                q_op1    = forward(operator_net,dlarray(test_op1,'CB'));
                                q_op2    = forward(operator_net,dlarray(test_op2,'CB'));
                                if q_op1 > q_op2
                                    action_op = 1;
                                elseif q_op1 < q_op2
                                    action_op = 2;
                                else
                                    action_op = randi(num_op);
                                end
                            end
                            step = 0;
                        end

                    end
                end
                PopData = [PopData;Population1.objs];
                if size(PopData,1) >= 1000
                    PopData = PopData(101:size(PopData,1),:);
                end
                
                % Update networks every 200 generations
                if model_built && mod(g,200) == 0
                    if size(Data,1) < 120
                        use_data = 1:size(Data,1);
                    else
                        use_data = randperm(size(Data,1),120);
                    end
                    % update Q-net
                    tr_x       = Data(use_data,1:Problem.N+1);
                    tr_y       = Data(use_data,Problem.N+2);
                    critic_net = UpdateCritic(critic_net,tr_x,tr_y);
                    % update policy net
                    tr_x      = Data(use_data,1:Problem.N);
                    actor_net = UpdateActor(actor_net,critic_net,tr_x);
                    
                    % update operator network
                    if size(op_Data,1) <= 120
                        use_data = 1:size(op_Data,1);
                    else
                        use_data = randperm(size(op_Data,1),120);
                    end
                    tr_x         = op_Data(use_data,1:Problem.N+1);
                    tr_y         = op_Data(use_data,Problem.N+2);
                    operator_net = UpdateDQN(operator_net,tr_x,tr_y);
                end
                if Problem.FE >= Problem.maxFE
                    Population1 = Archive;
                end
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon)
% Calculate the fitness of each solution, code of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFitnessFe.m`
```matlab
function [Fitness,R] = CalFitnessFe(PopObj,PopCon,F_e)
% Calculate the fitness of each solution with factor F_e

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    N  = size(PopObj,1);
    CV = sum(max(0,PopCon),2);
    CV(CV<=F_e) = 0;
    
    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `DAE.m`
```matlab
classdef DAE < handle
% Feedforward neural network based denoising autoencoder

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ye Tian

    properties(SetAccess = private)
        nVisible  = 0;
        nHidden   = 0;
        Epoch     = 10;
        BatchSize = 1;
        InputZeroMaskedFraction = 0.5;
        Momentum  = 0.5;
        LearnRate = 0.1;
        WA        = [];
        WB        = [];
        lower     = [];
        upper     = [];
    end
    methods
        %% Constructor
        function obj = DAE(nVisible,nHidden,Epoch,BatchSize,InputZeroMaskedFraction,Momentum,LearnRate)
            obj.nVisible  = nVisible;
            obj.nHidden   = nHidden;
            obj.Epoch     = Epoch;
            obj.BatchSize = BatchSize;
            obj.InputZeroMaskedFraction = InputZeroMaskedFraction;
            obj.Momentum  = Momentum;
            obj.LearnRate = LearnRate; 
            obj.WA = (rand(nHidden,nVisible+1)-0.5)*8*sqrt(6/(nHidden+nVisible));   
            obj.WB = (rand(nVisible,nHidden+1)-0.5)*8*sqrt(6/(nVisible+nHidden));  
        end
        
        %% Train
        function train(obj,X)
            obj.lower = min(X,[],1);
            obj.upper = max(X,[],1);
            X = (X-repmat(obj.lower,size(X,1),1))./repmat(obj.upper-obj.lower,size(X,1),1);% normalization
            vW{1} = zeros(size(obj.WA));
            vW{2} = zeros(size(obj.WB));
            if(obj.InputZeroMaskedFraction ~= 0)
                theta = rand(size(X)) > obj.InputZeroMaskedFraction;
            else
                theta = true(size(X));
            end
            X_temp = X.*theta;% produce noise
            X_temp = [ones(size(X,1),1),X_temp];
            for i = 1 : obj.Epoch
                kk = randperm(size(X,1));
                for batch = 1 : size(X,1)/obj.BatchSize
                    batch_x = X_temp(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);
                    batch_y = X(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);

                    % Feedforward pass
                    poshid1 = 1./(1+exp(-batch_x*obj.WA'));
                    poshid1 = [ones(obj.BatchSize,1),poshid1];
                    poshid2 = 1./(1+exp(-poshid1*obj.WB'));

                    % BP
                    e     = batch_y - poshid2;
                    d{3}  = -e.*(poshid2.*(1-poshid2));
                    d_act = poshid1.*(1-poshid1);
                    d{2}  = d{3}*obj.WB.*d_act;
                    for i = 1 : 2
                        if i+1 == 3
                            dW{i} = (d{i+1}'*poshid1/size(d{3},1));
                        else
                            dW{i} = (d{i+1}(:,2:end)'*batch_x)/size(d{i+1},1);
                        end
                    end
                    for i = 1 : 2
                        dW{i} = obj.LearnRate*dW{i};
                        if obj.Momentum > 0
                            vW{i} = obj.Momentum*vW{i} + dW{i};
                            dW{i} = vW{i};
                        end
                        if i == 1
                            obj.WA = obj.WA - dW{i};
                        else
                            obj.WB = obj.WB - dW{i};
                        end
                    end
                end
            end
        end

        %% Reduce
        function H = reduce(obj,X)
            X = (X-repmat(obj.lower,size(X,1),1))./repmat(obj.upper-obj.lower,size(X,1),1);
            H = 1./(1+exp(-X*obj.WA(:,2:end)'-repmat(obj.WA(:,1)',size(X,1),1)));
        end

        %% Recover
        function X = recover(obj,H)
            X = 1./(1+exp(-H*obj.WB(:,2:end)'-repmat(obj.WB(:,1)',size(H,1),1)));
            X = X.*repmat(obj.upper-obj.lower,size(X,1),1) + repmat(obj.lower,size(X,1),1);
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness,Next] = EnvironmentalSelection(Population,N,isOrigin)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    if isOrigin
        Fitness = CalFitness(Population.objs,Population.cons);
    else
        Fitness = CalFitness(Population.objs);
    end

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `EnvironmentalSelectionFe.m`
```matlab
function [Population,Fitness,R] = EnvironmentalSelectionFe(Population,N,F_e)
% The environmental selection with F_e

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    %% Calculate the fitness of each solution
    [Fitness,R] = CalFitnessFe(Population.objs,Population.cons,F_e);
    
    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```
