# CMOEA-AOP

**Tags**: <2026> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Automated operator portfolio based constrained MOEA

## Reference
S. Shao, Y. Tian, S. Yang, and X. Zhang. Deep reinforcement learning- assisted automated operator portfolio for constrained multi-objective optimization. IEEE Transactions on Emerging Topics in Computational Intelligence, 2026.

## Source Code

### `CMOEAAOP.m`
```matlab
classdef CMOEAAOP < ALGORITHM
% <2026> <multi> <real/integer/label/binary/permutation> <constrained>
% Automated operator portfolio based constrained MOEA

%------------------------------- Reference --------------------------------
% S. Shao, Y. Tian, S. Yang, and X. Zhang. Deep reinforcement learning-
% assisted automated operator portfolio for constrained multi-objective
% optimization. IEEE Transactions on Emerging Topics in Computational
% Intelligence, 2026.
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
            Population{1}  = Problem.Initialization();
            Population{2}  = Problem.Initialization();
            Fitness{1}     = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{2}     = CalFitness(Population{2}.objs);
            transfer_state = 0;
            cnt            = 0;

            state_dim    = Problem.M*2 + 2;
            action_dim   = 3;
            action_bound = 5;
            minnumber    = 1e-5;
            
            state1 = GenerateSample(Problem,rand(1,action_dim),Population{1},Population{1});
            agent  = DDPG(state_dim,action_dim,action_bound);

            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                action = agent.selectAction(state1,true);
                action = (action+action_bound) / sum(action+action_bound+minnumber);
                cnt    = cnt + 1;                
                LastPopulation1 = Population{1};
                if transfer_state == 0
                    for i = 1 : 2
                        valOffspring{i} = OperatorConstrainedAOP(Problem,Population,randperm(Problem.N,Problem.N),action,i);
                    end
                    for i = 1 : 2
                        if i == 1
                            [Population{i},Fitness{i},~] = EnvironmentalSelection([Population{1},valOffspring{1},valOffspring{2}],Problem.N,i);
                        else
                            [Population{i},Fitness{i},~] = EnvironmentalSelection([Population{2},valOffspring{2},valOffspring{1}],Problem.N,i);
                        end
                    end
                    if Problem.FE/Problem.maxFE >= 0.2
                        transfer_state = 1;
                    end
                else                    
                    for i = 1 : 2
                        MatingPool      = TournamentSelection(2,Problem.N,Fitness{i});                        
                        valOffspring{i} = OperatorConstrainedAOP(Problem,Population,MatingPool,action,i);
                    end
                    [~,~,Next]       = EnvironmentalSelection([Population{2},valOffspring{2}],Problem.N,1);
                    succ_rate(1,cnt) = (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
                    
                    [~,~,Next]       = EnvironmentalSelection([Population{1},valOffspring{1}],Problem.N,2);
                    succ_rate(2,cnt) = (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
                    
                    for i = 1 : 2
                        if succ_rate(i,cnt) > 0
                            rand_number = randperm(Problem.N);
                            [Population{i},Fitness{i},~] = EnvironmentalSelection([Population{i},valOffspring{i},Population{2/i}(rand_number(1:Problem.N/2))],Problem.N,i);
                        else
                            [Population{i},Fitness{i},~] = EnvironmentalSelection([Population{i},valOffspring{i},valOffspring{2/i}],Problem.N,i);
                        end
                    end
                end
                [state1,action1,nextstate1,ter,reward1] = GenerateSample(Problem,action,LastPopulation1,Population{1});
                agent.store(state1,action1,reward1,nextstate1,ter);
                agent.train();
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon)
% Calculate the fitness of each solution

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

### `DDPG.m`
```matlab
classdef DDPG < handle

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        state_dim;
        action_dim;
        action_bound;
        opts;
        
        actor;
        actor_target;
        critic;
        critic_target;
        
        ou_theta;
        ou_sigma;
        ou_mu;
        ou_state;

        avgGradCritic;
        avgSqCritic;
        avgGradActor;
        avgSqActor;
        adamIter;

        buffer_capacity;
        buffer_ptr;
        buffer_size_count;
        buf_states;
        buf_actions;
        buf_rewards;
        buf_next_states;
        buf_dones;
    end
    
    methods
        function obj = DDPG(state_dim, action_dim, action_bound, optOverrides)
            if nargin < 4
                optOverrides = struct();
            end
            obj.state_dim    = state_dim;
            obj.action_dim   = action_dim;
            obj.action_bound = action_bound;
            obj.opts         = DDPG.mergeOptions(DDPG.defaultOptions(), optOverrides);
            
            buffer_cap   = obj.opts.buffer_size;
            hidden_sizes = obj.opts.hidden_sizes;
            
            obj.ou_state = zeros(1, action_dim);
            
            obj.avgGradCritic = [];
            obj.avgSqCritic   = [];
            obj.avgGradActor  = [];
            obj.avgSqActor    = [];
            obj.adamIter      = 0;
            
            obj.buffer_capacity   = buffer_cap;
            obj.buffer_ptr        = 1;
            obj.buffer_size_count = 0;
            obj.buf_states        = zeros(buffer_cap, state_dim);
            obj.buf_actions       = zeros(buffer_cap, action_dim);
            obj.buf_rewards       = zeros(buffer_cap, 1);
            obj.buf_next_states   = zeros(buffer_cap, state_dim);
            obj.buf_dones         = zeros(buffer_cap, 1);
            
            obj.actor = DDPG.buildActorDlnetwork(state_dim, action_dim, hidden_sizes);
            obj.actor_target = DDPG.buildActorDlnetwork(state_dim, action_dim, hidden_sizes);
            DDPG.copyLearnables(obj.actor_target, obj.actor);
            
            obj.critic = DDPG.buildCriticDlnetwork(state_dim, action_dim, hidden_sizes);
            obj.critic_target = DDPG.buildCriticDlnetwork(state_dim, action_dim, hidden_sizes);
            DDPG.copyLearnables(obj.critic_target, obj.critic);
        end
        
        function action = selectAction(obj, state, explore)
            if nargin < 3
                explore = true;
            end
            state  = single(reshape(state, [], 1));
            s_dl   = dlarray(state, 'CB');
            a_raw  = forward(obj.actor, s_dl);
            action = extractdata(a_raw)' * obj.action_bound;
            if explore
                ot = obj.opts.ou_theta;
                os = obj.opts.ou_sigma;
                om = obj.opts.ou_mu;
                obj.ou_state = obj.ou_state + ot * (om - obj.ou_state) + os * randn(1, obj.action_dim);
                noise  = obj.ou_state * obj.opts.noise_scale;
                action = action + noise;
                ab     = obj.action_bound;
                action = max(-ab, min(ab, action));
            end
        end
        
        function store(obj, state, action, reward, next_state, done)
            state      = reshape(state, 1, []);
            action     = reshape(action, 1, []);
            next_state = reshape(next_state, 1, []);
            p = obj.buffer_ptr;
            obj.buf_states(p, :)  = state;
            obj.buf_actions(p, :) = action;
            obj.buf_rewards(p)    = reward;
            obj.buf_next_states(p, :) = next_state;
            obj.buf_dones(p) = done;
            obj.buffer_ptr   = mod(p, obj.buffer_capacity) + 1;
            obj.buffer_size_count = min(obj.buffer_size_count + 1, obj.buffer_capacity);
        end
        
        function [critic_loss, actor_loss] = train(obj)
            critic_loss = 0;
            actor_loss  = 0;
            bs          = obj.opts.batch_size;
            if obj.buffer_size_count < bs
                return;
            end
            n           = min(bs, obj.buffer_size_count);
            idx         = randi(obj.buffer_size_count, [n, 1]);
            states      = obj.buf_states(idx, :);
            actions     = obj.buf_actions(idx, :);
            rewards     = obj.buf_rewards(idx);
            next_states = obj.buf_next_states(idx, :);
            dones       = obj.buf_dones(idx);
            
            s_next      = single(next_states');
            s_dl_next   = dlarray(s_next, 'CB');
            a_next_tanh = forward(obj.actor_target, s_dl_next);
            a_next      = extractdata(a_next_tanh) * obj.action_bound;
            sa_next     = [s_next; a_next];
            q_tgt_num   = extractdata(forward(obj.critic_target, dlarray(sa_next, 'CB')));
            y           = single(rewards + obj.opts.gamma .* (1 - dones) .* q_tgt_num(:));
            
            sa    = single([states'; actions']);
            y_row = reshape(y, 1, []);
            obj.adamIter = obj.adamIter + 1;
            [critic_loss, gradC] = dlfeval(@ddpgCriticGradients, obj.critic, dlarray(sa, 'CB'), dlarray(y_row, 'CB'));
            [obj.critic, obj.avgGradCritic, obj.avgSqCritic] = adamupdate(obj.critic, gradC, obj.avgGradCritic, obj.avgSqCritic, obj.adamIter, obj.opts.lr_critic);
            
            s_tr = single(states');
            [actor_loss, gradA] = dlfeval(@ddpgActorGradients, obj.actor, obj.critic, dlarray(s_tr, 'CB'), obj.action_bound);
            [obj.actor, obj.avgGradActor, obj.avgSqActor] = adamupdate(obj.actor, gradA, obj.avgGradActor, obj.avgSqActor, obj.adamIter, obj.opts.lr_actor);
            
            DDPG.softUpdateLearnables(obj.actor_target, obj.actor, obj.opts.tau);
            DDPG.softUpdateLearnables(obj.critic_target, obj.critic, obj.opts.tau);
            
            o             = obj.opts;
            o.noise_scale = o.noise_scale * o.noise_decay;
            obj.opts      = o;
        end
        
        function resetNoise(obj)
            obj.ou_state = zeros(1, obj.action_dim);
        end
    end
    
    methods (Static)
        function o = defaultOptions()         
            o              = struct();
            o.gamma        = 0.99;
            o.tau          = 0.005;
            o.lr_actor     = 1e-4;
            o.lr_critic    = 1e-3;
            o.batch_size   = 32;
            o.buffer_size  = 500;
            o.hidden_sizes = [32, 32];
            o.noise_scale  = 0.3;
            o.noise_decay  = 0.9995;
            o.ou_theta     = 0.15;
            o.ou_sigma     = 0.2;
            o.ou_mu        = 0;
        end
    end
    
    methods (Static, Access = private)
        function out = mergeOptions(base, override)
            out = base;
            if nargin < 2 || isempty(override)
                return;
            end
            fn = fieldnames(override);
            for i = 1 : numel(fn)
                f = fn{i};
                if isfield(base, f)
                    out.(f) = override.(f);
                end
            end
        end
        function net = buildActorDlnetwork(state_dim, action_dim, hidden_sizes)
            layers = featureInputLayer(state_dim, 'Normalization', 'none', 'Name', 'in');
            for k = 1 : numel(hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(hidden_sizes(k), 'Name', sprintf('a_fc%d', k))
                    reluLayer('Name', sprintf('a_relu%d', k))];
            end
            layers = [layers
                fullyConnectedLayer(action_dim, 'Name', 'a_out_fc')
                tanhLayer('Name', 'a_out_tanh')];
            lg  = layerGraph(layers);
            net = dlnetwork(lg);
        end
        
        function net = buildCriticDlnetwork(state_dim, action_dim, hidden_sizes)
            in_dim = state_dim + action_dim;
            layers = featureInputLayer(in_dim, 'Normalization', 'none', 'Name', 'cin');
            for k = 1 : numel(hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(hidden_sizes(k), 'Name', sprintf('c_fc%d', k))
                    reluLayer('Name', sprintf('c_relu%d', k))];
            end
            layers = [layers
                fullyConnectedLayer(1, 'Name', 'c_q')];
            lg  = layerGraph(layers);
            net = dlnetwork(lg);
        end
        
        function copyLearnables(dstNet, srcNet)
            Ld = dstNet.Learnables;
            Ls = srcNet.Learnables;
            for i = 1 : height(Ld)
                Ld.Value{i} = dlarray(extractdata(Ls.Value{i}));
            end
            dstNet.Learnables = Ld;
        end
        
        function softUpdateLearnables(targetNet, onlineNet, tau)
            Lt = targetNet.Learnables;
            Lo = onlineNet.Learnables;
            for i = 1 : height(Lt)
                Lt.Value{i} = tau * Lo.Value{i} + (1 - tau) * Lt.Value{i};
            end
            targetNet.Learnables = Lt;
        end
    end
end

function [loss, gradients] = ddpgCriticGradients(critic, sa_dl, y_dl)
    q         = forward(critic, sa_dl);
    loss      = mean((q - y_dl).^2, 'all');
    gradients = dlgradient(loss, critic.Learnables);
end

function [loss, gradients] = ddpgActorGradients(actor, critic, s_dl, action_bound)
    a_tanh    = forward(actor, s_dl);
    a         = a_tanh * action_bound;
    sa        = cat(1, s_dl, a);
    q         = forward(critic, sa);
    loss      = -mean(q, 'all');
    gradients = dlgradient(loss, actor.Learnables);
end
```

### `DEBest.m`
```matlab
function [Offspring] = DEBest(Problem,Population,ProblemN)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------     

    FrontNo = NDSort(Population.objs,Population.cons,1);   
    index1  = find(FrontNo==1);
    r       = floor(rand*length(index1))+1;
    best    = index1(r);

    [N,D] = size(Population(1).decs);       
    trial = zeros(1*ProblemN,D);
       
    for i = 1 : ProblemN          
        l = rand;
        if l <= 1/3 
        	F = 0.6;
        elseif l <= 2/3
            F = 0.8;
        else
            F = 1.0;
        end   
        l = rand;
        if l <= 1/3
            CR = 0.1;
        elseif l <= 2/3
            CR = 0.2;
        else
            CR = 1.0;
        end
        indexset    = 1 : ProblemN;
        indexset(i) = [];
        r1  = floor(rand*(ProblemN-1))+1;
        xr1 = indexset(r1);
        indexset(r1) = [];
        r2  = floor(rand*(ProblemN-2))+1;
        xr2 = indexset(r2)  ;
        r3  = floor(rand*(ProblemN-3))+1;
        xr3 = indexset(r3);
        Best_index = Population(best).decs;
        v      = Population(xr1).decs+rand*(Best_index-Population(xr1).decs)+F*(Population(xr2).decs-Population(xr3).decs);  
        Lower  = repmat(Problem.lower,N,1);
        Upper  = repmat(Problem.upper,N,1);
        v      = min(max(v,Lower),Upper);
        Site   = rand(N,D) < CR;
        j_rand = floor(rand * D) + 1;
        Site(1, j_rand) = 1;
        Site_  = 1-Site;
        trial(i, :) = Site.*v+Site_.*Population(i).decs;         
        
    end
    Offspring = trial;
    Offspring = Problem.Evaluation(Offspring);
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
    if isOrigin==1
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
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population = Population(rank);
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

### `GenerateSample.m`
```matlab
function [state1, action1, nextstate1, ter, reward1] = GenerateSample(Problem, action,LastPopulation,Population)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    LastPopulationHV = HV(LastPopulation,max([LastPopulation.objs;Population.objs]));
    PopulationHV     = HV(Population,max([LastPopulation.objs;Population.objs]));
    reward           = (PopulationHV - LastPopulationHV)/LastPopulationHV;
    if isnan(reward)
        reward = 0;
    end

    LastObjsVar = var(LastPopulation.objs);
    LastObjsCon = sum(LastPopulation.objs);
    LastCV      = sum(max(Population.cons,0),'all');
    LastRatio   = Problem.FE/Problem.maxFE;
    LastState   = [LastObjsVar, LastObjsCon, LastCV, LastRatio];

    CurrentObjsVar = var(Population.objs);
    CurrentObjsCon = sum(Population.objs);
    CurrentCV      = sum(max(Population.cons,0),'all');
    CurrentRatio   = Problem.FE/Problem.maxFE;
    CurrentState   = [CurrentObjsVar, CurrentObjsCon, CurrentCV, CurrentRatio];

    state1     = LastState;
    action1    = action;
    nextstate1 = CurrentState;
    ter        = 0;
    reward1    = reward;
end
```

### `OperatorConstrainedAOP.m`
```matlab
function Offspring = OperatorConstrainedAOP(Problem, Population, MatingPool, action, i)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    rates    = action;
    ratesNum = floor(rates.*Problem.N);
    for item = 1 : length(action)
        if ratesNum(item) < 4
            ratesNum(item) = 4; % Set the minimum value to 4 so that each operator can be called normally
        end
    end
    MatingPool = repmat(MatingPool,1,20);
    Offspring1 = OperatorGAhalf(Problem,Population{i}(MatingPool(1:ratesNum(1)*2)));
    Offspring2 = OperatorDE(Problem,Population{i}(randi(ceil(Problem.N),1,ratesNum(2))),Population{i}(randi(ceil(Problem.N),1,ratesNum(2))),Population{i}(randi(ceil(Problem.N),1,ratesNum(2))));
    p3         = Population{i};
    Offspring3 = DEBest(Problem, p3(randi(ceil(Problem.N),1,ratesNum(3))),ratesNum(3));
    Offspring  = [Offspring1 Offspring2 Offspring3];
end
```
