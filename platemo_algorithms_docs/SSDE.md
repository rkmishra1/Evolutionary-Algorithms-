# SSDE

**Tags**: <2024> <multi/many> <real/integer> <constrained/none> <expensive>

## Description
Self-organized surrogate-assisted differential evolution

## Reference
A. F. R. Araújo, L. R. C. Farias, and A. R. C. Gonçalves. Self-organizing surrogate-assisted non-dominated sorting differential evolution. Swarm and Evolutionary Computation, 2024, 91: 101703.

## Source Code

### `Operator.m`
```matlab
function [Population, Samples, winning_weights] = Operator(Problem, Population, W, winning_weights)    

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias (email: lrcf@cin.ufpe.br)

    % Generate all candidate offsprings
    MatingPool   = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2))';
    CandidateDec = Population.decs;
    OffspringDec = OperatorDE(Problem,CandidateDec(MatingPool,:), CandidateDec(randi(Problem.N,1,Problem.N),:) , CandidateDec(randi(Problem.N,1,Problem.N),:));    

	% Normalize offspring for SOM mapping
	Normalized_OffspringDec = rescale(OffspringDec,'InputMin',Problem.lower,'InputMax',Problem.upper);
	
	% Distance between each solution to a neuron
    Distance = pdist2(Normalized_OffspringDec,W(:,1:Problem.D)); 
    [~,rank] = sort(Distance,2);    
    
	% Estimate the offspring objective values using SOM
	Offspring_Labels = W(rank(:,1),Problem.D+1:Problem.D+Problem.M);

	%% Assign one to winning nodes
    winning_weights(rank(:,1)) = true;	
	
	%% Selection of survivors by ranking
	objs = [Population.objs;Offspring_Labels];
	cons = [Population.cons;zeros(size(Offspring_Labels,1),size(Population.cons,2))];
    
	%% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(objs,cons,Problem.N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:Problem.N-sum(Next)))) = true;
    
    %% Population for next generation
    out = Next(1:Problem.N);
    in  = Next(Problem.N+1:end);

	%% Selection
    if sum(in) >= 1
        Offspring = Problem.Evaluation(OffspringDec(in,:));
        Samples   = Offspring;
        Population(~out) = Offspring;
    else
        Samples = Population;
    end
end
```

### `SSDE.m`
```matlab
classdef SSDE < ALGORITHM
% <2024> <multi/many> <real/integer> <constrained/none> <expensive>
% Self-organized surrogate-assisted differential evolution
% num_nodes ---     --- Number of neurons in each dimension of the latent space
% eta0      --- 0.2 --- Initial learning rate
% sigma0    ---     --- Size of neighborhood mating pools

%------------------------------- Reference --------------------------------
% A. F. R. Araújo, L. R. C. Farias, and A. R. C. Gonçalves. Self-organizing
% surrogate-assisted non-dominated sorting differential evolution. Swarm
% and Evolutionary Computation, 2024, 91: 101703.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias (email: lrcf@cin.ufpe.br)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [num_nodes,eta0,sigma0] = Algorithm.ParameterSet(Problem.N, 0.2, Problem.N);

            %% Generate random population
			Population = Problem.Initialization();

			%% Initialize the sample set
			Samples = Population;
			
			%% Initialize the SOM weight vectors
            a = 0.001;
            b = 0.5;
            W = a.*randn(num_nodes,Problem.D+Problem.M) + b;
			winning_weights = false(1,Problem.N);			
			
			%% Position of each neuron and neighborhood
			D = arrayfun(@(S)1:S,floor(num_nodes),'UniformOutput',false);
			eval(sprintf('[%s]=ndgrid(D{:});',sprintf('c%d,',1:length(D))))
			eval(sprintf('V=[%s];',sprintf('c%d(:),',1:length(D))))			
			LDis = pdist2(V,V); % Distance between each two neurons in latent space
			
            %% Optimization
            while Algorithm.NotTerminated(Population)
				%% Training
                if size(Samples,2) >= Problem.N
                    W = Training(Problem, W, LDis, Samples, num_nodes, eta0, sigma0, winning_weights);
                    winning_weights = false(1,num_nodes);
					Samples = [];
                end
                
                %% Evolutionary process
                [Population, new_samples, winning_weights] = Operator(Problem, Population, W, winning_weights);
				Samples = [Samples, new_samples];
            end
        end
    end
end
```

### `Training.m`
```matlab
function W = Training(Problem, W, LDis, Samples, num_nodes, eta0, sigma0, winning_weights)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias (email: lrcf@cin.ufpe.br)

    %% Memory-based reinitialization
	if any(winning_weights) == true                    
        % If there is a node that did not win
        if sum(winning_weights) < num_nodes   
            % Select indexes based on diversity criteria
            [FrontNo,~] = NDSort(W(winning_weights,Problem.D+1:end),sum(winning_weights));

			% Calculate the crowding distance of each solution
            CrowdDis   = CrowdingDistance(W(winning_weights,Problem.D+1:end),FrontNo); 
            [~,p]      = sort(CrowdDis);
            factor     = 1 : length(CrowdDis);
            factor(p)  = factor; % each winning node receives a weight according to the diversity criterion
            chosen_idx = randsample(sum(winning_weights),2*(num_nodes-sum(winning_weights)),true,factor);
            chosen_idx = reshape(chosen_idx,num_nodes-sum(winning_weights),2);

            % linear combination: calculate average and add noise to new nodes
            W(~winning_weights,:) = ((W(chosen_idx(:,1),:) + W(chosen_idx(:,2),:) ) ./ 2) + 0.001.*randn(num_nodes-sum(winning_weights),Problem.D+Problem.M);

            % repair if it exceeds the upper limit, 1, or lower limit, 0
            W(:,1:Problem.D) = max(min(W(:,1:Problem.D),1),0);
        end
	end
		
	%% Normalize Sample Set
	Samples = [Samples.decs,Samples.objs];
    Samples(:,1:Problem.D) = rescale(Samples(:,1:Problem.D),'InputMin',Problem.lower,'InputMax',Problem.upper);

    %% reset parameters
	win_count_set = zeros(1,num_nodes);

    %% SOM surrogate model training
	for epoch = 1 : 50
		% shuffle the samples
		randpos = randperm(size(Samples,1));        
		% Neighborhood radius 
		sigma = sigma0*exp((-win_count_set/size(Samples,1))); 
		% Learning rate		
		eta = eta0*exp((-win_count_set/size(Samples,1))); 
		
        for i = 1 : size(Samples,1) 
			s = randpos(i);
            
            % define winning node, u1, from input sample, s, using quadratic Euclidean distance
			[~,u1] = min(pdist2(Samples(s,1:end-Problem.M),W(:,1:end-Problem.M)));

			% If it is the node's first win, assign the sample to the node
            if win_count_set(u1) == 0
				W(u1,:) = Samples(s,:);
            end
            
            % limit node win counter to current epoch
            if win_count_set(u1) < epoch
                win_count_set(u1) = win_count_set(u1) + 1;
            end
            
            % update winning node and its neighborhood
			U      = LDis(u1,:) < sigma;
			W(U,:) = W(U,:) + repmat(eta(U)',1,size(W,2)).*repmat(exp(-LDis(u1,U))',1,size(W,2)).*(repmat(Samples(s,:),sum(U),1)-W(U,:));
        end
	end
end
```
