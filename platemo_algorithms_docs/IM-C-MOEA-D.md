# IM-C-MOEA-D

**Tags**: <2024> <multi> <real/integer> <large/none> <constrained/none>

## Description
Inverse modeling constrained MOEA/D

## Reference
L. R. C. Farias and A. F. R. Araujo. An inverse modeling constrained multi-objective evolutionary algorithm based on decomposition. Proceedings of the IEEE International Conference on Systems, Mans and Cybernetics, 2024.

## Source Code

### `IMCMOEAD.m`
```matlab
classdef IMCMOEAD < ALGORITHM
% <2024> <multi> <real/integer> <large/none> <constrained/none>
% Inverse modeling constrained MOEA/D
% K --- 10 --- Number of clusters

%------------------------------- Reference --------------------------------
% L. R. C. Farias and A. F. R. Araujo. An inverse modeling constrained
% multi-objective evolutionary algorithm based on decomposition.
% Proceedings of the IEEE International Conference on Systems, Mans and
% Cybernetics, 2024.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias (email: lucas.farias@unicap.br)

    methods
		function main(Algorithm,Problem)
			%% Parameter setting
			K = Algorithm.ParameterSet(10); % Number of clusters
            T = ceil(Problem.N/10); % Size of neighborhood
            
			%% Generate weight vectors
			[W,Problem.N] = UniformPoint(Problem.N,Problem.M);
			
			%% Detect the neighbours of each solution
			B = pdist2(W,W);
			[~,B] = sort(B,2);
			B = B(:,1:T);
			
			%% Generate random population
			Population = Problem.Initialization();			
			idealPoint = min(Population.objs,[],1); % Initial ideal point

			%% Optimization loop
			while Algorithm.NotTerminated(Population)
				% Clustering (k-means)
				[partition,~] = kmeans(Population.objs,K); 
				
				% Modeling and reproduction per cluster
                Offsprings=[];
				for k = unique(partition)'
					parents = Population(partition==k);
                    MatingPool = TournamentSelection(2,size(parents,2),sum(max(0,parents.cons),2));
					Offspring  = Operator(Problem,parents(MatingPool));
					Offsprings = [Offsprings,Offspring];
				end
				
				% Apply constraint handling to Population and Offsprings
                PopObj = applyConstraintHandling(Population, Problem.M);
                OffObj = applyConstraintHandling(Offsprings, Problem.M);
				
				% Update the ideal and nadir points
		        idealPoint = min([idealPoint;OffObj],[],1);
                nadirPoint = max([PopObj;OffObj],[],1);
				
				% Normalize objectives
                PopObj = (PopObj-repmat(idealPoint,size(PopObj,1),1))./repmat(nadirPoint-idealPoint,size(PopObj,1),1);
                OffObj = (OffObj-repmat(idealPoint,size(OffObj,1),1))./repmat(nadirPoint-idealPoint,size(OffObj,1),1);
						
				% Performs global replacement based on the Tchebycheff approach
				Population = globalReplacement(Population, Offsprings, W, B, T, PopObj, OffObj);
			end
		end
	end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Population,L)
% The Gaussian process based reproduction

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://www.soft-computing.de/jin-pub_year.html

    %% Parameter setting
    if nargin < 3
        L = 3;
    end
    PopDec = Population.decs;
	PopObj = Population.objs;
    [N,D]  = size(PopDec);
    if D < 3
       L = D; 
    end
    %% Gaussian process based reproduction
    if length(Population) < 2*Problem.M
        OffDec = PopDec;
    else
        OffDec = [];
        fmin   = 1.5*min(PopObj,[],1) - 0.5*max(PopObj,[],1);
        fmax   = 1.5*max(PopObj,[],1) - 0.5*min(PopObj,[],1);
        % Train one groups of GP models for each objective
        for m = 1 : Problem.M
            parents = randperm(N,floor(N/Problem.M));
            offDec  = PopDec(parents,:);
            for d = randperm(D,L)
                % Gaussian Process
                try
                    [ymu,ys2] = gp(struct('mean',[],'cov',[],'lik',log(0.01)),...
                                   @infExact,@meanZero,@covLIN,@likGauss,...
                                   PopObj(parents,m),PopDec(parents,d),...
                                   linspace(fmin(m),fmax(m),size(offDec,1))');
                    offDec(:,d) = ymu + rand*sqrt(ys2).*randn(size(ys2));
                catch
                end
            end
            OffDec = [OffDec;offDec];
        end
    end
    
    %% Convert invalid values to random values
    [N,D]   = size(OffDec);
    Lower   = repmat(Problem.lower,N,1);
    Upper   = repmat(Problem.upper,N,1);
    randDec = unifrnd(Lower,Upper);
    invalid = OffDec<Lower | OffDec>Upper;
    OffDec(invalid) = randDec(invalid);

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Site   = rand(N,D) < proM/D;
    mu     = rand(N,D);
    temp   = Site & mu<=0.5;
    OffDec = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec);
end
```

### `applyConstraintHandling.m`
```matlab
function PopObj = applyConstraintHandling(Population,M)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias (email: lucas.farias@unicap.br)

	% Handles constraint violations by penalizing infeasible solutions
	PopObj     = Population.objs;
	PopCon     = max(0, Population.cons);
	Infeasible = any(PopCon > 0, 2);
	PopObj(Infeasible,:) = repmat(max(PopObj,[],1),sum(Infeasible),1) + repmat(sum(max(0,PopCon(Infeasible,:)),2),1,M);
end
```

### `globalReplacement.m`
```matlab
function Population = globalReplacement(Population, Offsprings, W, B, T,  PopObj, OffObj)
% Global replacement

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lucas Farias (email: lucas.farias@unicap.br)

	for i = 1 : length(Offsprings)
		% Calculate the Tchebycheff function value for each weight vector (W) using the offspring's objective values (OffObj)
		tchebycheff_values = max(OffObj(i,:).*W,[],2);		
		
		% Identify the minimum Tchebycheff value, which corresponds to the best solution according to the decomposition method
		best_tchebycheff_value = min(tchebycheff_values);		
		
		% Find the index of the weight vector that corresponds to the best Tchebycheff value
		best_weight_index  = find(tchebycheff_values (:,1) == best_tchebycheff_value);	
		
		% Randomly select a subset of neighborhood solutions from the neighbors of the chosen solution
		P = B(best_weight_index (1),randperm(size(B,2)));					
		
		% Calculate the constraint violation of offspring and P neighborhood
		CVO = sum(max(0,Offsprings(i).con));
		CVP = sum(max(0,Population(P).cons),2);
		
		% Update the solutions in P by Tchebycheff approach
		g_old = max(PopObj(P,:).*W(P,:),[],2);
		g_new = max(repmat(OffObj(i,:),length(P),1).*W(P,:),[],2);
		
		% Replace solutions
		Population(P(find(g_old>=g_new & CVP>=CVO,T))) = Offsprings(i);
	end
end
```
