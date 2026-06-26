# LERD

**Tags**: <2024> <multi/many> <real> <large/none>

## Description
Large-scale evolutionary algorithm with reformulated decision variable analysis

## Reference
C. He, R. Cheng, L. Li, K. C. Tan, and Y. Jin. Large-scale multiobjective optimization via reformulated decision variable analysis. IEEE Transactions on Evolutionary Computation, 2024, 28(1): 47-61.

## Source Code

### `EnvironmentSelection.m`
```matlab
function Population = EnvironmentSelection(Population,N)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    %% Non-dominated sorting
    PopObj = Population.objs;
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next   = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
end
```

### `EvolveByMOEAD.m`
```matlab
function Population = EvolveByMOEAD(Problem,Population,W,deltaG)
% Uniformity optimization by MOEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Detect the neighbours of each solution
    W = W.*repmat(max(Population.objs,[],1)-min(Population.objs,[],1),size(W,1),1);
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:ceil(Problem.N/10));
    
    %% Associate each subproblem with one solution
    % The ideal point
    Z = min(Population.objs,[],1);
    % The value of each solution on each subproblem (modified Tchebycheff approach)
    g = zeros(Problem.N);
    for i = 1 : Problem.N
        g(i,:) = max(repmat(abs(Population(i).obj-Z),Problem.N,1)./W,[],2)';
    end
    [~,rank] = sort(g,2);
    % The index of solution which each subproblem associated with
    associate = zeros(1,Problem.N);
    for i = 1 : Problem.N
        x = find(~associate(rank(i,:)),1);
        associate(rank(i,x)) = i;
    end
    Population = Population(associate);
    
    %% Optimization
    for k = 1 : deltaG
        % For each solution
        for i = 1 : Problem.N
            % Choose the parents
            if rand < 0.9
                P = B(i,randperm(size(B,2)));
            else
                P = randperm(Problem.N);
            end

            % Generate an offspring
            Offspring = OperatorDE(Problem,Population(i),Population(P(1)),Population(P(2)));
            % Update the ideal point
            Z = min(Z,Offspring.obj);
            % Update the solutions in P by modified Tchebycheff approach
            g_old = max(abs(Population(P).objs-repmat(Z,length(P),1))./W(P,:),[],2);
            g_new = max(repmat(abs(Offspring.obj-Z),length(P),1)./W(P,:),[],2);
            Population(P(g_old>=g_new)) = Offspring;
        end
    end
end
```

### `FitnessSelection.m`
```matlab
function [Next,FrontNo,CrowdDis] = FitnessSelection(Fitness,N)
% The environmental selection of NSGA-II for seleting solutions based on
% fitness values

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Fitness,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Fitness,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    FrontNo  = FrontNo(Next);
    CrowdDis = CrowdDis(Next);
end
```

### `LERD.m`
```matlab
classdef LERD < ALGORITHM
% <2024> <multi/many> <real> <large/none>
% Large-scale evolutionary algorithm with reformulated decision variable analysis
% sN  ---  3 --- Number of sampled points
% N   --- 10 --- Population size in DVA optimization
% gen --- 20 --- Number of iterations for DVA optimization

%------------------------------- Reference --------------------------------
% C. He, R. Cheng, L. Li, K. C. Tan, and Y. Jin. Large-scale multiobjective
% optimization via reformulated decision variable analysis. IEEE
% Transactions on Evolutionary Computation, 2024, 28(1): 47-61.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    methods
        function main(Algorithm,Problem)
	        %% Initalization of population and structure
            [sN,N,gen]    = Algorithm.ParameterSet(3,10,20);
	        [W,Problem.N] = UniformPoint(Problem.N,Problem.M); 
	        Population    = Problem.Initialization();
            Population    = EvolveByMOEAD(Problem,Population,W,20);
            Par           = struct('N',N,'sN',sN,'Dec',rand(N,Problem.D)>0.5,...
                                   'fit',zeros(N,2),'gen',gen,'t',1);
            %% Optimization
            while Algorithm.NotTerminated(Population)
                t0 = Problem.FE;
                [Par,NewPop] = ReformulatedOptimization(Problem,Population,Par);
                Population   = EnvironmentSelection([Population,NewPop],Problem.N);
                for g = 1 : Par.N
                    DV = find(Par.Dec(g,:)==1);
                    Population = SecondOptimization(Problem,Population,DV,1);
                    drawnow();
                    PV = find(Par.Dec(g,:)==0);
                    Population = SecondOptimization(Problem,Population,PV,1);
                end
                deltaG     = ceil((Problem.FE-t0)/Problem.N);
                Population = EvolveByMOEAD(Problem,Population,W,deltaG);
            end
        end
    end
end

function [Population,fitness] = SparseFit(Problem,Archieve,Par,mask)
    Range       = [min(Archieve.objs,[],1);max(Archieve.objs,[],1)];
    fitness     = zeros(Par.N,2);
	[Dmin,Dmax] = deal(min(Archieve.decs,[],1),max(Archieve.decs,[],1));
    Upper       = repmat(Dmax,Par.sN,1);
    Lower       = repmat(Dmin,Par.sN,1);
    [~,index]   = min(calCon(Range,Archieve.objs));
	ArcDec      = repmat(Archieve(index).decs,Par.sN,1);

	Population = [];
    alpha = 0.25;
    for i = 1 : Par.N
        tempDec = ArcDec;    
        noise   = alpha.*unifrnd(0,1,Par.sN,Problem.D).*(Upper-Lower);
        tempDec(:,mask(i,:)) = noise(:,mask(i,:)) ;
        TEMP         = Problem.Evaluation(tempDec);
        fitness(i,1) = min(calCon(Range,TEMP.objs));
        fitness(i,2) = sum(mask(i,:));
        Population   = [Population,TEMP];
    end
end

function Population = SecondOptimization(Problem,Population,PV,flag)
% Convergence/Distribution optimization
    N          = length(Population);
    Range      = [min(Population.objs,[],1);max(Population.objs,[],1)];
	MatingPool = TournamentSelection(2,N,calCon(Range,Population.objs));
    OffDec     = Population(MatingPool).decs;
    if flag == 0
        NewDec = OperatorGA(Problem,Population(randi(N,1,N+1)).decs);
        NewDec = NewDec(1:N,:);
    else
        next   = randi(N,1,2*N);
        NewDec = OperatorDE(Problem,Population.decs,Population(next(1:end/2)).decs, ...
                 Population(next(end/2+1:end)).decs, ...
                 {1,0.5,size(Range,2)/length(PV)/2,20});
    end
    OffDec(:,PV) = NewDec(:,PV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentSelection([Population,Offspring],N);
end

function [Par,TEMP] = ReformulatedOptimization(Problem,Population,Par)
    [TEMP,Par.fit]       = SparseFit(Problem,Population,Par,Par.Dec>0);
    [~,FrontNo,CrowdDis] = FitnessSelection(Par.fit,Par.N);
    for i = 1 : Par.gen
        MatingPool = TournamentSelection(2,2*Par.N,FrontNo,-CrowdDis);    
        OffMask    = BinaryVariation(Par.Dec(MatingPool(1:Par.N),:), ...
                                      Par.Dec(MatingPool(Par.N+1:end),:));
        [OffPop,MaskFit] = SparseFit(Problem,Population,Par,OffMask>0);         
        TEMP    = [TEMP,OffPop];
        fitness = [Par.fit;MaskFit];
        PopDec  = [Par.Dec;OffMask];
        [Next,FrontNo,CrowdDis] = FitnessSelection(fitness,Par.N);
        Par.Dec = PopDec(Next,:);
        Par.fit = fitness(Next,:);
        drawnow();
    end
end

function Offspring = BinaryVariation(Parent1,Parent2)
% One point crossover and bitwise mutation

    [proC,proM] = deal(1,1);
    [N,D] = size(Parent1);
    k     = repmat(1:D,N,1) > repmat(randi(D,N,1),1,D);
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
    Site = rand(2*N,D) < proM/D;
    Offspring(Site) = ~Offspring(Site);
end

function Con = calCon(Range,PopuObj)
% Calculate the convergence of each solution
    PopuObj = (PopuObj - Range(1,:)) ./ (Range(2,:)-Range(1,:)+1e-6);
    Con     = sum(PopuObj,2);
end
```
