# LSMOF

**Tags**: <2019> <multi> <real/integer> <large/none>

## Description
Large-scale multi-objective optimization framework with NSGA-II

## Reference
C. He, L. Li, Y. Tian, X. Zhang, R. Cheng, Y. Jin, and X. Yao. Accelerating large-scale multi-objective optimization via problem reformulation. IEEE Transactions on Evolutionary Computation, 2019, 23(6): 949-961.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of LSMOF

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He
 
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Population = Population(Next);
end
```

### `LSMOF.m`
```matlab
classdef LSMOF < ALGORITHM
% <2019> <multi> <real/integer> <large/none>
% Large-scale multi-objective optimization framework with NSGA-II
% wD       --- 10 --- The generation of weight optimization with DE
% SubN     --- 30 --- The population size of the transferred problem
% operator ---  1 --- Original operators 1. GA 2. DE

%------------------------------- Reference --------------------------------
% C. He, L. Li, Y. Tian, X. Zhang, R. Cheng, Y. Jin, and X. Yao.
% Accelerating large-scale multi-objective optimization via problem
% reformulation. IEEE Transactions on Evolutionary Computation, 2019,
% 23(6): 949-961.
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
            %% Parameter settings
            [wD,SubN,Operator] = Algorithm.ParameterSet(10,30,2);
            Population = Problem.Initialization();
            G          = ceil(Problem.maxFE*0.05/(SubN*2*wD));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Problem.FE < 0.6*Problem.maxFE
                    Archive    = WeightOptimization(Problem,G,Population,wD,SubN);
                    Population = EnvironmentalSelection([Population,Archive],Problem.N);
                else
                    Population = subNSGAII(Problem,Population,Operator,Problem.N);
                end
            end
        end
    end
end
```

### `WeightOptimization.m`
```matlab
function Arc = WeightOptimization(Problem,G2,Population,wD,N)
% The second step of LSMOF, which aims to search the PF according to the
% bi-direction weight vectors

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

	%% Choose NF solutions as the reference solutions
	Reference    = max(Population.objs,[],1);
    [RefPop,~,~] = EnvironmentalSelection(Population,wD);
    
    %% Calculate the reference directions
	Direction = [sum((RefPop.decs-repmat(Problem.lower,wD,1)).^2,2).^(0.5);sum((repmat(Problem.upper,wD,1)-RefPop.decs).^2,2).^(0.5)];
	Direct    = [(RefPop.decs-repmat(Problem.lower,wD,1));(repmat(Problem.upper,wD,1)-RefPop.decs)]./repmat(Direction,1,Problem.D);
	wmax      = sum((Problem.upper-Problem.lower).^2)^(0.5)*0.5;
    
    %% Optimize the weight variables by DE
	w0 = rand(N,2*wD).*wmax;                                    % Initialize the population
    [fitness,PopNew] = fitfunc(Problem,w0,Direct,Reference);	% Calculate the fitness and store the solutions
	Arc      = PopNew(NDSort(PopNew.objs,1)==1);
	pCR      = 0.2;
    beta_min = 0.2;   % Lower Bound of Scaling Factor
    beta_max = 0.8;   % Upper Bound of Scaling Factor
    empty_individual.Position = [];
    empty_individual.Cost     = [];
    pop = repmat(empty_individual,N,1);
    for i = 1 : N
        pop(i).Position = w0(i,:);
        pop(i).Cost     = fitness(i);
    end
    temp = [];
    for it = 1 : G2
        for i = 1 : N
            x = pop(i).Position;
            A = randperm(N);
            A(A==i) = [];
            a = A(1);
            b = A(2);
            c = A(3);
            % Mutation
            beta = unifrnd(beta_min,beta_max,[1 2*wD]);
            y    = pop(a).Position + beta.*(pop(b).Position - pop(c).Position);
            y    = min(max(y,0),wmax);
            % Crossover
            z  = zeros(size(x));
            j0 = randi([1 numel(x)]);
            for j = 1 : numel(x)
                if j==j0 || rand<=pCR
                    z(j) = y(j);
                else
                    z(j) = x(j);
                end
            end
            NewSol.Position = z;
            [fit,PopNew]    = fitfunc(Problem,z,Direct,Reference);
            temp = [temp,PopNew];
            temp = temp(NDSort(temp.objs,1)==1);
            NewSol.Cost = fit;
            if NewSol.Cost < pop(i).Cost
                pop(i) = NewSol;
            end
        end
    end
    % Update and store the non-dominated solutions
    Arc = [Arc,temp];
    if length(Arc) > Problem.N
        [frontNo,~] = NDSort(Arc.objs,1);
        Arc         = Arc(frontNo==1);
    end
end

function [Obj,OffSpring] = fitfunc(Problem,w0,direct,Reference)
    [SubN,WD] = size(w0); 
    WD        = WD/2;
    Obj   	  = zeros(SubN,1);
    OffSpring = [];
    for i = 1 : SubN 
        PopDec    = [repmat(w0(i,1:WD)',1,Problem.D).*direct(1:WD,:)+repmat(Problem.lower,WD,1);
                     repmat(Problem.upper,WD,1) - repmat(w0(i,WD+1:end)',1,Problem.D).*direct(WD+1:end,:)];
        OffWPop   = Problem.Evaluation(PopDec);
        OffSpring = [OffSpring,OffWPop];
        Obj(i)    = -HV(OffWPop,Reference);
    end
end
```

### `subNSGAII.m`
```matlab
function Population = subNSGAII(Problem,Population,Operator,N)
% Sub-optimizer in LSMOF (NSGA-II)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He
    
    FrontNo  = NDSort(Population.objs,inf);
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    if Operator == 1
        MatingPool = TournamentSelection(2,N,FrontNo,-CrowdDis);
        Offspring  = OperatorGA(Problem,Population(MatingPool));
    else
        MatingPool1 = TournamentSelection(2,N,FrontNo,-CrowdDis);
        MatingPool2 = TournamentSelection(2,N,FrontNo,-CrowdDis);
        MatingPool3 = TournamentSelection(2,N,FrontNo,-CrowdDis);
        Offspring   = OperatorDE(Problem,Population(MatingPool1),Population(MatingPool2),Population(MatingPool3));
    end
    Population = EnvironmentalSelection([Population,Offspring],N);
end
```
