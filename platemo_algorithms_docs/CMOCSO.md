# CMOCSO

**Tags**: <2023> <multi> <real> <large/none> <constrained>

## Description
Competitive and cooperative swarm optimization constrained multi-objective optimization algorithm

## Reference
F. Ming, W. Gong, D. Li, L. Wang, and L. Gao. A competitive and cooperative swarm optimizer for constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2023, 27(5): 1313-1326.

## Source Code

### `CMOCSO.m`
```matlab
classdef CMOCSO < ALGORITHM
% <2023> <multi> <real> <large/none> <constrained>
% Competitive and cooperative swarm optimization constrained multi-objective optimization algorithm

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, D. Li, L. Wang, and L. Gao. A competitive and
% cooperative swarm optimizer for constrained multi-objective optimization
% problems. IEEE Transactions on Evolutionary Computation, 2023, 27(5):
% 1313-1326.
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
            %% Generate random population
            Population = Problem.Initialization();
            CV         = sum(max(0,Population.cons),2);
            CVmax      = max(CV);
            epsilon_0  = CVmax;
            epsilon    = epsilon_0;
            Competitive_Pop = UpdateP1(Population,Problem.N,epsilon);
            [Cooperative_Pop,Cooperative_Pop_Fitness] = UpdateP2(Population,Problem.N);
            Population = UpdateP(Population,Problem.N);
            Tc    = 0.9 * ceil(Problem.maxFE/Problem.N);
            cp    = 2;
            alpha = 0.95;
            tao   = 0.05;
            y     = 10;
            G     = Problem.maxFE/Problem.N;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                gen       = ceil(Problem.FE/Problem.N);
                CV        = sum(max(0,Competitive_Pop.cons),2);
                CV_max    = max(CV);
                CVmax     = max([CV_max,CVmax]);
                epsilon_0 = CVmax;
                rf = sum(CV <= 1e-6) / length(Competitive_Pop);
                epsilon = update_epsilon(tao,epsilon,epsilon_0,rf,alpha,gen,Tc,cp);
                
                Competitive_Pop_Fitness = CalFitness(Competitive_Pop.objs,Competitive_Pop.cons,epsilon);
                
                if length(Competitive_Pop) >= 2
                    Rank = randperm(length(Competitive_Pop),floor(length(Competitive_Pop)/2)*2);
                else
                    Rank = [1,1];
                end
                Loser  = Rank(1:end/2);
                Winner = Rank(end/2+1:end);
                Change = Competitive_Pop_Fitness(Loser) <= Competitive_Pop_Fitness(Winner);
                Temp   = Winner(Change);
                Winner(Change) = Loser(Change);
                Loser(Change)  = Temp;
                Offspring1 = CompetitiveOperator(Problem,Competitive_Pop(Loser),Competitive_Pop(Winner),y);
                
                LearningPool = TournamentSelection(2,Problem.N,Cooperative_Pop_Fitness);
                Offspring2   = CooperativeOperator(Problem,Cooperative_Pop(LearningPool));
                
                Offspring = [Offspring1,Offspring2];
                
                Population = UpdateP([Population,Offspring],Problem.N);
                Competitive_Pop = UpdateP1([Competitive_Pop,Offspring],Problem.N,epsilon);
                gen = ceil(Problem.FE/Problem.N);
                y   = (Problem.M)^2*((gen/G)-1)^2+1;
                [Cooperative_Pop,Cooperative_Pop_Fitness] = UpdateP2([Offspring,Cooperative_Pop],Problem.N);
            end
        end
    end
end

function epsilon = update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp)
% Update the value of epsilon

    if gen > Tc
        epsilon = 0;
    else
        if rf < alpha
            epsilon = (1 - tao) * epsilon_k;
        else
            epsilon = epsilon_0 * ((1 - (gen / Tc)) ^ cp);
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon,epsilon)
% Calculate the fitness of each solution based on different epsilon

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    N = size(PopObj,1);
    CV = sum(max(0,PopCon),2);
    CV(CV < epsilon) = 0;

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

### `CompetitiveOperator.m`
```matlab
function Offspring = CompetitiveOperator(Problem,Loser,Winner,y)
% The competitive swarm optimizer

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    %% Parameter setting
    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(LoserDec);
	LoserVel  = Loser.adds(zeros(N,D));
    WinnerVel = Winner.adds(zeros(N,D));

    %% Competitive swarm optimizer
    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    OffVel = r1.*LoserVel + r2.*(WinnerDec-LoserDec)*y;
    n = randi(2,1,1);
    OffDec = LoserDec + OffVel + r1.*(OffVel-LoserVel)*(-1)^n;
    
    %% Add the winners
    OffDec = [OffDec;WinnerDec];
    OffVel = [OffVel;WinnerVel];
 
    %% Polynomial mutation
    Lower  = repmat(Problem.lower,2*N,1);
    Upper  = repmat(Problem.upper,2*N,1);
    disM   = 20;
    Site   = rand(2*N,D) < 1/D;
    mu     = rand(2*N,D);
    temp   = Site & mu<=0.5;
    OffDec       = max(min(OffDec,Upper),Lower);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `CooperativeOperator.m`
```matlab
function Offspring = CooperativeOperator(Problem,Parent,Parameter)
% The cooperative swarm optimizer based on GA formula

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    %% Parameter setting
    if nargin > 2
        [proC,disC,proM,disM] = deal(Parameter{:});
    else
        [proC,disC,proM,disM] = deal(1,20,1,20);
    end
    if isa(Parent(1),'SOLUTION')
        calObj = true;
        Parent = Parent.decs;
    else
        calObj = false;
    end
    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);

    %% Genetic operators for real encoding
    % Simulated binary crossover
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
    % Polynomial mutation
    Lower = repmat(Problem.lower,2*N,1);
    Upper = repmat(Problem.upper,2*N,1);
    Site  = rand(2*N,D) < proM/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
                  
    if calObj
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `UpdateP.m`
```matlab
function Population = UpdateP(Population,N)
% Select feasible and non-dominated solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    %% Select feasible solutions
    fIndex     = all(Population.cons <= 0,2);
    Population = Population(fIndex);

    if isempty(Population)
        return
    elseif length(Population)>N
        Fitness = CalFitness(Population.objs,Population.cons,0);
        Next = Fitness < 1;
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
        Population = Population(Next);
    end
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

### `UpdateP1.m`
```matlab
function [Population,Fitness] = UpdateP1(Population,N,epsilon)
% Selecte epsilon feasible solutions

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
    Fitness = CalFitness(Population.objs,Population.cons,epsilon);

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

### `UpdateP2.m`
```matlab
function [Population,Fitness] = UpdateP2(Population,N)
% Select non-dominated solutions

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
    Fitness = CalFitness(Population.objs,Population.cons,inf);

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
    Fitness = Fitness(Next);
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
