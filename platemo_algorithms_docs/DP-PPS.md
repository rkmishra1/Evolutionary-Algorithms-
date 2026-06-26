# DP-PPS

**Tags**: <2022> <multi> <real> <constrained>

## Description
Tri-population based push and pull search

## Reference
F. Ming, W. Gong, L. Wang, and C. Lu. A tri-population based co-evolutionary framework for constrained multi-objective optimization problems. Swarm and Evolutionary Computation, 2022, 70: 101055.

## Source Code

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

% This function is written by Fei Ming

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

### `DPPPS.m`
```matlab
classdef DPPPS< ALGORITHM
% <2022> <multi> <real> <constrained>
% Tri-population based push and pull search
% delta --- 0.9 --- The probability of choosing parents locally
% nr    ---   2 --- Maximum number of solutions replaced by each offspring

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, L. Wang, and C. Lu. A tri-population based
% co-evolutionary framework for constrained multi-objective optimization
% problems. Swarm and Evolutionary Computation, 2022, 70: 101055.
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
            [delta,nr] = Algorithm.ParameterSet(0.9,2);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z          = min(Population.objs,[],1);
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();

            alpha_c       = 2./(1+exp(1).^(-Problem.FE*10/(3*Problem.maxFE)))-1;
            para        = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/(3*Problem.N));

            %% Evaluate the Population
            Tc               = 0.9 * ceil(Problem.maxFE/Problem.N);
            last_gen         = 20;
            change_threshold = 1e-1;
            search_stage     = 1; % 1 for push stage,otherwise,it is in pull stage.
            max_change       = 1;
            epsilon_k        = 0;
            epsilon_0        = 0;
            cp               = 2;
            alpha            = 0.95;
            tao              = 0.05;
            ideal_points     = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            nadir_points     = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            arch             = archive(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                gen        = ceil(Problem.FE/(2*Problem.N));
                pop_cons   = Population.cons;
                cv         = overall_cv(pop_cons);
                population = [Population.decs,Population.objs,cv];
                rf         = sum(cv <= 1e-6) / Problem.N;
                ideal_points(gen,:) = Z;
                nadir_points(gen,:) = max(population(:,Problem.D + 1 : Problem.D + Problem.M),[],1);

                % The maximumrate of change of ideal and nadir points rk is calculated.
                if gen >= last_gen
                    max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen);
                end

                % The value of e(k) and the search strategy are set.
                if gen < Tc
                    if max_change <= change_threshold && search_stage == 1
                        search_stage = -1;
                        epsilon_0 = max(population(:,end),[],1);
                        epsilon_k = epsilon_0;
                    end
                    if search_stage == -1
                        epsilon_k =  update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp);
                    end
                else
                    epsilon_k = 0;
                end

                % For each solution
                for i = 1 : Problem.N
                    % Choose the parents
                    if rand < delta
                        P = B(i,randperm(size(B,2)));
                    else
                        P = randperm(Problem.N);
                    end

                    % Generate an offspring
                    Offspring = OperatorDE(Problem,Population(i),Population(P(1)),Population(P(2)));

                    % Update the ideal point
                    Z = min(Z,Offspring.obj);

                    g_old  = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                    g_new  = max(repmat(abs(Offspring.obj-Z),length(P),1).*W(P,:),[],2);
                    cv_old = overall_cv(Population(P).cons);
                    cv_new = overall_cv(Offspring.con) * ones(length(P),1);

                    if search_stage == 1 % Push Stage
                        Population(P(find(g_old>=g_new,nr))) = Offspring;
                    else  % Pull Stage  &&  An improved epsilon constraint-handling is employed to deal with constraints
                        Population(P(find(((g_old >= g_new) & (((cv_old <= epsilon_k) & (cv_new <= epsilon_k)) | (cv_old == cv_new)) | (cv_new < cv_old) ), nr))) = Offspring;
                    end
                end

                % Re-rank
                Population1 = Population1(randperm(Problem.N));
                Population2 = Population2(randperm(Problem.N));
                Lia   = ismember(Population2.objs,Population1.objs, 'rows');
                gamma = 1-sum(Lia)/Problem.N;
                [Population1,FrontNo1] = EnvironmentalSelection(Population1,Problem.N,alpha_c);
                [Population2,FrontNo2] = EnvironmentalSelection_noCon(Population2,Problem.N,alpha_c,gamma,para);

                % Offspring Reproduction
                Population_all   = [Population1,Population2];
                RankSolution_all = [FrontNo1,FrontNo2];
                MatingPool = TournamentSelection(2,2*Problem.N,RankSolution_all);
                Offspring  = OperatorGAhalf(Problem,Population_all(MatingPool));

                % Environmental Selection
                alpha_c = 2./(1+exp(1).^(-Problem.FE*5/Problem.maxFE)) - 1;
                para  = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/(2*Problem.N));
                [Population1,~] = EnvironmentalSelection([Population1,Offspring],Problem.N,alpha_c);
                [Population2,~] = EnvironmentalSelection_noCon([Population2,Offspring],Problem.N,alpha_c,gamma,para);

                % Output the non-dominated and feasible solutions.
                arch = archive([arch,Population],Problem.N);
                if Problem.FE >= Problem.maxFE
                    arch = archive([arch,Population1],Problem.N);
                    Population = arch;
                end
            end
        end
    end
end

function result = overall_cv(cv)
% The Overall Constraint Violation

    cv(cv <= 0) = 0;cv = abs(cv);
    result = sum(cv,2);
end

function max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen)
% Calculate the Maximum Rate of Change

    delta_value = 1e-6 * ones(1,size(ideal_points,2));
    rz  = abs((ideal_points(gen,:) - ideal_points(gen - last_gen + 1,:)) ./ max(ideal_points(gen - last_gen + 1,:),delta_value));
    nrz = abs((nadir_points(gen,:) - nadir_points(gen - last_gen + 1,:)) ./ max(nadir_points(gen - last_gen + 1,:),delta_value));
    max_change = max([rz, nrz]);
end

function result = update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp)
    if rf < alpha
        result = (1 - tao) * epsilon_k;
    else
        result = epsilon_0 * ((1 - (gen / Tc)) ^ cp);
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,RankSolution] = EnvironmentalSelection(Population,N,alpha)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Mengjun Ming

    %% Parameter
    popSize    = length(Population);
    NumSeq     = (1:popSize)';
    RankConvg  = zeros(popSize,1);
    RankDivs   = zeros(popSize,1);
    
    %% Modify the infeasible solutions
    PopObj = Population.objs;
    PopCon = Population.cons;
    z      = min(PopObj,[],1);
    n      = max(PopObj,[],1);
    M      = length(z);
    [W,~]  = UniformPoint(N,M);
    [~,Region]     = min(pdist2(PopObj-z,W,'cosine'),[],2);  
    PopObj_2       = PopObj;
    Infeasible_all = any(PopCon>0,2);
    if sum(Infeasible_all) ~= 0
        [~,Region_Fmax] = min(pdist2(n-z,W,'cosine'),[],2); 
        PopObj_2(Infeasible_all,:) = repmat(n,sum(Infeasible_all),1)+sum(max(0,PopCon(Infeasible_all,:)),2)*W(Region_Fmax,:)/norm(W(Region_Fmax,:));
    end
    
    %% Non-dominated sorting
    CV       = sum(max(0,PopCon),2);
    Dominate = false(popSize);
    for i = 1 : popSize-1
        for j = i+1 : popSize
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
    R = zeros(1,popSize);
    for i = 1 : popSize
        R(i) = sum(S(Dominate(:,i)));
    end
    FrontNo = R + 1;
    
    %% Calculate the crowding distance of each solution
    PopObj   = Population.objs;
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    CrowdDis = Distance(:,floor(sqrt(popSize)));
    
    %% Add a middle column
    MiddleLevel = zeros(popSize,1);
    
    %% Environmental selection -- convergence
    Next = FrontNo == 1;
    if sum(Next) <= N
        [~,indx_Convg] = sortrows([FrontNo',-CrowdDis]);
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
        MiddleLevel(Temp(Del)) = 1;
        [~,indx_Convg] = sortrows([FrontNo',MiddleLevel,-CrowdDis]);        
    end
    RankConvg(indx_Convg) = NumSeq;
    
    %% Environmental selection -- diversity
    FrontNo_D = ones(popSize,1);
    for i = 1:size(W,1)
        index = find(Region==i);
        if (~isempty(index))
            Objs_temp = PopObj_2(index,:);        
            g_temp = sum((Objs_temp-z).*W(i,:),2);
            [~,index_FrontNo_D] = sort(g_temp);
            FrontNo_D(index(index_FrontNo_D)) = (1:length(g_temp))';
        end
    end
    FrontNo_D           = FrontNo_D+Infeasible_all*popSize;
    [~,indx_divs]       = sortrows([FrontNo_D,-CrowdDis]);
    RankDivs(indx_divs) = NumSeq;
    
    %% Population for next generation
    RankSolution = alpha*RankConvg+(1-alpha)*RankDivs;
    [~,Rank]     = sort(RankSolution);
    Population   = Population(Rank(1:N));
    RankSolution = 1 : N;
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

### `EnvironmentalSelection_noCon.m`
```matlab
function [Population,RankSolution] = EnvironmentalSelection_noCon(Population,N,alpha,gamma,para)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Mengjun Ming

    %% Parameter
    popSize   = length(Population);
    NumSeq    = (1:popSize)';
    RankConvg = zeros(popSize,1);
    RankDivs  = zeros(popSize,1);
    
    %% Modify the infeasible solutions
    PopObj = Population.objs;
    PopCon = Population.cons;
    z      = min(PopObj,[],1);
    n      = max(PopObj,[],1);
    
    Infeasible_all = any(PopCon>0,2);
    phi_max = max(sum(max(0,PopCon(Infeasible_all,:)),2));
    
    M          = length(z);
    [W,~]      = UniformPoint(N,M);
    [~,Region] = min(pdist2(PopObj-z,W,'cosine'),[],2);  
    PopObj_2   = PopObj;
    for i = 1:size(W,1)
        index = find(Region==i);
        if (~isempty(index))
            Objs_temp  = PopObj_2(index,:);
            Cons_temp  = PopCon(index,:);
            Infeasible = any(Cons_temp>0,2);
            if (sum(Infeasible)~=0)
                F_max = max(Objs_temp,[],1);
                PopObj_2(index(Infeasible),:) = Objs_temp(Infeasible,:)+(sum(max(0,Cons_temp(Infeasible,:)),2)/phi_max).^(exp(para)/max(gamma,0.000001)).*(F_max-Objs_temp(Infeasible,:));
            end
        end
    end

    %% Non-dominated sorting
    Dominate = false(popSize);
    for i = 1 : popSize-1
        for j = i+1 : popSize
            k = any(PopObj_2(i,:)<PopObj_2(j,:)) - any(PopObj_2(i,:)>PopObj_2(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,popSize);
    for i = 1 : popSize
        R(i) = sum(S(Dominate(:,i)));
    end
    FrontNo = R + 1;
    
    %% Calculate the crowding distance of each solution
    PopObj   = Population.objs;
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    CrowdDis = Distance(:,floor(sqrt(popSize)));

    %% Add a middle column
    MiddleLevel = zeros(popSize,1);
    
    %% Environmental selection
    Next = FrontNo == 1;
    if sum(Next) <= N
        [~,indx_Convg] = sortrows([FrontNo',-CrowdDis]);        
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
        MiddleLevel(Temp(Del)) = 1;
        [~,indx_Convg] = sortrows([FrontNo',MiddleLevel,-CrowdDis]);
    end
    RankConvg(indx_Convg) = NumSeq;
    
    %% Environmental selection -- diversity
    FrontNo_D = ones(popSize,1);
    for i = 1:size(W,1)
        index = find(Region==i);
        if (~isempty(index))
            Objs_temp = PopObj_2(index,:);          
            g_temp = sum((Objs_temp-z).*W(i,:),2);
            [~,index_FrontNo_D] = sort(g_temp);
            FrontNo_D(index(index_FrontNo_D)) = (1:length(g_temp))';
        end
    end
    [~,indx_divs] = sortrows([FrontNo_D,-CrowdDis]);
    RankDivs(indx_divs) = NumSeq;
    
    %% Population for next generation
    RankSolution = alpha*RankConvg+(1-alpha)*RankDivs;
    [~,Rank]     = sort(RankSolution);
    Population   = Population(Rank(1:N));
    RankSolution = 1 : N;
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

### `archive.m`
```matlab
function Population = archive(Population,N)
% Select feasible and non-dominated solutions by using NSGA-II

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
    elseif size(Population,2) > N
        Fitness = CalFitness(Population.objs,Population.cons);
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
