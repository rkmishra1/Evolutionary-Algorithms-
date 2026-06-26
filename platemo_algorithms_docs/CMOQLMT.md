# CMOQLMT

**Tags**: <2023> <multi> <real> <constrained>

## Description
Constrained multi-objective optimization based on Q-learning and multitasking

## Reference
F. Ming, W. Gong, and L. Gao. Adaptive auxiliary task selection for multitasking-assisted constrained multi-objective optimization [feature]. IEEE Computational Intelligence Magazine, 2023, 18(2): 18-30.

## Source Code

### `CMOQLMT.m`
```matlab
classdef CMOQLMT < ALGORITHM
% <2023> <multi> <real> <constrained>
% Constrained multi-objective optimization based on Q-learning and multitasking
% delta --- 0.9 --- The probability of choosing parents locally
% nr    ---   2 --- Maximum number of solutions replaced by each offspring

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, and L. Gao. Adaptive auxiliary task selection for 
% multitasking-assisted constrained multi-objective optimization [feature].
% IEEE Computational Intelligence Magazine, 2023, 18(2): 18-30.
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
            Population{1} = Problem.Initialization();
            Population{2} = Problem.Initialization();
            Population{3} = Problem.Initialization();
            alpha1 = 2./(1+exp(1).^(-Problem.FE*10/Problem.maxFE))-1;
            para = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/Problem.N);
            Population{4} = Problem.Initialization();
            Zmin       = min(Population{4}.objs,[],1);
            Fmin       = min(Population{4}(all(Population{4}.cons<=0,2)).objs,[],1);
            Fitness{1} = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{3} = CalFitness(Population{3}.objs);
            LengthO{1} = Problem.N/2; LengthO{2} = Problem.N/2; LengthO{3} = Problem.N/2; LengthO{4} = Problem.N/2;
            W          = UniformPoint(Problem.N,Problem.M);
            Ra         = 1;
            
            %% For QL
            num_task = 3;
            Q_Table = zeros(num_task,num_task);
            learning = 0.5;
            gama_ql = 0.9;
            alpha_ql = 0.8;
            greedy_ql = 0.9;
            current_state = randi(num_task);

            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                if Problem.FE < learning * Problem.maxFE
                    % choose an action by greedy strategy based on Q table
                    if rand > greedy_ql || ( Q_Table(current_state,1) == Q_Table(current_state,2) && Q_Table(current_state,1) == Q_Table(current_state,3) )
                        action = randi(num_task);
                    else
                        [~,action] = max(Q_Table(current_state,:));
                    end   
                    
                    % Re-rank
                    Population{1} = Population{1}(randperm(Problem.N));
                    Population{2} = Population{2}(randperm(Problem.N));
                    Lia   = ismember(Population{2}.objs,Population{1}.objs, 'rows');
                    gamma = 1-sum(Lia)/Problem.N;
                    [Population{2},Fitness{2},~] = EnvironmentalSelectionT2(Population{2},Problem.N,alpha1,gamma,para);
                    
                    % Offspring Reproduction
                    MatingPool{1} = TournamentSelection(2,2*LengthO{1},Fitness{1});
                    Offspring{1}  = OperatorGAhalf(Problem,Population{1}(MatingPool{1}));
                    MatingPool{2} = TournamentSelection(2,2*LengthO{2},Fitness{2});
                    Offspring{2}  = OperatorGAhalf(Problem,Population{2}(MatingPool{2}));
                    MatingPool{3} = TournamentSelection(2,2*LengthO{3},Fitness{3});
                    Offspring{3}  = OperatorGAhalf(Problem,Population{3}(MatingPool{3}));
                    Nt     = floor(Ra*Problem.N);
                    if length(Population{1}) > Problem.N-Nt
                        MatingPool{4} = [Population{4}(randsample(Problem.N,Nt)),Population{1}(randsample(Problem.N,Problem.N-Nt))];
                    else
                        MatingPool{4} = Population{4}(randsample(Problem.N,Problem.N));
                    end
                    [Mate1,Mate2,Mate3]        = Neighbor_Pairing_Strategy(MatingPool{4},Zmin);
                    if rand > 0.5
                        Offspring{4}  = OperatorDE(Problem,Mate1,Mate2,Mate3);
                    else
                        Offspring{4}  = OperatorDE(Problem,Mate1,Mate2,Mate3,{0.5,0.5,0.5,0.75});
                    end
                    
                    % determine the transfer rate to update Q table
                    [~,~,Next1] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{action+1},Offspring{action+1}],Problem.N);
                    succ_rate =  (sum(Next1(length(Population{1})+length(Offspring{1})+1:end))) / (length(Population{action+1})+length(Offspring{action+1}));
                    Q_Table(current_state,action) = Q_Table(current_state,action) + alpha_ql * (succ_rate + gama_ql * (max(Q_Table(action,:))) - Q_Table(current_state,action));
                    
                    current_state = action;
                    
                    % Environmental Selection
                    alpha1 = 2./(1+exp(1).^(-Problem.FE*10/Problem.maxFE)) - 1;
                    para  = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/Problem.N);
                    Fmin       = min([Fmin;Offspring{4}(all(Offspring{4}.cons<=0,2)).objs],[],1);
                    Zmin       = min([Zmin;Offspring{4}.objs],[],1);
                    [Population{1},Fitness{1},~] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{current_state+1},Offspring{current_state+1}],Problem.N);
                    [Population{2},~,~] = EnvironmentalSelectionT2([Population{2},Offspring{2}],Problem.N,alpha1,gamma,para);
                    [Population{3},Fitness{3},~] = EnvironmentalSelectionT3([Population{3},Offspring{3}],Problem.N);
                    [Population{4},~] = ICMA_Update([Population{4},Offspring{4}],Problem.N,W,Zmin,Fmin);
                    
                else
                    % choose an action by greedy strategy based on Q table
                    if rand > greedy_ql || ( Q_Table(current_state,1) == Q_Table(current_state,2) && Q_Table(current_state,1) == Q_Table(current_state,3) )
                        action = randi(num_task);
                    else
                        [~,action] = max(Q_Table(current_state,:));
                    end
                    
                    % Offspring Reproduction
                    MatingPool{1} = TournamentSelection(2,2*LengthO{1},Fitness{1});
                    Offspring{1}  = OperatorGAhalf(Problem,Population{1}(MatingPool{1}));
                    
                    if action == 1
                        % Re-rank
                        Population{1} = Population{1}(randperm(Problem.N));
                        Population{2} = Population{2}(randperm(Problem.N));
                        Lia   = ismember(Population{2}.objs,Population{1}.objs, 'rows');
                        gamma = 1-sum(Lia)/Problem.N;
                        [Population{2},Fitness{2},~] = EnvironmentalSelectionT2(Population{2},Problem.N,alpha1,gamma,para);
                        MatingPool{2} = TournamentSelection(2,2*LengthO{2},Fitness{2});
                        Offspring{2}  = OperatorGAhalf(Problem,Population{2}(MatingPool{2}));
                        
                        % determine the transfer rate to update Q table
                        [~,~,Next1] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{action+1},Offspring{action+1}],Problem.N);
                        succ_rate =  (sum(Next1(length(Population{1})+length(Offspring{1})+1:end))) / (length(Population{action+1})+length(Offspring{action+1}));
                        Q_Table(current_state,action) = Q_Table(current_state,action) + alpha_ql * (succ_rate + gama_ql * (max(Q_Table(action,:))) - Q_Table(current_state,action));
                        
                        current_state = action;
                        
                        % Environmental Selection
                        alpha1 = 2./(1+exp(1).^(-Problem.FE*10/Problem.maxFE)) - 1;
                        para  = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/Problem.N);
                        [Population{1},Fitness{1},~] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{current_state+1},Offspring{current_state+1}],Problem.N);
                        [Population{2},~,~] = EnvironmentalSelectionT2([Population{2},Offspring{2}],Problem.N,alpha1,gamma,para);
                    elseif action == 2
                        MatingPool{3} = TournamentSelection(2,2*LengthO{3},Fitness{3});
                        Offspring{3}  = OperatorGAhalf(Problem,Population{3}(MatingPool{3}));
                        
                        % determine the transfer rate to update Q table
                        [~,~,Next1] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{action+1},Offspring{action+1}],Problem.N);
                        succ_rate =  (sum(Next1(length(Population{1})+length(Offspring{1})+1:end))) / (length(Population{action+1})+length(Offspring{action+1}));
                        Q_Table(current_state,action) = Q_Table(current_state,action) + alpha_ql * (succ_rate + gama_ql * (max(Q_Table(action,:))) - Q_Table(current_state,action));
                        
                        current_state = action;
                        
                        % Environmental Selection
                        [Population{1},Fitness{1},~] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{current_state+1},Offspring{current_state+1}],Problem.N);
                        [Population{3},Fitness{3},~] = EnvironmentalSelectionT3([Population{3},Offspring{3}],Problem.N);
                    else
                        Nt     = floor(Ra*Problem.N);
                        if length(Population{1}) > Problem.N-Nt
                            MatingPool{4} = [Population{4}(randsample(Problem.N,Nt)),Population{1}(randsample(Problem.N,Problem.N-Nt))];
                        else
                            MatingPool{4} = Population{4}(randsample(Problem.N,Problem.N));
                        end
          
                        [Mate1,Mate2,Mate3]        = Neighbor_Pairing_Strategy(MatingPool{4},Zmin);
                        if rand > 0.5
                            Offspring{4}  = OperatorDE(Problem,Mate1,Mate2,Mate3);
                        else
                            Offspring{4}  = OperatorDE(Problem,Mate1,Mate2,Mate3,{0.5,0.5,0.5,0.75});
                        end
                        
                        % determine the transfer rate to update Q table
                        [~,~,Next1] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{action+1},Offspring{action+1}],Problem.N);
                        succ_rate =  (sum(Next1(length(Population{1})+length(Offspring{1})+1:end))) / (length(Population{action+1})+length(Offspring{action+1}));
                        Q_Table(current_state,action) = Q_Table(current_state,action) + alpha_ql * (succ_rate + gama_ql * (max(Q_Table(action,:))) - Q_Table(current_state,action));
                        
                        current_state = action;
                        
                        % Environmental Selection
                        Fmin       = min([Fmin;Offspring{4}(all(Offspring{4}.cons<=0,2)).objs],[],1);
                        Zmin       = min([Zmin;Offspring{4}.objs],[],1);
                        [Population{1},Fitness{1},~] = EnvironmentalSelectionT1([Population{1},Offspring{1},Population{current_state+1},Offspring{current_state+1}],Problem.N);
                        [Population{4},~] = ICMA_Update([Population{4},Offspring{4}],Problem.N,W,Zmin,Fmin);
                    end
                end
                Ra = 1 - Problem.FE/Problem.maxFE;
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

### `EnvironmentalSelectionT1.m`
```matlab
function [Population,Fitness,Next] = EnvironmentalSelectionT1(Population,N)

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
    Fitness = CalFitness(Population.objs,Population.cons);

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

### `EnvironmentalSelectionT2.m`
```matlab
function [Population,Fitness,Next] = EnvironmentalSelectionT2(Population,N,alpha,gamma,para)

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
    Next1 = FrontNo == 1;
    if sum(Next1) <= N
        [~,indx_Convg] = sortrows([FrontNo',-CrowdDis]);        
    elseif sum(Next1) > N
        Del  = Truncation(Population(Next1).objs,sum(Next1)-N);
        Temp = find(Next1);
        Next1(Temp(Del)) = false;
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
    Next = zeros(1,length(Population));
    Population   = Population(Rank(1:N));
    Fitness = RankSolution(Rank(1:N));
    Choosed = Rank(1:N);
    Next(Choosed) = 1;
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

### `EnvironmentalSelectionT3.m`
```matlab
function [Population,Fitness,Next] = EnvironmentalSelectionT3(Population,N)

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
    Fitness = CalFitness(Population.objs);

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

### `ICMA_Update.m`
```matlab
function [Population,Archive] = ICMA_Update(MaxPop,N,W,Zmin,Fmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    PopObj               = MaxPop.objs;
    [Num,M]              = size(PopObj);
    Cons                 = max(0,MaxPop.cons);
    NormCons             = Cons./repmat(max(1,max(Cons,[],1)),Num,1);

    CV                   = sum(NormCons,2);

    % shift the objective space to R+
    PopObj               = PopObj - repmat(Zmin,Num,1) + 1e-6;

    % calculate the indicator matrix
    IMatrix              = ones(Num,Num);
    for i = 1:1:Num
        Ci               = CV(i);    
        if Ci == 0 %%%%% Xi is feasible
            Fi               = PopObj(i,:);
            Ir               = log(repmat(Fi,Num,1)./PopObj);
            MaxIr            = max(Ir,[],2);
            MinIr            = min(Ir,[],2);
            CVA              = MaxIr;
            DomInds          = find(MaxIr<=0);
            CVA(DomInds)     = MinIr(DomInds);
            IndicatorV       = CVA;
        else  %%%%% Xi is an infeasible solution
            IC               = repmat(Ci+1e-6,Num,1)./(CV+1e-6); 
            Fi = PopObj(i,:);
            MaxF = max(repmat(Fi,Num,1),PopObj);
            MinF = min(repmat(Fi,Num,1),PopObj);
            CVF = max(MaxF./MinF,[],2);
            IndicatorV       = log(max([CVF,IC],[],2));
        end
        IMatrix(:,i)     = IndicatorV;
        IMatrix(i,i)     = Inf;
    end

    FeasibleInd                   = find(CV==0);
    Len_F                         = length(FeasibleInd);

    if Len_F<=N
        [~,CV_SortInd]            = sort(CV);
        Archive                   = MaxPop(CV_SortInd(1:N));
    else
        FPopObj                   = PopObj(FeasibleInd,:) + repmat(Zmin,Len_F,1) - repmat(Fmin,Len_F,1);
        SelInd                    = Selection_Operator_of_PREA(FPopObj,IMatrix(FeasibleInd,FeasibleInd),N);
        Archive                   = MaxPop(FeasibleInd(SelInd));
    end

    %%%%%%%%%%%%%  using indicator-based CHT to update the population
    SelInd                        = Indicator_based_CHT(PopObj,IMatrix,W,N);
    Population                    = MaxPop(SelInd);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
```

### `Indicator_based_CHT.m`
```matlab
function SelInd = Indicator_based_CHT(PopObj,IMatrix,W,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    % calculate the size of individuals in the promising areas
    IrFitness                         = min(IMatrix,[],2);
    Level1Index                       = find(IrFitness>=0);
    Len_Level1                        = length(Level1Index);

    if Len_Level1<=N
        [~,SortIndex]                 = sort(-IrFitness);
        SelInd                        = SortIndex(1:N);
    else
        % only focus on the solutions in the promising areas
        SelInd                        = Level1Index;
        PopObj                        = PopObj(Level1Index,:);
        IMatrix                       = IMatrix(Level1Index,Level1Index);

        [Num,M]                       = size(PopObj);
        NormW                         = W./repmat(sqrt(sum(W.^2,2)),1,M);
        NormPopObj                    = PopObj./repmat(sqrt(sum(PopObj.^2,2)),1,M);
        [~,ZoneIndex]                 = max(NormPopObj * NormW',[],2);
        Num_W                         = size(W,1);
        ZoneDensity                   = zeros(1,Num_W);
        zone.index                    = [];
        Zone                          = repmat(zone,1,Num_W);
        for j = 1:Num
            Zj                        = ZoneIndex(j);
            Zone(Zj).index            = [Zone(Zj).index,j];
            ZoneDensity(Zj)           = ZoneDensity(Zj) + 1;
        end

        [NDensity,SortIndex]          = sort(-ZoneDensity);
        Density                       = abs(NDensity);
        [Values,Neightboor]           = min(IMatrix,[],2);

        DelNum           = Num - N;
        Have_Delect      = zeros(1,DelNum);

        for i = 1:DelNum
            [MDen,MDInd] = max(Density);
            CandidateIndex           = Zone(SortIndex(MDInd)).index;

            [~,NowDel_Ind]           = min(Values(CandidateIndex));
            Del_Ind                  = CandidateIndex(NowDel_Ind);
            CandidateIndex(NowDel_Ind) = [];
            Have_Delect(i) = Del_Ind;
            IMatrix(Del_Ind,:) = Inf;
            IMatrix(:,Del_Ind) = Inf;
            Need_Updata = find(Neightboor==Del_Ind);
            L_Need=length(Need_Updata);
            if L_Need>0
                [Values(Need_Updata),Neightboor(Need_Updata)]=min(IMatrix(Need_Updata,:),[],2);
            end
            Values(Del_Ind) = Inf;

            Zone(SortIndex(MDInd)).index = CandidateIndex;
            Density(MDInd) = MDen - 1;

        end
        SelInd(Have_Delect) = [];
    end
end
```

### `Neighbor_Pairing_Strategy.m`
```matlab
function [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPop,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    Objs = MatingPop.objs;
    [Num,M] = size(Objs);
    Objs = (Objs - repmat(Zmin,Num,1));
    Objs = Objs./repmat(sqrt(sum(Objs.^2,2)),1,M);

    CosV = Objs * Objs';
    CosV = CosV - 3*eye(Num,Num);

    [~,SInd] = sort(-CosV,2);

    Nr = 10;
    Neighbor = SInd(:,1:Nr);

    Mate1 = MatingPop;

    P = ones(Num,2);
    for i = 1:Num
        P(i,1:2) = Neighbor(i,randsample(Nr,2));
        if rand>0.7
            P(i,2) = randsample(Num,1);
        end
    end

    Mate2 = MatingPop(P(:,1));
    Mate3 = MatingPop(P(:,2));
end
```

### `Selection_Operator_of_PREA.m`
```matlab
function NextIndex = Selection_Operator_of_PREA(PopObj,IMatrix,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    [~,M]                    = size(PopObj);

    % calculate the size of individuals in the first nondominant level
    IrFitness                = min(IMatrix,[],2);
    Level1Index              = find(IrFitness>=0);
    Len_Level1               = length(Level1Index);

    if Len_Level1<=N
        [~,SortIndex]    = sort(-IrFitness);
        NextIndex        = SortIndex(1:N);
    else
        % only focus on the solutions in the first level
        AllIndex         = Level1Index;
        PopObj           = PopObj(Level1Index,:);
        IMatrix          = IMatrix(Level1Index,Level1Index);

        %% select the valuable solutions in the current population
        MiddleIMatrix    = IMatrix;
        Ag1_n            = Len_Level1 - N;
        [Values,Neightboor] = min(MiddleIMatrix,[],2);
        BestInd          = 1:Len_Level1;
        Have_Delect      = zeros(1,Ag1_n);

        % mark the N solutions with the best fitness value by excluding the
        % worst individuals one by one
        for i=1:Ag1_n
            [~,Del_Ind]  = min(Values);
            Have_Delect(i) = Del_Ind;
            MiddleIMatrix(Del_Ind,:) = Inf;
            MiddleIMatrix(:,Del_Ind) = Inf;
            Need_Updata = find(Neightboor==Del_Ind);
            L_Need=length(Need_Updata);
            if L_Need>0
                [Values(Need_Updata),Neightboor(Need_Updata)]=min(MiddleIMatrix(Need_Updata,:),[],2);
            end
            Values(Del_Ind) = Inf;
        end
        BestInd(Have_Delect) = [];

        % determine the boundary of promising region
        Zmax                 = max(PopObj(BestInd,:),[],1);

        % remove the individuals outside the promising region
        OutIndex             = find(min(repmat(Zmax,Len_Level1,1) - PopObj,[],2) < 0);
        AllIndex(OutIndex)   = [];
        PopObj(OutIndex,:)   = [];
        IMatrix(OutIndex,:)  = [];
        IMatrix(:,OutIndex)  = [];
        Num                  = length(AllIndex);


        %% diversity maintance mechanism based on parallel distance
        % normalize the promising region
        PopObj               = PopObj./repmat(Zmax,Num,1);

        [Ir_Values,Ir_Neightboor] = min(IMatrix,[],2);

        DelectInd2 = [];

        DMatrix              = zeros(Num,Num);

            % calculate the parallel distance matrix
            for i = 1:Num
                Fi = PopObj(i,:);
                Fdelta = PopObj - repmat(Fi,Num,1);
                DMatrix(i,:) = sqrt(sum(Fdelta.^2,2) - (sum(Fdelta,2)).^2./M);
                DMatrix(i,i) = Inf;
            end

        [Dis_Values,Dis_Neightboor] = min(DMatrix,[],2);


        for l=1:(Num - N)
            [~,individual1]=min(Dis_Values);
            individual2=Dis_Neightboor(individual1);

            if Ir_Values(individual1)<Ir_Values(individual2)
                k = individual1;
            else
                k = individual2;
            end

            DelectInd2 = [DelectInd2,k];

            DMatrix(k,:) = Inf;
            DMatrix(:,k) = Inf;
            Need_Updata_Dis=find(Dis_Neightboor==k);
            L_Need_Dis=length(Need_Updata_Dis);
            if L_Need_Dis>0
                [Dis_Values(Need_Updata_Dis),Dis_Neightboor(Need_Updata_Dis)]=min(DMatrix(Need_Updata_Dis,:),[],2);
            end
            Dis_Values(DelectInd2)= Inf;



            IMatrix(k,:) = Inf;
            IMatrix(:,k) = Inf;
            Need_Updata_Ir=find(Ir_Neightboor==k);
            L_Need_Ir=length(Need_Updata_Ir);
            if L_Need_Ir>0
                [Ir_Values(Need_Updata_Ir),Ir_Neightboor(Need_Updata_Ir)]=min(IMatrix(Need_Updata_Ir,:),[],2);
            end
        end
        AllIndex(DelectInd2) = [];

        NextIndex = AllIndex;
    end
end
```
