# c-DPEA

**Tags**: <2021> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Constrained dual-population evolutionary algorithm

## Reference
M. Ming, A. Trivedi, R. Wang, and D. Srinivasan. A dual-population based evolutionary algorithm for constrained multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(4): 739-753.

## Source Code

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

### `cDPEA.m`
```matlab
classdef cDPEA < ALGORITHM
% <2021> <multi> <real/integer/label/binary/permutation> <constrained>
% Constrained dual-population evolutionary algorithm

%------------------------------- Reference --------------------------------
% M. Ming, A. Trivedi, R. Wang, and D. Srinivasan. A dual-population based
% evolutionary algorithm for constrained multi-objective optimization. IEEE
% Transactions on Evolutionary Computation, 2021, 25(4): 739-753.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Mengjun Ming

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population1 = Problem.Initialization(); 
            Population2 = Population1; 
            alpha       = 2./(1+exp(1).^(-Problem.FE*10/Problem.maxFE))-1;
            para        = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                % Re-rank
                Population1 = Population1(randperm(Problem.N));
                Population2 = Population2(randperm(Problem.N));
                Lia   = ismember(Population2.objs,Population1.objs, 'rows');
                gamma = 1-sum(Lia)/Problem.N;            
                [Population1,FrontNo1] = EnvironmentalSelection(Population1,Problem.N,alpha);
                [Population2,FrontNo2] = EnvironmentalSelection_noCon(Population2,Problem.N,alpha,gamma,para);

                % Offspring Reproduction
                Population_all   = [Population1,Population2];
                RankSolution_all = [FrontNo1,FrontNo2];
                MatingPool = TournamentSelection(2,2*Problem.N,RankSolution_all);
                Offspring  = OperatorGAhalf(Problem,Population_all(MatingPool));

                % Environmental Selection
                alpha = 2./(1+exp(1).^(-Problem.FE*10/Problem.maxFE)) - 1; 
                para  = ceil(Problem.maxFE/Problem.N)/2 - ceil(Problem.FE/Problem.N);
                [Population1,~] = EnvironmentalSelection([Population1,Offspring],Problem.N,alpha);
                [Population2,~] = EnvironmentalSelection_noCon([Population2,Offspring],Problem.N,alpha,gamma,para);
            end
        end
    end
end
```
