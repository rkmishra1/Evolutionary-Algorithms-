# S-ECSO

**Tags**: <2022> <multi> <real> <large/none> <sparse>

## Description
Enhanced competitive swarm optimizer for sparse optimization

## Reference
X. Wang, K. Zhang, J. Wang, and Y. Jin. An enhanced competitive swarm optimizer with strongly convex sparse operator for large-scale multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 859-871.

## Source Code

### `A_get.m`
```matlab
function [AA,gBest] = A_get(Problem,Population,A,iter)
% The updating strategy of archive(A) in S-ECSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    x = Population.decs;
    xObj_fitness = Population.objs;
    if iter == 1
        A       = [];
        Adecs   = [];
        value_A = [];
    else
        Adecs   = A.decs;
        value_A = A.objs;
    end

    %% update A
    Compare_Obj = [value_A;xObj_fitness];
    compare_x   = [Adecs;x];
    CC          = [A,Population];

    [FrontNo,~]   = NDSort(Compare_Obj,Inf);
    FrontNo_index = find(FrontNo == 1);

    Adecs = compare_x(FrontNo_index,:);
    AA    = CC(FrontNo_index);


    [Adecs,index] = unique(Adecs,'rows');
    AA            = AA(index);

    %% adding perturbation
    ELS_A = AA.decs;
    for i = 1 : size(Adecs,1)
        j = ceil(rand*size(Adecs,2));
        ELS_A(i,j) = ELS_A(i,j)+ (Problem.upper(j)-Problem.lower(j))*normrnd(0,1);
        ELS_A(i,j) = min(max(ELS_A(i,j),Problem.lower(j)),Problem.upper(j));
    end

    ELS_A1 = Problem.Evaluation(ELS_A);

    %%  truncating (using SPEA2)
    AA    = [AA,ELS_A1];
    row_A = size(AA.decs,1);
    if row_A > Problem.N
        [FrontNo,MaxFNo] = NDSort(AA.objs,Problem.N);
        Next = false(1,length(FrontNo));
        Next(FrontNo<MaxFNo) = true;
        PopObj = AA.objs;
        fmax   = max(PopObj(FrontNo==1,:),[],1);
        fmin   = min(PopObj(FrontNo==1,:),[],1);
        PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);

        %% Select the solutions in the last front
        Last = find(FrontNo==MaxFNo);
        del  = Truncation(PopObj(Last,:),length(Last)-Problem.N+sum(Next));
        Next(Last(~del)) = true;
        AA = AA(Next);
    end

    %% gBest
    [M,~]     = size(AA.decs);
    r         = rand(1,size(AA.objs,2));
    r_matr    = repmat(r,M,1);
    f_gBest   = sum(r_matr.*AA.objs,2)/sum(r);
    [~,index] = min(f_gBest);
    B         = AA.decs;
    gBest     = B(index,:);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation
 
    N = size(PopObj,1);

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,N);
    while sum(Del) < K 
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `ECSO.m`
```matlab
function [Population,v] = ECSO(Problem,Population,v,gBest,subswarm_index)
% The particle updating strategy of S-ECSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    w  = 0.7968;
    c1 = 1.4962;
    c2 = 1.4962;
    x  = Population.decs;
    Obj_fitness = Population.objs;

    %% subswarm
    subswarm1_x = x(subswarm_index(1):subswarm_index(2)-1,:);
    subswarm2_x = x(subswarm_index(2):subswarm_index(3)-1,:);
    subswarm3_x = x(subswarm_index(3):size(x,1),:);

    [Front_rank1,~] = NDSort(Obj_fitness(subswarm_index(1):subswarm_index(2)-1,:),inf);
    subswa     = x(find(Front_rank1==1),:);
    lbest(1,:) = subswa(randi(size(subswa,1)),:);
    llBest(subswarm_index(1):subswarm_index(2)-1,:) = repmat(lbest(1,:),size(subswarm1_x,1),1);

    [Front_rank2,~] = NDSort(Obj_fitness(subswarm_index(2):subswarm_index(3)-1,:),inf);
    subswa     = x(find(Front_rank2==1)+subswarm_index(2)-1,:);
    lbest(2,:) = subswa(randi(size(subswa,1)),:);
    llBest(subswarm_index(2):subswarm_index(3)-1,:) = repmat(lbest(2,:),size(subswarm2_x,1),1);

    [Front_rank3,~] = NDSort(Obj_fitness(subswarm_index(3):size(x,1),:),inf);
    subswa     = x(find(Front_rank3==1)+subswarm_index(3)-1,:);
    lbest(3,:) = subswa(randi(size(subswa,1)),:);
    llBest(subswarm_index(3):size(x,1),:) = repmat(lbest(3,:),size(subswarm3_x,1),1);

    randswarm1 = randperm(size(subswarm1_x,1));
    randswarm2 = randperm(size(subswarm2_x,1)) + size(subswarm1_x,1);
    randswarm3 = randperm(size(subswarm3_x,1)) + size(subswarm1_x,1) + size(subswarm2_x,1);

    sub_pair_size = floor(size(x,1)/3);

    [Front_rank,~] = NDSort(Obj_fitness,inf);
    
    %% Compare
    for i = 1 : sub_pair_size
        randswarm   = [randswarm1(i),randswarm2(i),randswarm3(i)];
        R_randswarm = Front_rank(randswarm);
        [~,rrank]   = min(R_randswarm);
        if rrank == 1
            winner_index = randswarm1(i);
            loser1_index = randswarm2(i);
            loser2_index = randswarm3(i);
        elseif rrank == 2
            winner_index = randswarm2(i);
            loser1_index = randswarm1(i);
            loser2_index = randswarm3(i);
        elseif rrank == 3
            winner_index = randswarm3(i);
            loser1_index = randswarm1(i);
            loser2_index = randswarm2(i);
        end

        %% update
        if rand < 0.5
            v(loser1_index,:) = rand*v(loser1_index,:)+rand*(x(winner_index,:)-x(loser1_index,:));
            x(loser1_index,:) = x(loser1_index,:) + v(loser1_index,:);

            v(loser2_index,:) = w*v(loser2_index,:) + c1*rand*(llBest(loser2_index,:)-x(loser2_index,:))...
                + c2*rand*(gBest-x(loser2_index,:));
            x(loser2_index,:) = x(loser2_index,:) + v(loser2_index,:);
        else
            v(loser2_index,:) = rand*v(loser2_index,:)+rand*(x(winner_index,:)-x(loser2_index,:));
            x(loser2_index,:) = x(loser2_index,:) + v(loser2_index,:);

            v(loser1_index,:) = w*v(loser1_index,:) + c1*rand*(llBest(loser1_index,:)-x(loser1_index,:))...
                + c2*rand*(gBest-x(loser1_index,:));
            x(loser1_index,:) = x(loser1_index,:) + v(loser1_index,:);
        end
    end
    
    %% Restrict the range
    N = size(x,1);
    for irange = 1 : N
        Upper_flag   = Problem.upper<x(irange,:);
        Upper_flag_T = sum(Upper_flag);
        if Upper_flag_T > 0
            Upper_index = find(Upper_flag == 1);
            x(irange,Upper_index) = Problem.upper(Upper_index);
        end

        Low_flag   = Problem.lower > x(irange,:);
        Low_flag_T = sum(Low_flag);
        if Low_flag_T > 0
            Low_index = find(Low_flag == 1);
            x(irange,Low_index) = Problem.lower(Low_index);
        end
    end
    
    %% Calculate
    Population = Problem.Evaluation(x);
end
```

### `SCSparse.m`
```matlab
function Population = SCSparse(Problem,Population,L)
% The sparse operator of S-ECSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    x = Population.decs;
    
    %% Make the population sparse
    [row_x,col_x] = size(x);
    new_x = [];
    for i = 1 : row_x
        new_x(i,:) = zeros(1,col_x);
        flag_T     = L < abs(x(i,:));
        new_x(i,flag_T) = x(i,flag_T)- L(flag_T).*sign(x(i,flag_T));
    end

    %% Restrict the range
    x_maxmin = (Problem.upper-Problem.lower);
    for irange = 1 : row_x
        Upper_flag   = Problem.upper<new_x(irange,:);
        Upper_flag_T = sum(Upper_flag);
        if Upper_flag_T > 0
            Upper_index = find(Upper_flag == 1);
            new_x(irange,Upper_index) = rand(1,size(Upper_index,2)).*x_maxmin(Upper_index);
        end

        Low_flag   = Problem.lower > new_x(irange,:);
        Low_flag_T = sum(Low_flag);
        if Low_flag_T > 0
            Low_index = find(Low_flag == 1);
            new_x(irange,Low_index) = rand(1,size(Low_index,2)).*x_maxmin(Low_index);
        end
    end

    %% Calculate obj
    new_x = Problem.Evaluation(new_x);
    XX    = [Population, new_x];
    
    %% Save the better solutions
    N = length(new_x);
    [FrontNo,~] = NDSort([XX.objs],Inf);
    for i = 1 : N
        if FrontNo(i+N) <= FrontNo(i)
            Population(i) =  XX(i+N);
        end
    end
end
```

### `SECSO.m`
```matlab
classdef SECSO < ALGORITHM
% <2022> <multi> <real> <large/none> <sparse>
% Enhanced competitive swarm optimizer for sparse optimization
    
%------------------------------- Reference --------------------------------
% X. Wang, K. Zhang, J. Wang, and Y. Jin. An enhanced competitive swarm
% optimizer with strongly convex sparse operator for large-scale
% multi-objective optimization. IEEE Transactions on Evolutionary
% Computation, 2022, 26(5): 859-871.
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
            %% Parameter setting
            % Used in ECSO
            subswarm       = floor(Problem.N/3);         
            subswarm_index = [1,1+subswarm,Problem.N-subswarm];
            % Used in SCSparse 
            [LMax,LMin] = Algorithm.ParameterSet(0.35,0);
            step        = (LMax-LMin).*(Problem.upper-Problem.lower)/((Problem.maxFE/Problem.N/3)-1);
            lamb        = LMax.*(Problem.upper-Problem.lower);

            %% Population initialization
            x = rand(Problem.N,Problem.D);
            v = rand(Problem.N,Problem.D);
            x = x.*(repmat(Problem.upper,Problem.N,1)-repmat(Problem.lower,Problem.N,1)) + repmat(Problem.lower,Problem.N,1);
            v = v.*(repmat(Problem.upper,Problem.N,1)-repmat(Problem.lower,Problem.N,1)) + repmat(Problem.lower,Problem.N,1);
            Population  = Problem.Evaluation(x);
            Population1 = Population; % 'Population1'is the population(x), 'Population' is A in the paper
            
            %% Optimization
            iter = 0;
            while Algorithm.NotTerminated(Population)
                iter = iter + 1;
                if iter > Problem.maxFE/Problem.N/3-1
                    Problem.FE = Problem.maxFE;
                end
                [Population1]      = SCSparse(Problem,Population1,lamb);
                [Population,gBest] = A_get(Problem,Population1,Population,iter);
                [Population1,v]    = ECSO(Problem,Population1,v,gBest,subswarm_index);
                lamb = lamb - step;     % Update lamb in SCSparse
            end
        end
    end
end
```

### `gBest_get.m`
```matlab
function gBest = gBest_get(AA)
% The gBest (global best) updating strategy of S-ECSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [M,~]     = size(AA.decs);
    r         = rand(1,size(AA.objs,2));
    r_matr    = repmat(r,M,1);
    f_gBest   = sum(r_matr.*AA.objs,2)/sum(r);
    [~,index] = min(f_gBest);
    B         = AA.decs;
    gBest     = B(index,:);
end
```
