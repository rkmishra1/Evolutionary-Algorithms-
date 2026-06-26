# DISK

**Tags**: <2025> <multi/many> <real/integer> <expensive>

## Description
Distribution-based Kriging-assisted evolutionary algorithm

## Reference
Z. Zhang, Y. Wang, G. Sun, and T. Pang. A distribution information based Kriging-assisted evolutionary algorithm for expensive many-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2656-2670.

## Source Code

### `DISK.m`
```matlab
classdef DISK < ALGORITHM
% <2025> <multi/many> <real/integer> <expensive>
% Distribution-based Kriging-assisted evolutionary algorithm
% wmax  --- 60 --- Generations of evolutionary search
% alpha ---  5 --- Number of selected candidates

%------------------------------- Reference --------------------------------
% Z. Zhang, Y. Wang, G. Sun, and T. Pang. A distribution information based 
% Kriging-assisted evolutionary algorithm for expensive many-objective 
% optimization problems. IEEE Transactions on Evolutionary Computation, 
% 2025, 29(6): 2656-2670.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            global NI mu K 
            [wmax,alpha] = Algorithm.ParameterSet(60,5);
            
            %% Initialization
            NI    = Problem.N;
            OP    = UniformPoint(NI,Problem.D,'Latin');
            A2    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*OP+repmat(Problem.lower,NI,1));
            A1    = A2;
            THETA = 5.*ones(Problem.M,Problem.D);
            Model = cell(1,Problem.M);
              
            while Algorithm.NotTerminated(A2)
                %% Surrogate Construction
                [Model,THETA] = model_train(A2,Model,THETA);
                
                %% Learning Distribution
                [F,~]  = NDSort(A2.objs,inf);
                PopDec = A2(F==1).decs;
                if size(PopDec,1) <= 1
                    PopDec = [PopDec; A2(F==2).decs];
                end
                mu = mean(PopDec,1);
                K  = (PopDec-mu)'*(PopDec-mu)/(size(PopDec,1)-1);
                  
                %% Evolutionary Search
                OP = optimizaiton(A1,wmax,Model,Problem);
                
                %% Candidate Selection
                C = NewSelect(OP,A2,alpha,Problem);
                
                %% Adaptive Exploration
                % Judgement
                flag = 0;
                if ~isempty(C)
                    flag = judgeLS(C,A2);
                    A2   = [A2,C];
                end
                if flag == 1
                    % Surrogate Construction
                    [Model,THETA] = model_train(A2,Model,THETA);
                    % Exploration
                    [W,ideal]     = IdentifyW(A2,Problem.N,Problem.M);
                    A2            = LocalSearch(OP,W,ideal,wmax,Model,A2,Problem);
                end
                
                %% Population Update
                index = EnvironmentalSelection(A2.objs,NI);
                A1    = A2(index);
            end
        end
    end
end

function [Model,THETA] = model_train(A2,Model,THETA)
    Dec     = A2.decs;
    Obj     = A2.objs;
    Len_dec = size(Dec,2);
    Len_obj = size(Obj,2);
    for i = 1 : Len_obj
        [~,distinct1] = unique(round(Dec*1e100)/1e100,'rows');
        [~,distinct2] = unique(round(Obj(:,i)*1e100)/1e100,'rows');
        distinct      = intersect(distinct1,distinct2);

        dmodel     = dacefit(Dec(distinct,:),Obj(distinct,i),'regpoly1','corrgauss',THETA(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
        Model{i}   = dmodel;
        THETA(i,:) = dmodel.theta;
    end
end

function [OffObj,Off_ObjMSE] = model_predict(Model,OffDec)
    N          = size(OffDec,1);
    Len_obj    = length(Model);
    OffObj     = zeros(N,Len_obj);
    Off_ObjMSE = zeros(N,Len_obj);

    for i = 1 : N
        for j = 1 : Len_obj
            [OffObj(i,j),~,Off_ObjMSE(i,j)] = predictor(OffDec(i,:),Model{j});
        end
    end
    OffObj     = real(OffObj);
    Off_ObjMSE = abs(real(Off_ObjMSE));
end

function P = optimizaiton(Population,wmax,Model,Problem)
    w      = 1;
    [N,~]  = size(Population.decs);
    P.decs = Population.decs;
    while w <= wmax    
        OffDec = OperatorGA(Problem,P.decs);
        P.decs = [P.decs;OffDec];
        [P.objs,P.objmse] = model_predict(Model,P.decs);
                
        index = SEnvironmentalSelection(P,N);  
        
        P.decs   = P.decs(index,:);
        P.objs   = P.objs(index,:);
        P.objmse = P.objmse(index,:);
             
        w = w + 1;
    end
end

function flag = judgeLS(C,A2)
    [F1,~] = NDSort(C.objs,1);
    AObj   = C(F1==1).objs;
    [F2,~] = NDSort(A2.objs,1);
    A2Obj  = A2(F2==1).objs;
    N1     = size(AObj,1);
    N2     = size(A2Obj,1);

    dominate = zeros(N1,N2);
    for i = 1 : N1
        for j = 1 : N2
            if all(AObj(i,:) <= A2Obj(j,:)) && ~all(AObj(i,:) == A2Obj(j,:))
                dominate(i,j) = true;
            end
        end
    end
    if any(any(dominate))
        flag = 0;
    else
        flag = 1;
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Next = EnvironmentalSelection(PopObj,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    %% Non-dominated sorting
    zmin   = min(PopObj);
    zmax   = max(PopObj);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next   = FrontNo < MaxFNo;
    Last   = find(FrontNo == MaxFNo);

    %% Select the solutions in the last front
    if MaxFNo == 1
        Del = Truncation(PopObj(Last,:),N);
        Next(Last(Del)) = true; 
    else
        Choose = Dist_Selection(PopObj(Next,:),PopObj(Last,:),N - sum(Next));
        Next(Last(Choose)) = true;
    end
end

function Choose = Dist_Selection(PopObj1,PopObj2,mu)
    PopObj = [PopObj1;PopObj2];
    N      = size(PopObj,1);
    N1     = size(PopObj1,1);
    
    %% Calculate the angle-based distance between each two solutions
    Distance = acos(1-pdist2(PopObj,PopObj,'cosine'));
    Distance(logical(eye(length(Distance)))) = inf;
    
    %% Calculate D
    Next1 = 1 : N1;
    Next2 = N1+1 : N;
    for i = 1 : mu
        Distance1 = sort(Distance(Next2,Next1),2);
        [~,index] = max(Distance1(:,1));
        Next1     = [Next1,Next2(index)];
        Next2(index) = [];
    end
    Choose = Next1(N1+1:end) - N1;
end

function Del = Truncation(PopObj,K)
    %% Select part of the solutions by truncation
    [N,~] = size(PopObj);
    
    %% Calculate the angle-based distance between each two solutions
    Distance = acos(1-pdist2(PopObj,PopObj,'cosine'));
    
    %% Truncation
    Distance(logical(eye(length(Distance)))) = inf;
    Del = true(1,N);
    while sum(Del) > K
        Remain   = find(Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = false;
    end
end
```

### `IdentifyW.m`
```matlab
function [W,ideal] = IdentifyW(DB,N,M)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    %% Preparing Data 
    V      = UniformPoint(10*N,M);
    A2Obj  = DB.objs;
    [F_,~] = NDSort(A2Obj,1);
    A2Obj  = A2Obj(F_==1,:);

    %% Translate Coordinate
    nadir = max(A2Obj,[],1);
    ideal = min(A2Obj,[],1);
    ideal = ideal - (nadir-ideal)/10 - 0.1*ones(1,M);
    A2Obj = A2Obj - ideal; 
    
    %% Calculate Angle between A2Obj and V
    Angle  = acos(1-pdist2(V,A2Obj,'cosine'));
    Angle_ = sort(Angle,2);
    index  = find(Angle_(:,1)==max(Angle_(:,1)));
    
    %% Identify the Farthest W
    if length(index) > 1
        index = index(randperm(length(index),1));
    end
    W = V(index,:);  
end
```

### `LocalSearch.m`
```matlab
function DB = LocalSearch(P,W,ideal,wmax,Model,DB,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    w     = 1;
    [N,~] = size(P.decs);
    while w <= wmax   
        OffDec1 = OperatorGA(Problem,P.decs);
        OffDec2 = OperatorDE_current_rand_1(Problem,P.decs);
        OffDec3 = OperatorDE_rand_1(Problem,P.decs);
        OffDec4 = OperatorDE_current_rand_1(Problem,P.decs);
        P.decs  = [P.decs;OffDec1;OffDec2;OffDec3;OffDec4];
        P.decs  = unique(P.decs,'rows');
        
        [P.objs,P.objmse] = model_predict(Model,P.decs);
        P.objmse = sqrt(P.objmse);
                
        fitness = max(abs(P.objs - ideal).*W,[],2) - 2*mean(P.objmse,2);       
        
        [~,Rank] = sort(fitness);    
        P.decs   = P.decs(Rank(1:N),:);
        P.objs   = P.objs(Rank(1:N),:);
        P.objmse = P.objmse(Rank(1:N),:);
        w = w + 1;
    end
    
    fitness  = max(abs(P.objs - ideal).*W,[],2) - 2*mean(P.objmse,2);
    [~,Rank] = sort(fitness);
    PopNew   = P.decs(Rank(1),:); 
    dist2    = pdist2(real(PopNew),real(DB.decs));
    if min(dist2) > 1e-50
        DB = [DB,Problem.Evaluation(PopNew)];
    end
end

function [OffObj,Off_ObjMSE] = model_predict(Model,OffDec)
    N          = size(OffDec,1);
    Len_obj    = length(Model);
    OffObj     = zeros(N,Len_obj);
    Off_ObjMSE = zeros(N,Len_obj);
   
    for i = 1 : N
        for j = 1 : Len_obj
            [OffObj(i,j),~,Off_ObjMSE(i,j)] = predictor(OffDec(i,:),Model{j});
        end
    end
    OffObj     = real(OffObj);
    Off_ObjMSE = abs(real(Off_ObjMSE));
end
```

### `NDSort_DIPD.m`
```matlab
function [FrontNo,MaxFNo] = NDSort_DIPD(PopDec,PopObj,ObjMSE,nSort)
% Do non-dominated sorting by distribution-based Pareto dominance (DIPD)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    global  mu K
    [N,~] = size(PopObj);
    [~,D] = size(PopDec);
    
    Pro   = zeros(N,1);
    for j = 1 : N
        Pro(j,:) = (1/(det(K)^(1/2)*(2*pi)^(D/2)))*exp(-0.5*(PopDec(j,:) - mu)*(K^-1)*(PopDec(j,:) - mu)');
    end
    
    sigma = sqrt(ObjMSE(reshape(ones(N,1)*(1:N),N*N,1),:) + repmat(ObjMSE,N,1));
    mean  = PopObj(reshape(ones(N,1)*(1:N),N*N,1),:) - repmat(PopObj,N,1);
    x_PD  = normcdf((0-mean)./sigma);
    y_PD  = 1 - x_PD;
    
    x_PD = - x_PD.*Pro(reshape(ones(N,1)*(1:N),N*N,1),:);
    y_PD = - y_PD.*repmat(Pro,N,1);
    
    dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if all(x_PD(N*(i-1)+j,:) <= y_PD(N*(i-1)+j,:)) && ~all(x_PD(N*(i-1)+j,:) == y_PD(N*(i-1)+j,:))
                dominate(i,j) = true;
            elseif all(x_PD(N*(i-1)+j,:) >= y_PD(N*(i-1)+j,:)) && ~all(x_PD(N*(i-1)+j,:) == y_PD(N*(i-1)+j,:))
                dominate(j,i) = true;
            end
        end
    end

    FrontNo = inf(1,N);
    MaxFNo  = 0;
    while sum(FrontNo~=inf) < min(nSort,N)
        MaxFNo                     = MaxFNo + 1;
        current                    = find(FrontNo==inf);
        dominate_                  = sum(dominate(current,current),1);
        index                      = find(dominate_==min(dominate_));
        FrontNo(current(index))    = MaxFNo;
        dominate(current(index),:) = false;
    end
end
```

### `NewSelect.m`
```matlab
function C = NewSelect(P,DB,alpha,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    
    %% Preparing Data
    C     = [];
    index = [];
    for i = 1 : size(P.decs,1)
        dist2 = pdist2(real(P.decs(i,:)),real(DB.decs));
        if min(dist2) > 1e-50
            index = [index,i];
        end
    end
    if length(index) <= alpha
       PopNew = P.decs(index,:);
       if ~isempty(PopNew)
           PopNew = Problem.Evaluation(PopNew);
           C      = [C,PopNew];
       end
       return; 
    end
    
    PopDec = P.decs(index,:);
    PopObj = P.objs(index,:);
    ObjMSE = P.objmse(index,:);

    A2Obj  = DB.objs;
    [F_,~] = NDSort(A2Obj,1);
    A2Obj  = unique(A2Obj(F_==1,:),'rows');
    
    zmin   = min([A2Obj;PopObj],[],1); zmax = max([A2Obj;PopObj],[],1);
    A2Obj  = (A2Obj - zmin)./max(zmax - zmin,10e-10);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    ObjMSE = ObjMSE./(max(zmax - zmin,10e-10).^2);
    
    %% Selection1
    [FrontNo,~] = NDSort_DIPD(PopDec,PopObj,ObjMSE,1);
    PopDec      = PopDec(FrontNo==1,:);
    PopObj      = PopObj(FrontNo==1,:);
    ObjMSE      = ObjMSE(FrontNo==1,:);
    
    if size(PopDec,1) <= alpha
        C = [C,Problem.Evaluation(PopDec)];
        return;
    end
    
    %% Selection2
    Pindex = true(1,size(PopObj,1));
    while length(find(Pindex==0)) < alpha 
        Last     = find(Pindex==1);
        Dis      = Distance(PopObj(Last,:),A2Obj);
        [~,Rank] = sort(Dis,'descend');
        PopNew   = PopDec(Last(Rank(1)),:);
        C        = [C,Problem.Evaluation(PopNew)];
        A2Obj    = [DB.objs;C.objs]; 
        [F_P,~]  = NDSort(A2Obj,1);
        A2Obj    = unique(A2Obj(F_P==1,:),'rows');
        A2Obj    = (A2Obj - zmin)./max(zmax - zmin,10e-10);
        Pindex(Last(Rank(1))) = 0;
    end
end

function dis = Distance(PopObj,OffObj)
    %% Calculate the angle-based distance between each two solutions
    dis = acos(1-pdist2(PopObj,OffObj,'cosine'));
    
    %% Calculate D
    dis = sort(dis,2);
    dis = dis(:,1);
end
```

### `OperatorDE_current_rand_1.m`
```matlab
function Offspring = OperatorDE_current_rand_1(Problem,Parent)
% DE/current-to-rand/1

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    %% Parameter setting
    [N,D] = size(Parent);
    Fm    = [0.6,0.8,1.0];
    CRm   = [0.1,0.2,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));
    index = randi([1,length(CRm)],N,1);
    CR    = CRm(index);
    CR    = CR';
    
    %% Differental evolution
    Site            = rand(N,D) < CR;
    Parent1         = Parent(randperm(N),:);
    Parent2         = Parent(randperm(N),:);
    Parent3         = Parent(randperm(N),:);
    Offspring       = Parent;
    Offspring(Site) = Parent(Site) + F(Site).*(Parent1(Site)-Parent(Site)) + F(Site).*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    proM  = 1;
    disM  = 20;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end
```

### `OperatorDE_rand_1.m`
```matlab
function Offspring = OperatorDE_rand_1(Problem,Parent)
% DE/rand/1

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    %% Parameter setting
    [N,D] = size(Parent);
    Fm    = [0.6,0.8,1.0];
    CRm   = [0.1,0.2,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));
    index = randi([1,length(CRm)],N,1);
    CR    = CRm(index);
    CR    = CR';
    
    %% Differental evolution
    Site            = rand(N,D) < CR;
    Parent1         = Parent(randperm(N),:);
    Parent2         = Parent(randperm(N),:);
    Parent3         = Parent(randperm(N),:);
    Offspring       = Parent;
    Offspring(Site) = Parent1(Site) + F(Site).*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    proM  = 1;
    disM  = 20;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end
```

### `SEnvironmentalSelection.m`
```matlab
function Next = SEnvironmentalSelection(P,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: z.zhang0@csu.edu.cn)

    PopDec = P.decs;
    PopObj = P.objs;
    PopMSE = P.objmse;

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort_DIPD(PopDec,PopObj,PopMSE,N);
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    
    zmin   = min(PopObj,[],1);zmax = max(PopObj,[],1);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    
    %% Select the solutions in the last front
    if MaxFNo == 1
        Del  = Truncation(PopObj(Last,:),N);
        Next(Last(Del)) = true; 
    else
        Choose = Dist_Selection(PopObj(Next,:),PopObj(Last,:),N - sum(Next));
        Next(Last(Choose)) = true;
    end
end

function Choose = Dist_Selection(PopObj1,PopObj2,mu)
    PopObj = [PopObj1;PopObj2];
    N      = size(PopObj,1);
    N1     = size(PopObj1,1);
    
    %% Calculate the angle-based distance between each two solutions
    Distance = acos(1-pdist2(PopObj,PopObj,'cosine'));
    Distance(logical(eye(length(Distance)))) = inf;
    
    %% Calculate D
    Next1 = 1 : N1;
    Next2 = N1+1 : N;
    for i = 1 : mu
        Distance1 = sort(Distance(Next2,Next1),2);
        [~,index] = max(Distance1(:,1));
        Next1     = [Next1,Next2(index)];
        Next2(index) = [];
    end
    Choose = Next1(N1+1:end) - N1;
end

function Del = Truncation(PopObj,K)
    %% Select part of the solutions by truncation
    [N,~] = size(PopObj);
    
    %% Calculate the angle-based distance between each two solutions
    Distance = acos(1-pdist2(PopObj,PopObj,'cosine'));

    %% Truncation
    Distance(logical(eye(length(Distance)))) = inf;
    Del = true(1,N);
    while sum(Del) > K
        Remain   = find(Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = false;
    end
end
```

### `dacefit.m`
```matlab
function  [dmodel,perf] = dacefit(S,Y,regr,corr,theta0,lob,upb)
%dacefit - Constrained non-linear least-squares fit of a given correlation
%model to the provided data set and regression model
%
% Call
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0)
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
%
% Input
% S, Y    : Data points (S(i,:), Y(i,:)), i = 1,...,m
% regr    : Function handle to a regression model
% corr    : Function handle to a correlation function
% theta0  : Initial guess on theta, the correlation function parameters
% lob,upb : If present, then lower and upper bounds on theta
%           Otherwise, theta0 is used for theta
%
% Output
% dmodel  : DACE model: a struct with the elements
%    regr   : function handle to the regression model
%    corr   : function handle to the correlation function
%    theta  : correlation function parameters
%    beta   : generalized least squares estimate
%    gamma  : correlation factors
%    sigma2 : maximum likelihood estimate of the process variance
%    S      : scaled design sites
%    Ssc    : scaling factors for design arguments
%    Ysc    : scaling factors for design ordinates
%    C      : Cholesky factor of correlation matrix
%    Ft     : Decorrelated regression matrix
%    G      : From QR factorization: Ft = Q*G' .
%    perf   : struct with performance information. Elements
%    nv     : Number of evaluations of objective function
%    perf   : (q+2)*nv array, where q is the number of elements 
%             in theta, and the columns hold current values of
%                 [theta;  psi(theta);  type]
%             |type| = 1, 2 or 3, indicate 'start', 'explore' or 'move'
%             A negative value for type indicates an uphill step

% hbn@imm.dtu.dk  
% Last update September 3, 2002

    % Check design points
    [m,n] = size(S);  % number of design sites and their dimension
    sY    = size(Y);
    if min(sY) == 1
        Y = Y(:);  
        lY  = max(sY);  
    else       
        lY  = sY(1);
    end
    if m ~= lY
        error('S and Y must have the same number of rows')
    end
    % Check correlation parameters if it is given
    lth = length(theta0);
    if nargin > 5  % optimization case
        if length(lob) ~= lth || length(upb) ~= lth
            error('theta0, lob and upb must have the same length')
        end
        if any(lob <= 0) || any(upb < lob)
            error('The bounds must satisfy  0 < lob <= upb')
        end
    else  % given theta
        if any(theta0 <= 0)
            error('theta0 must be strictly positive')
        end
    end
    % Normalize data
    mS = mean(S);   sS = std(S);
    mY = mean(Y);   sY = std(Y);
    % 02.08.27: Check for 'missing dimension'
    j = find(sS == 0);
    if ~isempty(j)
        sS(j) = 1;
    end
    j = find(sY == 0);
    if  ~isempty(j)
        sY(j) = 1;
    end
    S = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);
    Y = (Y - repmat(mY,m,1)) ./ repmat(sY,m,1);
    % Calculate distances D between points
    mzmax = m*(m-1) / 2;        % number of non-zero distances
    ij    = zeros(mzmax, 2);  	% initialize matrix with indices
    D     = zeros(mzmax, n);  	% initialize matrix with distances
    LL    = 0;
    for k = 1 : m-1
        LL       = LL(end) + (1 : m-k);
        ij(LL,:) = [repmat(k,m-k,1) (k+1:m)']; % indices for sparse matrix
        D(LL,:)  = repmat(S(k,:),m-k,1)-S(k+1:m,:); % differences between points
    end
%     if min(sum(abs(D),2) ) == 0
%         error('Multiple design sites are not allowed')
%     end
    % Regression matrix
    F      = feval(regr, S);  
    [mF,p] = size(F);
    if mF ~= m
        error('number of rows in  F  and  S  do not match')
    end
    if p > mF 
        error('least squares problem is underdetermined')
    end
    % parameters for objective function
    par = struct('corr',corr,'regr',regr,'y',Y,'F',F,'D',D,'ij',ij,'scS',sS);
    % Determine theta
    if nargin > 5
        % Bound constrained non-linear optimization
        [theta, f, fit, perf] = boxmin(theta0, lob, upb, par);
        if  isinf(f)
            error('Bad parameter region.  Try increasing  upb')
        end
    else
        % Given theta
        theta   = theta0(:);   
        [f,fit] = objfunc(theta, par);
        perf    = struct('perf',[theta; f; 1], 'nv',1);
        if  isinf(f)
            error('Bad point.  Try increasing theta0')
        end
    end
    % Return values
    dmodel = struct('regr',regr,'corr',corr,'theta',theta.','beta',fit.beta,...
                    'gamma',fit.gamma,'sigma2',sY.^2.*fit.sigma2,'S',S,'Ssc',[mS; sS],...
                    'Ysc',[mY; sY],'C',fit.C,'Ft',fit.Ft,'G',fit.G);
end

function  [obj, fit] = objfunc(theta, par)
    % Initialize
    obj = inf; 
    fit = struct('sigma2',NaN,'beta',NaN,'gamma',NaN,'C',NaN,'Ft',NaN,'G',NaN);
    m   = size(par.F,1);
    % Set up  R
    r   = feval(par.corr, theta, par.D);
    idx = find(r > 0);   o = (1 : m)';   
    mu  = (10+m)*eps;
    R   = sparse([par.ij(idx,1); o],[par.ij(idx,2); o],[r(idx); ones(m,1)+mu]);  
    % Cholesky factorization with check for pos. def.
    [C,rd] = chol(R);
    if rd
        return;
    end
    % Get least squares solution
    C     = C';
    Ft    = C \ par.F;
    [Q,G] = qr(Ft,0);
    if rcond(G) < 1e-10
        % Check   F  
        if cond(par.F) > 1e15 
            error('F is too ill conditioned\nPoor combination of regression model and design sites')
        else  % Matrix  Ft  is too ill conditioned
            return 
        end 
    end
    Yt   = C \ par.y;
    beta = G \ (Q'*Yt);
    rho  = Yt - Ft*beta;  sigma2 = sum(rho.^2)/m;
    detR = prod( full(diag(C)) .^ (2/m) );
    obj  = sum(sigma2) * detR;
    if nargout > 1
        fit = struct('sigma2',sigma2,'beta',beta,'gamma',rho'/C,'C',C,'Ft',Ft,'G',G');
    end
end

function  [t,f,fit,perf] = boxmin(t0,lo,up,par)
%BOXMIN  Minimize with positive box constraints

    % Initialize
    [t, f, fit, itpar] = start(t0, lo, up, par);
    if  ~isinf(f)
        % Iterate
        p = length(t);
        if  p <= 2
            kmax = 2;
        else
            kmax = min(p,4);
        end
        for k = 1 : kmax
            th = t;
            [t, f, fit, itpar] = explore(t, f, fit, itpar, par);
            [t, f, fit, itpar] = move(th, t, f, fit, itpar, par);
        end
    end
    perf = struct('nv',itpar.nv, 'perf',itpar.perf(:,1:itpar.nv));
end

function [t,f,fit,itpar] = start(t0,lo,up,par)
% Get starting point and iteration parameters

    % Initialize
    t  = t0(:);
    lo = lo(:);
    up = up(:);
    p  = length(t);
    D  = 2 .^((1:p)'/(p+2));
    ee = find(up == lo);  % Equality constraints
    if ~isempty(ee)
        D(ee) = ones(length(ee),1);
        t(ee) = up(ee); 
    end
    ng = find(t < lo | up < t);  % Free starting values
    if ~isempty(ng)
        t(ng) = (lo(ng) .* up(ng).^7).^(1/8);  % Starting point
    end
    ne = find(D ~= 1);
    % Check starting point and initialize performance info
    [f,fit] = objfunc(t,par);
    nv      = 1;
    itpar   = struct('D',D,'ne',ne,'lo',lo,'up',up,'perf',zeros(p+2,200*p),'nv',1);
    itpar.perf(:,1) = [t; f; 1];
    if isinf(f)    % Bad parameter region
        return
    end
    if length(ng) > 1  % Try to improve starting guess
        d0 = 16;  d1 = 2;   q = length(ng);
        th = t;   fh = f;   jdom = ng(1);  
        for k = 1 : q
            j  = ng(k);
            fk = fh;
            tk = th;
            DD = ones(p,1);  DD(ng) = repmat(1/d1,q,1);  DD(j) = 1/d0;
            alpha = min(log(lo(ng) ./ th(ng)) ./ log(DD(ng))) / 5;
            v = DD .^ alpha;
            for rept = 1 : 4
                tt = tk .* v; 
                [ff, fitt] = objfunc(tt,par);  nv = nv+1;
                itpar.perf(:,nv) = [tt; ff; 1];
                if ff <= fk 
                    tk = tt;
                    fk = ff;
                    if  ff <= f
                        t   = tt;
                        f   = ff;
                        fit = fitt;
                        jdom = j;
                    end
                else
                    itpar.perf(end,nv) = -1;
                    break
                end
            end
        end % improve
        % Update Delta  
        if  jdom > 1
            D([1 jdom]) = D([jdom 1]); 
            itpar.D = D;
        end
    end % free variables
    itpar.nv = nv;
end

function [t,f,fit,itpar] = explore(t,f,fit,itpar,par)
% Explore step

    nv = itpar.nv;
    ne = itpar.ne;
    for k = 1 : length(ne)
        j  = ne(k);
        tt = t;
        DD = itpar.D(j);
        if t(j) == itpar.up(j)
            atbd  = 1;
            tt(j) = t(j) / sqrt(DD);
        elseif t(j) == itpar.lo(j)
            atbd  = 1;
            tt(j) = t(j) * sqrt(DD);
        else
            atbd  = 0;
            tt(j) = min(itpar.up(j), t(j)*DD);
        end
        [ff,fitt] = objfunc(tt,par);
        nv = nv+1;
        itpar.perf(:,nv) = [tt; ff; 2];
        if ff < f
            t   = tt;
            f   = ff;
            fit = fitt;
        else
            itpar.perf(end,nv) = -2;
            if ~atbd  % try decrease
                tt(j) = max(itpar.lo(j), t(j)/DD);
                [ff,fitt] = objfunc(tt,par);
                nv = nv+1;
                itpar.perf(:,nv) = [tt; ff; 2];
                if ff < f
                    t   = tt;
                    f   = ff;
                    fit = fitt;
                else
                    itpar.perf(end,nv) = -2;
                end
            end
        end
    end
    itpar.nv = nv;
end

function [t,f,fit,itpar] = move(th,t,f,fit,itpar,par)
% Pattern move

    nv = itpar.nv;
    p  = length(t);
    v  = t ./ th;
    if  all(v == 1)
        itpar.D = itpar.D([2:p 1]).^.2;
        return;
    end
    % Proper move
    rept = 1;
    while  rept
        tt = min(itpar.up, max(itpar.lo, t .* v));  
        [ff,fitt] = objfunc(tt,par); 
        nv = nv+1;
        itpar.perf(:,nv) = [tt; ff; 3];
        if  ff < f
            t   = tt;
            f   = ff;
            fit = fitt;
            v   = v .^ 2;
        else
            itpar.perf(end,nv) = -3;
            rept = 0;
        end
        if any(tt == itpar.lo | tt == itpar.up)
            rept = 0;
        end
    end
    itpar.nv = nv;
    itpar.D  = itpar.D([2:p 1]).^.25;
end

function [r,dr] = corrgauss(theta,d)
%CORRGAUSS  Gaussian correlation function,

    [m,n] = size(d);  % number of differences and dimension of data
    if length(theta) == 1
        theta = repmat(theta,1,n);
    elseif length(theta) ~= n
        error('Length of theta must be 1 or %d',n)
    end
    td = d.^2 .* repmat(-theta(:).',m,1);
    r  = exp(sum(td, 2));
	dr = repmat(-2*theta(:).',m,1) .* d .* repmat(r,1,n);
end

function [f,df] = regpoly0(S)
%REGPOLY0  Zero order polynomial regression function

    f  = ones(size(S,1),1);
	df = zeros(size(S,2),1);
end

function [f,df] = regpoly1(S)
%REGPOLY1  First order polynomial regression function

    f  = [ones(size(S,1),1),S];
	df = [zeros(size(S,2),1),eye(size(S,2))];
end
```

### `predictor.m`
```matlab
function [y,or1,or2,dmse] = predictor(x,dmodel)
%PREDICTOR  Predictor for y(x) using the given DACE model.
%
% Call:   y = predictor(x, dmodel)
%         [y, or] = predictor(x, dmodel)
%         [y, dy, mse] = predictor(x, dmodel) 
%         [y, dy, mse, dmse] = predictor(x, dmodel) 
%
% Input
% x      : trial design sites with n dimensions.  
%          For mx trial sites x:
%          If mx = 1, then both a row and a column vector is accepted,
%          otherwise, x must be an mx*n matrix with the sites stored
%          rowwise.
% dmodel : Struct with DACE model; see DACEFIT
%
% Output
% y    : predicted response at x.
% or   : If mx = 1, then or = gradient vector/Jacobian matrix of predictor
%        otherwise, or is an vector with mx rows containing the estimated
%                   mean squared error of the predictor
% Three or four results are allowed only when mx = 1,
% dy   : Gradient of predictor; column vector with  n elements
% mse  : Estimated mean squared error of the predictor;
% dmse : Gradient vector/Jacobian matrix of mse

% hbn@imm.dtu.dk
% Last update August 26, 2002
 
    or1 = NaN; or2 = NaN; dmse = NaN;	% Default return values
    if isnan(dmodel.beta)
        error('DMODEL has not been found')
    end
    [m,n] = size(dmodel.S);     % number of design sites and number of dimensions
    sx    = size(x);            % number of trial sites and their dimension
    if min(sx) == 1 && n > 1    % Single trial point 
        nx = max(sx);
        if nx == n 
            mx = 1;
            x  = x(:).';
        end
    else
        mx = sx(1);
        nx = sx(2);
    end
    if nx ~= n
        error('Dimension of trial sites should be %d',n)
    end
    % Normalize trial sites  
    x = (x - repmat(dmodel.Ssc(1,:),mx,1)) ./ repmat(dmodel.Ssc(2,:),mx,1);
    q = size(dmodel.Ysc,2);  % number of response functions
    if mx == 1  % one site only
        dx = repmat(x,m,1) - dmodel.S;  % distances to design sites
        if nargout > 1                  % gradient/Jacobian wanted
            [f,df] = feval(dmodel.regr, x);
            [r,dr] = feval(dmodel.corr, dmodel.theta, dx);
            % Scaled Jacobian
            dy = (df * dmodel.beta).' + dmodel.gamma * dr;
            % Unscaled Jacobian
            or1 = dy .* repmat(dmodel.Ysc(2, :)', 1, nx) ./ repmat(dmodel.Ssc(2,:), q, 1);
            if q == 1
                % Gradient as a column vector
                or1 = or1';
            end
            if nargout > 2  % MSE wanted
                rt = dmodel.C \ r;
                u = dmodel.Ft.' * rt - f.';
                v = dmodel.G \ u;
                or2 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + sum(v.^2) - sum(rt.^2))',1,q);
                if nargout > 3  % gradient/Jacobian of MSE wanted
                    % Scaled gradient as a row vector
                    Gv = dmodel.G' \ v;
                    g = (dmodel.Ft * Gv - rt)' * (dmodel.C \ dr) - (df * Gv)';
                    % Unscaled Jacobian
                    dmse = repmat(2 * dmodel.sigma2',1,nx) .* repmat(g ./ dmodel.Ssc(2,:),q,1);
                    if q == 1
                    % Gradient as a column vector
                    dmse = dmse';
                    end
                end
            end
        else  % predictor only
            f = feval(dmodel.regr, x);
            r = feval(dmodel.corr, dmodel.theta, dx);
        end
        % Scaled predictor
        sy = f * dmodel.beta + (dmodel.gamma*r).';
        % Predictor
        y = (dmodel.Ysc(1,:) + dmodel.Ysc(2,:) .* sy)';
	else  % several trial sites
        % Get distances to design sites  
        dx = zeros(mx*m,n);
        kk = 1 : m;
        for k = 1 : mx
            dx(kk,:) = repmat(x(k,:),m,1) - dmodel.S;
            kk = kk + m;
        end
        % Get regression function and correlation
        f = feval(dmodel.regr, x);
        r = feval(dmodel.corr, dmodel.theta, dx);
        r = reshape(r, m, mx);
        % Scaled predictor 
        sy = f * dmodel.beta + (dmodel.gamma * r).';
        % Predictor
        y = repmat(dmodel.Ysc(1,:),mx,1) + repmat(dmodel.Ysc(2,:),mx,1) .* sy;
        if nargout > 1	% MSE wanted
            rt  = dmodel.C \ r;
            u   = dmodel.G \ (dmodel.Ft.' * rt - f.');
            or1 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + sum(u.^2,1) - sum(rt.^2,1))',1,q);
            if  nargout > 2
                disp('WARNING from PREDICTOR.  Only  y  and  or1=mse  are computed')
            end
        end
    end
end

function [r,dr] = corrgauss(theta,d)
%CORRGAUSS  Gaussian correlation function,

    [m,n] = size(d);  % number of differences and dimension of data
    if length(theta) == 1
        theta = repmat(theta,1,n);
    elseif length(theta) ~= n
        error('Length of theta must be 1 or %d',n)
    end
    td = d.^2 .* repmat(-theta(:).',m,1);
    r  = exp(sum(td, 2));
	dr = repmat(-2*theta(:).',m,1) .* d .* repmat(r,1,n);
end

function [f,df] = regpoly0(S)
%REGPOLY0  Zero order polynomial regression function

    f  = ones(size(S,1),1);
	df = zeros(size(S,2),1);
end

function [f,df] = regpoly1(S)
%REGPOLY1  First order polynomial regression function

    f  = [ones(size(S,1),1),S];
	df = [zeros(size(S,2),1),eye(size(S,2))];
end
```
