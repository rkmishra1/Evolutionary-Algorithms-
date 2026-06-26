# TEA

**Tags**: <2024> <multi/many> <real/integer> <expensive> <constrained/none>

## Description
Two-phase evolutionary algorithm

## Reference
Z. Zhang, Y. Wang, J. Liu, G. Sun, and K. Tang. A two-phase Kriging- assisted evolutionary algorithm for expensive constrained multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(8): 4579-4591.

## Source Code

### `Candi_Select.m`
```matlab
function C = Candi_Select(PopDec,PopObj,PopCon,ObjMSE,ConMSE,Database,mu)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao_zhang0@163.com)

    index = ismember(PopDec,Database.decs,'rows');
    if sum(index) == size(PopDec,1)
        C_ = [];
    else
        index_ = Selection(PopObj(~index,:),ObjMSE(~index,:),PopCon(~index,:),ConMSE(~index,:),Database,mu);
        index  = find(~index);
        C_     = PopDec(index(index_),:);
    end
    C = [];
    for i = 1 : size(C_,1)
        dist2 = pdist2(real(C_(i,:)),real(Database.decs));
        if min(dist2) > 1e-5
            C = [C;C_(i,:)];
        end
    end
end

function index = EucDistance_Select(PopObj,ALL_Obj)
    N1 = size(PopObj,1);
    N2 = size(ALL_Obj,1);
    Distance = zeros(N1,N2);

    for i = 1 : N1
        for j = 1 : N2
            Distance(i,j) = norm(PopObj(i,:)-ALL_Obj(j,:),2);    
        end
    end
    
    Distance  = sort(Distance,2);
    Distance  = Distance(:,1);
    [~,index] = max(Distance);
end

function Next = Selection(PopObj,ObjMSE,PopCon,ConMSE,Database,mu)
    %% Preparing Data
    ALL_Obj = Database.objs;
    ALL_Con = Database.cons;
    zmin    = min([ALL_Obj;PopObj]);zmax = max([ALL_Obj;PopObj]);
    ALL_Obj = (ALL_Obj - zmin )./max(zmax - zmin,10e-10);
    PopObj  = (PopObj - zmin)./max(zmax - zmin,10e-10);
    ObjMSE  =  ObjMSE./(max(zmax - zmin,10e-10).^2);

    %% Reference Set 
    global phase NI
    if phase == 2
        num = 0;
        for i= 1 : length(Database)
            if all(Database(i).cons<=0)
                num = num + 1;
            end
        end
        [FrontNo,~] = NDSort(ALL_Obj,ALL_Con,inf);
        if num > NI
            ALL_Obj = ALL_Obj(FrontNo==1,:);
        else
            i = 1;
            Next = FrontNo == i;
            while sum(Next) <= NI
                Next(FrontNo == i) = true;
                i = i + 1;
            end
            ALL_Obj = ALL_Obj(Next,:);
        end
    else
        [FrontNo,~] = NDSort(ALL_Obj,inf);
        ALL_Obj     = ALL_Obj(FrontNo==1,:);
    end

    %% Select mu points
    if phase == 2
        [FrontNo,MaxFNo] = NDSort_PDPD(PopObj,ObjMSE,PopCon,ConMSE,mu);
    else
        [FrontNo,MaxFNo] = NDSort_PDPD(PopObj,ObjMSE,mu);
    end
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    if length(Last) == mu - sum(Next)
        Next(Last) = true;
    elseif length(Last) > mu - sum(Next)
        ALL_Obj = [ALL_Obj;PopObj(Next,:)];
        for i = 1 : mu - sum(Next)
            index             = EucDistance_Select(PopObj(Last,:),ALL_Obj);
            Next(Last(index)) = true;
            ALL_Obj           = [ALL_Obj;PopObj(Last(index),:)];
            Last(index)       = [];
        end
    end
end
```

### `Evo_Search.m`
```matlab
function [PopDec,PopObj,PopCon,ObjMSE,ConMSE] = Evo_Search(P,wmax,Model_obj,Model_con,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao_zhang0@163.com)

    PopDec = P.decs;
    PopObj = P.objs;
    PopCon = P.cons;
    ObjMSE = zeros(Problem.N,Problem.M);
    ConMSE = zeros(Problem.N,size(PopCon,2));
    w      = 1;
    while w <= wmax
        drawnow();
        OffDec = OperatorGA(Problem,PopDec);

        [OffObj,Off_ObjMSE,OffCon,Off_ConMSE] = model_predict(Model_obj,Model_con,OffDec);

        PopDec = [PopDec;OffDec];
        PopObj = [PopObj;OffObj];
        PopCon = [PopCon;OffCon];
        ObjMSE = [ObjMSE;Off_ObjMSE];
        ConMSE = [ConMSE;Off_ConMSE];

        index  = EnvironmentalSelection(PopObj,ObjMSE,PopCon,ConMSE,length(P));

        PopDec = PopDec(index,:);
        PopObj = PopObj(index,:);
        PopCon = PopCon(index,:);
        ObjMSE = ObjMSE(index,:);
        ConMSE = ConMSE(index,:);

        w = w + 1;
    end
end

function [OffObj,Off_ObjMSE,OffCon,Off_ConMSE] = model_predict(Model_obj,Model_con,OffDec)
    global Len_con Len_obj phase
    [N,~]      = size(OffDec);
    OffObj     = zeros(N,Len_obj);
    OffCon     = zeros(N,Len_con);
    Off_ObjMSE = zeros(N,Len_obj);
    Off_ConMSE = zeros(N,Len_con);
    for i = 1 : N
        for j = 1 : Len_obj
            [OffObj(i,j),~,Off_ObjMSE(i,j)] = predictor(OffDec(i,:),Model_obj{j});
        end
        if phase == 2
            for j = 1 : Len_con
                [OffCon(i,j),~,Off_ConMSE(i,j)] = predictor(OffDec(i,:),Model_con{j});
            end
        end
    end
end

function Next = EnvironmentalSelection(PopObj,ObjMSE,PopCon,ConMSE,N)
    %% Non-dominated sorting
    zmin   = min(PopObj);
    zmax   = max(PopObj);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    ObjMSE = ObjMSE./(max(zmax - zmin,10e-10).^2);
    
    global phase
    if phase == 2
        [FrontNo,MaxFNo] = NDSort_PDPD(PopObj,ObjMSE,PopCon,ConMSE,N);
    else
        [FrontNo,MaxFNo] = NDSort_PDPD(PopObj,ObjMSE,N);
    end

    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);

    %% Select the solutions in the last front
    if MaxFNo == 1
        Del = Truncation(PopObj(Last,:),N);
        Next(Last(Del)) = true; 
    else
        Choose = Dis_Selection(PopObj,Last,N-sum(Next));
        Next(Last(Choose)) = true;
    end
end

function Choose = Dis_Selection(PopObj,Last,mu)
    N = size(PopObj,1);

    %% Calculate the distance between each two solutions
    for i = 1 : N
        for j = [1:i-1,i+1:N]
            Distance(i,j) = norm(PopObj(i,:)-PopObj(j,:),2);
        end
    end
    
    %% Calculate D
    Distance = sort(Distance,2);
    D = 1./(Distance(:,1) + 2);
    D = D(Last);
    [~,index] = sort(D);
    Choose    = index(1:mu);
end

function Del = Truncation(PopObj,K)
    %% Select part of the solutions by truncation
    [N,~] = size(PopObj);
    
    %% Calculate the distance between each two solutions
    Distance = inf(N);
    for i = 1 : N
         for j = 1 : N
            Distance(i,j) = norm(PopObj(i,:) - PopObj(j,:),2);
        end
    end
    
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

### `NDSort_PDPD.m`
```matlab
function[FrontNo,MaxFNo] = NDSort_PDPD(varargin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao_zhang0@163.com)

    %% Read data
    PopObj = varargin{1};
    ObjMSE = varargin{2};
    if nargin == 3
        nSort = varargin{3};
        mark  = 1;
    elseif nargin == 5
        PopCon      = varargin{3};
        ConMSE      = varargin{4};
        nSort       = varargin{5};
        [LPoF,TPoF] = Feasible_Probability(PopCon,ConMSE);
        mark        = 0;
    end
    epsilon = 0.75;

    %% Obtain Dominance Matrix 
    N        = size(PopObj,1);
    sigma    = sqrt(ObjMSE(reshape(ones(N,1)*(1:N),N*N,1),:) + repmat(ObjMSE,N,1));
    mean     = PopObj(reshape(ones(N,1)*(1:N),N*N,1),:) - repmat(PopObj,N,1);
    x_PD     = normcdf((0-mean)./sigma);
    y_PD     = 1 - x_PD;
    dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            Pi             = x_PD(N*(i-1)+j,:);
            Pj             = y_PD(N*(i-1)+j,:);
            index1         = find(abs(Pi - Pj)<=epsilon);
            index2         = 1 : length(Pi);
            index2(index1) = [];
            PDi            = prod(Pi(index1));
            PDj            = prod(Pj(index1));
    
            if mark == 1
                % PDPD
                if all([-Pi(index2),-PDi] <= [-Pj(index2),-PDj]) && ~all([-Pi(index2),-PDi] == [-Pj(index2),-PDj])
                    flag = 1;
                elseif all([-Pi(index2),-PDi] >= [-Pj(index2),-PDj]) && ~all([-Pi(index2),-PDi] == [-Pj(index2),-PDj])
                    flag = 2;
                else
                    flag = 3;
                end
            else 
                % Constrained PDPD
                if ((LPoF(i) >= 0.5) && (LPoF(j) >= 0.5)) || (LPoF(i) == LPoF(j))
                    if all([-Pi(index2),-PDi] <= [-Pj(index2),-PDj]) && ~all([-Pi(index2),-PDi] == [-Pj(index2),-PDj])
                        flag = 1;
                    elseif all([-Pi(index2),-PDi] >= [-Pj(index2),-PDj]) && ~all([-Pi(index2),-PDi] == [-Pj(index2),-PDj])
                        flag = 2;
                    else
                        flag = 3;
                    end
                else
                    [~,flag] = max([TPoF(i),TPoF(j)]);
                end
            end
            
            if flag == 1
                dominate(i,j) = true;
            elseif flag == 2
                dominate(j,i) = true;
            end
        end
    end
    
    %% Sorting
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

function [LPoF,TPoF] = Feasible_Probability(PopCon,ConMSE)
    [N,M] = size(PopCon);
    LPoF  = ones(N,1);
    TPoF  = ones(N,1);
    for i = 1 : N
        for j = 1 : M
             LPoF(i) = min([LPoF(i),normcdf((0-PopCon(i,j))/sqrt(ConMSE(i,j)))]);
             TPoF(i) = TPoF(i)*normcdf((0-PopCon(i,j))/sqrt(ConMSE(i,j)));
        end
    end
end
```

### `Phase_Trans.m`
```matlab
function [phase,ct] = Phase_Trans(A2,C,ct,ct_max,phase)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao_zhang0@163.com)

    if phase == 1
        feasible_num   = 0;
        feasible_index = [];
        index          = 0;
        for i= 1 : length(A2)
            if all(A2(i).cons<=0)
                feasible_num   = feasible_num + 1;
                feasible_index = [feasible_index,i];
            end
        end
        
        if feasible_num ~= 0
            feasible_newindex   = [];feasible_newnum = 0;
            infeasible_newindex = [];
            for i= 1 : length(C)
                if all(C(i).cons<=0)
                    feasible_newnum   = feasible_newnum + 1;
                    feasible_newindex = [feasible_newindex,i];
                else
                    infeasible_newindex = [infeasible_newindex,i];
                end
            end
           
            if ((isempty(feasible_newindex)==0 && set_dominate(C(feasible_newindex),A2(feasible_index)) == 3) || isempty(feasible_newindex)) &&...
               (isempty(infeasible_newindex)||(isempty(infeasible_newindex)==0 && (set_dominate(C(infeasible_newindex),A2(feasible_index)) == 3 ||...
               set_dominate(C(infeasible_newindex),A2(feasible_index)) == 1)))
                ct = ct + 1;
                if ct >= ct_max
                    index = 1;
                end
            else
                ct = 0;
            end
        end
        
        if (feasible_num ~= 0) && (phase == 1) && (index == 1)
            phase = 2;
        else
            phase = 1;
        end
    end
end

function flag = set_dominate(A,B)
    %flag = 1: one of A dominte B
    %flag = 2: B dominte A
    %flag = 3: A is nondominated with B
    %flag = 4: other

    [FrontNo,~]    = NDSort(B.objs,inf);
    B              = B(FrontNo == 1);
    [FrontNo,~]    = NDSort(A.objs,inf);
    A              = A(FrontNo == 1);
    Aobj           = A.objs;
    Bobj           = B.objs;
    Asize          = length(A);
    Bsize          = length(B);
    dominate_index = zeros(Asize,Bsize + 1);
    for i = 1 : Asize
        for j = 1 : Bsize
            if all(Aobj(i,:) == Bobj(j,:))
                dominate_index(i,j) = 3;
            elseif all(Aobj(i,:) <= Bobj(j,:))
                dominate_index(i,j) = 1;
            elseif all(Aobj(i,:) >= Bobj(j,:))
                dominate_index(i,j) = 2;
            else
                dominate_index(i,j) = 3;
            end
        end

        uni = unique(dominate_index(i,1:Bsize));
        uni = sort(uni);
        if length(uni) == 1
            dominate_index(i,Bsize + 1) = uni;
        elseif length(uni) == 2
            if all(uni == [1,3])
                dominate_index(i,Bsize + 1) = 1;
            elseif all(uni == [2,3])
                dominate_index(i,Bsize + 1) = 2;
            else
                dominate_index(i,Bsize + 1) = 4;
            end
        else
            dominate_index(i,Bsize + 1) = 4;
        end
    end

    uni = unique(dominate_index(:,Bsize + 1))';

    if length(uni) == 1
        flag = uni;
    elseif length(uni) == 2
        if all(uni == [1,3])
            flag = 1;
        elseif all(uni == [2,3])
            flag = 2;
        else
            flag = 4;
        end
    else
        flag = 4;
    end
end
```

### `Pop_Reselect.m`
```matlab
function Next = Pop_Reselect(PopObj,PopCon,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao_zhang0@163.com)

    %% Non-dominated sorting
    zmin   = min(PopObj);
    zmax   = max(PopObj);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    global phase
    if phase == 2
        [FrontNo,MaxFNo] = NDSort(PopObj,PopCon,N);
    else
        [FrontNo,MaxFNo] = NDSort(PopObj,N);
    end
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);

    %% Select the solutions in the last front
    if MaxFNo == 1
        Del = Truncation(PopObj(Last,:),N);
        Next(Last(Del)) = true;
    else
        Choose = Dis_Selection(PopObj,Last,N-sum(Next));  
        Next(Last(Choose)) = true;
    end
end

function Choose = Dis_Selection(PopObj,Last,mu)
    N = size(PopObj,1);
    Distance = inf(N);
    for i = 1 : N
        for j = [1:i-1,i+1:N]
           Distance(i,j) = norm(PopObj(i,:)-PopObj(j,:),2);
        end
    end
    Distance  = sort(Distance,2);
    D         = 1./(Distance(:,1)+2);
    D         = D(Last);
    [~,index] = sort(D);
    Choose    = index(1:mu);
end

function Del = Truncation(PopObj,K)
    %% Select part of the solutions by truncation
    N = size(PopObj,1);
    
    %% Calculate the distance between each two solutions
    Distance = inf(N);
    for i = 1 : N
        for j = 1 : N
            Distance(i,j) = norm(PopObj(i,:)-PopObj(j,:),2);
        end
    end
    
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

### `TEA.m`
```matlab
classdef TEA < ALGORITHM
% <2024> <multi/many> <real/integer> <expensive> <constrained/none>
% Two-phase evolutionary algorithm
% wmax --- 20 --- Generations of evolutionary search
% mu   ---  5 --- Number of selected candidates
    
%------------------------------- Reference --------------------------------
% Z. Zhang, Y. Wang, J. Liu, G. Sun, and K. Tang. A two-phase Kriging-
% assisted evolutionary algorithm for expensive constrained multiobjective
% optimization problems. IEEE Transactions on Systems, Man, and
% Cybernetics: Systems, 2024, 54(8): 4579-4591.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao_zhang0@163.com)
    
    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            global phase NI Len_con Len_obj 
            [wmax,mu]      = Algorithm.ParameterSet(20,5,2);
            ct             = 0;
            ct_max         = 2;
            sample_success = 1;
            phase          = 1;

            %% Initialization
            NI        = Problem.N;
            P         = UniformPoint(NI,Problem.D,'Latin');
            Database  = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            P         = Database;
            Len_obj   = Problem.M;
            THETA_obj = 5.*ones(Len_obj,Problem.D);
            Model_obj = cell(1,Len_obj); 
            Len_con   = size(Database.cons,2);
            THETA_con = 5.*ones(Len_con,Problem.D);
            Model_con = cell(1,Len_con);

            %% Optimization
            while Algorithm.NotTerminated(Database)
                % Surrogate Construction
                if sample_success
                    [Model_obj,Model_con,THETA_obj,THETA_con] = model_train(Database,THETA_obj,THETA_con);
                end
                % Evolutionary Search
                [PopDec,PopObj,PopCon,ObjMSE,ConMSE] = Evo_Search(P,wmax,Model_obj,Model_con,Problem);
                % Candidate Seletion
                C = Candi_Select(PopDec,PopObj,PopCon,ObjMSE,ConMSE,Database,mu);
                % Phase Transition
                sample_success = 0;
                if isempty(C) == 0
                    C = Problem.Evaluation(C);
                    sample_success = 1;
                    [phase,ct]     = Phase_Trans(Database,C,ct,ct_max,phase);
                end
                % Population Reselection
                Database = [Database,C];
                index    = Pop_Reselect(Database.objs,Database.cons,NI);
                P        = Database(index);
            end
        end
    end
end

function [Model_obj,Model_con,THETA_obj,THETA_con] = model_train(Database,THETA_obj,THETA_con)
    global Len_con Len_obj phase 
    Dec     = Database.decs;
    Obj     = Database.objs;
    Con     = Database.cons;
    Len_dec = size(Dec,2);
    Len_obj = size(Obj,2);
    Len_con = size(Con,2);
    for i = 1 : Len_obj
        [~,distinct1]  = unique(round(Dec*1e12)/1e12,'rows');
        [~,distinct2]  = unique(round(Obj(:,i)*1e12)/1e12,'rows');
        distinct       = intersect(distinct1,distinct2);
        dmodel         = dacefit(Dec(distinct,:),Obj(distinct,i),'regpoly1','corrgauss',THETA_obj(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
        Model_obj{i}   = dmodel;
        THETA_obj(i,:) = dmodel.theta;
    end
    if phase == 2
        for i = 1 : Len_con
            [~,distinct1]  = unique(round(Dec*1e12)/1e12,'rows');
            [~,distinct2]  = unique(round(Con(:,i)*1e12)/1e12,'rows');
            distinct       = intersect(distinct1,distinct2);
            dmodel         = dacefit(Dec(distinct,:),Con(distinct,i),'regpoly1','corrgauss',THETA_con(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
            Model_con{i}   = dmodel;
            THETA_con(i,:) = dmodel.theta;
        end
    else
        Model_con = [];
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
