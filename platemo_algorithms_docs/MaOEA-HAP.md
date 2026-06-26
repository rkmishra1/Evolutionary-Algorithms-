# MaOEA-HAP

**Tags**: <2026> <multi/many> <real/integer/label/binary/permutation>

## Description
Hyper-curvature balanced indicator and adaptive phase exploration co-driven many-objective evolutionary algorithm

## Reference
X. Yue, W. Wen, Y. Jiang, Y. Tian, and H. Peng. A hyper-curvature balanced indicator and adaptive phase exploration co-driven evolutionary algorithm for many-objective optimization. Swarm and Evolutionary Computation, 2026.

## Source Code

### `AssociateAngle.m`
```matlab
function AngleM = AssociateAngle(PopObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M]  = size(PopObj);
    cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    Angle1 = acos(cosine);
    Angle2 = sort(Angle1,2);
    AngleM = zeros(N,1);
    Angle  = Angle2(:,1:M);
    for i = 1 : N
        for j = 1 : M
            AngleM(i) = AngleM(i) + Angle(i,j)/j;
        end
    end
end
```

### `CrowdDistance.m`
```matlab
function MatingPool = CrowdDistance(CA,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [FrontNo,~] = NDSort(CA.objs,CA.cons,N);
    CrowdDis    = CrowdingDistance(CA.objs,FrontNo);
    for i = 1 : N
        Index = randperm(N,2);
        if CrowdDis(Index(1)) < CrowdDis(Index(2))
            MatingPool(i) = Index(2);
        elseif CrowdDis(Index(1)) > CrowdDis(Index(2))
            MatingPool(i) = Index(1);
        else
            if rand() < 0.5
                MatingPool(i) = Index(1);
            else
                MatingPool(i) = Index(2);
            end
        end
    end
end
```

### `MaOEAHAP.m`
```matlab
classdef MaOEAHAP < ALGORITHM
% <2026> <multi/many> <real/integer/label/binary/permutation>
% Hyper-curvature balanced indicator and adaptive phase exploration co-driven many-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% X. Yue, W. Wen, Y. Jiang, Y. Tian, and H. Peng. A hyper-curvature
% balanced indicator and adaptive phase exploration co-driven evolutionary
% algorithm for many-objective optimization. Swarm and Evolutionary
% Computation, 2026.
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
            CAsize = Problem.N;
            p      = 1/Problem.M;
            k      = 0;

            %% Generate random population
            Population = Problem.Initialization();
            CA = UpdateCA([],Population,CAsize,p);
            DA = UpdateDA([],Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(DA)
                [ParentC,ParentM] = MatingSelection(CA,DA,Problem.N);
                Offspring         = [OperatorGA(Problem,ParentC,{1,20,0,0}),OperatorGA(Problem,ParentM,{0,0,1,20})];
                CA = UpdateCA(CA,Offspring,CAsize,p);
                DA = UpdateDA(DA,Offspring,Problem.N,p,Problem,k);
                k  = flog(DA,Problem.N,p);
            end
        end
    end
end
```

### `MatingSelection.m`
```matlab
function [ParentC,ParentM] = MatingSelection(CA,DA,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CAParent1 = randi(length(CA),1,ceil(N/2));
    CAParent2 = randi(length(CA),1,ceil(N/2));
    Dominate  = any(CA(CAParent1).objs<CA(CAParent2).objs,2) - any(CA(CAParent1).objs>CA(CAParent2).objs,2);  
    ParentC   = [CA([CAParent1(Dominate==1),CAParent2(Dominate~=1)]),...
                 DA(randi(length(DA),1,ceil(N/2)))];
    ParentM   = CA(CrowdDistance(CA,N));
end
```

### `NDQSort.m`
```matlab
function [FrontNo,MaxFNo] = NDQSort(varargin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = varargin{1};
    [N,M]  = size(PopObj);
    if nargin == 2
        nSort = varargin{2};
    else
        PopCon     = varargin{2};
        nSort      = varargin{3};
        Infeasible = any(PopCon>0,2);
        PopObj(Infeasible,:) = repmat(max(PopObj,[],1),sum(Infeasible),1) + repmat(sum(max(0,PopCon(Infeasible,:)),2),1,M);
    end
    if M < 3 || N < 500
        [FrontNo,MaxFNo] = ENS_SS(PopObj,nSort);
    else
        [FrontNo,MaxFNo] = T_ENS(PopObj,nSort);
    end
end

function [FrontNo,MaxFNo] = ENS_SS(PopObj,nSort)
    [PopObj,~,Loc] = unique(PopObj,'rows');
    Table   = hist(Loc,1:max(Loc));
    [N,M]   = size(PopObj);
    Angle   = AssociateAngle(PopObj);
    FrontNo = inf(1,N);
    MaxFNo  = 0;
    while sum(Table(FrontNo<inf)) < min(nSort,length(Loc))
        MaxFNo = MaxFNo + 1;
        for i = 1 : N
            if FrontNo(i) == inf
                Dominated = false;
                for j = i-1 : -1 : 1
                    if FrontNo(j) == MaxFNo
                        m = 2;
                        while m <= M && PopObj(i,m) >= PopObj(j,m)
                            m = m + 1;
                        end
                        if m > M && Angle(i) < Angle(j)
                            Dominated = true;
                        else
                            Dominated = false;
                        end
                        if Dominated || M == 2
                            break;
                        end
                    end
                end
                if ~Dominated
                    FrontNo(i) = MaxFNo;
                end
            end
        end
    end

    FrontNo = FrontNo(:,Loc);
end

function [FrontNo,MaxFNo] = T_ENS(PopObj,nSort)
    [PopObj,~,Loc] = unique(PopObj,'rows');
    Table     = hist(Loc,1:max(Loc));
	[N,M]     = size(PopObj);
    FrontNo   = inf(1,N);
    MaxFNo    = 0;
    Forest    = zeros(1,N);
    Children  = zeros(N,M-1);
    LeftChild = zeros(1,N) + M;
    Father    = zeros(1,N);
    Brother   = zeros(1,N) + M;
    [~,ORank] = sort(PopObj(:,2:M),2,'descend');
    ORank     = ORank + 1;
    while sum(Table(FrontNo<inf)) < min(nSort,length(Loc))
        MaxFNo = MaxFNo + 1;
        root   = find(FrontNo==inf,1);
        Forest(MaxFNo) = root;
        FrontNo(root)  = MaxFNo;
        for p = 1 : N
            if FrontNo(p) == inf
                Pruning = zeros(1,N);
                q = Forest(MaxFNo);
                while true
                    m = 1;
                    while m < M && PopObj(p,ORank(q,m)) >= PopObj(q,ORank(q,m))
                        m = m + 1;
                    end
                    if m == M
                        break;
                    else
                        Pruning(q) = m;
                        if LeftChild(q) <= Pruning(q)
                            q = Children(q,LeftChild(q));
                        else
                            while Father(q) && Brother(q) > Pruning(Father(q))
                                q = Father(q);
                            end
                            if Father(q)
                                q = Children(Father(q),Brother(q));
                            else
                                break;
                            end
                        end
                    end
                end
                if m < M
                    FrontNo(p) = MaxFNo;
                    q = Forest(MaxFNo);
                    while Children(q,Pruning(q))
                        q = Children(q,Pruning(q));
                    end
                    Children(q,Pruning(q)) = p;
                    Father(p) = q;
                    if LeftChild(q) > Pruning(q)
                        Brother(p)   = LeftChild(q);
                        LeftChild(q) = Pruning(q);
                    else
                        bro = Children(q,LeftChild(q));
                        while Brother(bro) < Pruning(q)
                            bro = Children(q,Brother(bro));
                        end
                        Brother(p)   = Brother(bro);
                        Brother(bro) = Pruning(q);
                    end
                end
            end
        end
    end
    FrontNo = FrontNo(:,Loc);
end
```

### `Shape_Estimate.m`
```matlab
function p= Shape_Estimate(Population,N,z,znad)
% Estimate the  shape of PF

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [FrontNo,~] = NDSort(Population.objs,N);
    Population  = Population(FrontNo<=1);
    PopObj      = Population.objs ;
    [N,~]       = size(PopObj);
    PopObj      = (PopObj-repmat(z,size(PopObj,1),1))./repmat(znad-z ,size(PopObj,1),1);

    k  = 1.5;
    CP = 0.5 : 0.1 : 2;
    for i = 1 : length(CP)
        Gp   = (sum(PopObj.^CP(i),2)).^(1/CP(i));
        temp = sort(Gp);
        Q1   = temp(max(fix(N*0.25),1));
        Q3   = temp(max(fix(N*0.75),1));
        Max  = Q3+k*(Q3-Q1);
        Gp(Gp>Max) = []; % Gp is denoised using box plot
        Vp(i) = std(Gp./max(Gp));
    end
    [~,index] = min(Vp);
    p = CP(index);
end
```

### `UpdateCA.m`
```matlab
function CA = UpdateCA(CA,New,MaxSize,p)
% Update CA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CA = [CA,New];
    N  = length(CA);
    if N <= MaxSize
        return;
    end
    
    %% Calculate the fitness of each solution
    CAObj = CA.objs;
    CAObj = (CAObj-repmat(min(CAObj),N,1))./(repmat(max(CAObj)-min(CAObj),N,1));
    I     = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(CAObj(i,:)-CAObj(j,:));
        end
    end
    C = max(abs(I));
    F = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
    
    %% Delete part of the solutions by their fitnesses
    Choose = 1 : N;
    while length(Choose) > MaxSize
        [~,x] = min(F(Choose));
        F = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
        Choose(x) = [];
    end
    CA = CA(Choose);
end
```

### `UpdateDA.m`
```matlab
function DA = UpdateDA(DA,New,MaxSize,p,Problem,k)
% Update DA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Find the non-dominated solutions
    DA = [DA,New];
    ND = NDQSort(DA.objs,1);
    DA = DA(ND==1);
    N  = length(DA);
    if N <= MaxSize
        return;
    end
    Popobj = DA.objs;
    
    %% Normalization
    Zmin   = min(Popobj,[],1);
    Zmax   = max(Popobj,[],1);
    Popobj = (Popobj-repmat(Zmin,size(Popobj,1),1))./repmat(Zmax-Zmin,size(Popobj,1),1);  
    Choose = false(1,N);

    %% Slect the extreme solutions first
    M = size(Popobj,2);    
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        objmatrix = Popobj(~Choose,:);
        [~,Extreme(i)]     = min(max(Popobj(~Choose,:)./repmat(w(i,:),sum(~Choose),1),[],2)+0.1*objmatrix(:,i)/(1e-6)); % 带惩罚的AFS函数值
        Choose(Extreme(i)) = true; 
    end    
    
    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > MaxSize
        % Randomly delete several solutions
        Choosed = find(Choose);
        k       = randperm(sum(Choose),sum(Choose)-MaxSize);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < MaxSize
        % Calculate the angle
        angle    = acos(1-pdist2(Popobj,Popobj,'cosine'));
        Lp       = Shape_Estimate(DA,MaxSize,Zmin,Zmax);
        PopobjLP = DA.objs - Zmin;
        PopobjLP = PopobjLP+10^-6;
        tran_Obj = PopobjLP./repmat((sum(PopobjLP.^Lp,2)).^(1/Lp),1,M);
        for i = 1 : size(PopobjLP,1)
            DAPT(i) = norm(PopobjLP(i,:) - tran_Obj(i,:),p);
        end

        % Select the rest individuals
        while sum(Choose) < MaxSize
            if k == 0
                Select   = find(Choose);
                Remain   = find(~Choose);
                a        = min(angle(Remain,Select),[],2);
                bb       = a'.*(1 + Problem.FE^2*2/Problem.maxFE^2);
                cc       = 1./(DAPT(Remain).*(2-Problem.FE^2/Problem.maxFE^2));
                fitness  = bb.*(cc);
                [~,rho2] = max(fitness);
                Choose(Remain(rho2)) = true;
            elseif k == 1
                Select  = find(Choose);
                Remain  = find(~Choose);
                [~,rho] = max(min(angle(Remain,Select),[],2));
                Choose(Remain(rho)) = true;
            end
        end
    end
    DA = DA(Choose);
end
```

### `flog.m`
```matlab
function k = flog(DA,N,p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Popobj = DA.objs;
    M      = size(Popobj,2);
    
    %% Normalization
    Zmin     = min(Popobj,[],1);
    Zmax     = max(Popobj,[],1);
    Lp       = Shape_Estimate(DA,N,Zmin,Zmax);
    PopobjLP = DA.objs - Zmin;
    PopobjLP = PopobjLP+10^-6;
    tran_Obj = PopobjLP./repmat((sum(PopobjLP.^Lp,2)).^(1/Lp),1,M);
    for i = 1 : size(PopobjLP,1)
        DAPT(i) = norm(PopobjLP(i,:) - tran_Obj(i,:),p);
    end
    if all(DAPT<1)
        k = 1;
    else
        k = 0;
    end
end
```
