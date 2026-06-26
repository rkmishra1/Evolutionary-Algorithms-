# KTA2

**Tags**: <2021> <multi/many> <real/integer> <expensive>

## Description
Kriging-assisted Two_Arch2

## Reference
Z. Song, H. Wang, C. He, and Y. Jin. A Kriging-assisted two-archive evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6): 1013-1027.

## Source Code

### `Adaptive_sampling.m`
```matlab
function Offspring01 = Adaptive_sampling(CAobj,DAobj,CAdec,DAdec,DAvar,DA,mu,p,phi)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    Ideal_Point = min([CAobj;DAobj],[],1);  
    flag        = Cal_Convergence(CAobj,DAobj,Ideal_Point); 
    if flag == 1   
        % convergence sampling strategy
        N       = size(CAobj,1);
        CAobj01 = (CAobj-repmat(min(CAobj),N,1))./(repmat(max(CAobj)-min(CAobj),N,1));
        I       = zeros(N);
        for i = 1 : N
            for j = 1 : N
                I(i,j) = max(CAobj01(i,:)-CAobj01(j,:));
            end
        end
        C      = max(abs(I));
        F      = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
        Choose = 1 : N;
        while length(Choose) > mu
            [~,x]     = min(F(Choose));
            F         = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
            Choose(x) = [];
        end
        Offspring01 = CAdec(Choose,:);
    else
        if PD(DAobj) < PD(DA.objs)
            % uncertainty sampling strategy
            An     = size(DAvar,1);
            Choose = zeros(1,5);
            for i = 1 : mu
                A_num       = randperm(size(DAvar,1));
                Uncertainty = mean(DAvar(A_num(1:ceil(phi*An)),:),2);
                [~,best]    = max(Uncertainty);
                Choose (i)  = A_num(best);
            end
            Offspring01 = DAdec(Choose ,:);
        else
            % diversity sampling strategy
            DA_Nor = (DA.objs - repmat(min([DAobj;DA.objs],[],1),length(DA),1))...
                ./repmat(max([DAobj;DA.objs],[],1) - min([DAobj;DA.objs],[],1),length(DA),1);
            DA_Nor_pre = (DAobj - repmat(min([DAobj;DA.objs],[],1),size(DAobj,1),1))...
                ./repmat(max([DAobj;DA.objs],[],1) - min([DAobj;DA.objs],[],1),size(DAobj,1),1);
            N       = size(DA_Nor,1);
            Pop     = [DA_Nor;DA_Nor_pre];
            Pop_dec = [DA.decs;DAdec];
            NN      = size(Pop,1);
            Choose  = false(1,NN);
            Choose(1:N) = true;
            MaxSize     = N+mu;
            Distance    = inf(N);
            for i = 1 : NN-1
                for j = i+1 : NN
                    Distance(i,j) = norm(Pop(i,:)-Pop(j,:),p);
                    Distance(j,i) = Distance(i,j);
                end
            end
            Offspring01 = [];
            while sum(Choose) < MaxSize
                Remain = find(~Choose);
                [~,x]  = max(min(Distance(~Choose,Choose),[],2));
                Choose(Remain(x)) = true;
                Offspring01 = [Offspring01;Pop_dec(Remain(x),:)];
            end
        end
    end
end

function Score = PD(PopObj)
% Pure diversity

    N = size(PopObj,1);
    C = false(N);
    C(logical(eye(size(C)))) = true;
    D = pdist2(PopObj,PopObj,'minkowski',0.1);
    D(logical(eye(size(D)))) = inf;
    Score = 0;
    for k = 1 : N-1
        while true
            [d,J] = min(D,[],2);
            [~,i] = max(d);
            if D(J(i),i) ~= -inf
                D(J(i),i) = inf;
            end
            if D(i,J(i)) ~= -inf
                D(i,J(i)) = inf;
            end
            P = any(C(i,:),1);
            while ~P(J(i))
                newP = any(C(P,:),1);
                if P == newP
                    break;
                else
                    P = newP;
                end
            end
            if ~P(J(i))
                break;
            end
        end
        C(i,J(i)) = true;
        C(J(i),i) = true;
        D(i,:)    = -inf;
        Score     = Score + d(i);
    end
end
```

### `Cal_Convergence.m`
```matlab
function flag = Cal_Convergence(PopObj1,PopObj2,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    N1 = size(PopObj1,1);
    N2 = size(PopObj2,1);
    if N1~=N2
        flag = 0;
    else
        PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
        PopObj = (PopObj)./repmat(max(PopObj,[],1) - Zmin,size(PopObj,1),1);
        Distance1 = zeros(1,N1);
        Distance2 = zeros(1,N2);
        % calculate the distance sets of CCA and CDA
        for i = 1:N1
            Distance1(i) = sqrt(sum(PopObj(i,:),2));
        end
        for i = 1: N2
            Distance2(i) = sqrt(sum(PopObj(N1+i,:),2));
        end
        % rank-sum test, alpha = 0.05
        [~,flag,~,r1,r2]=signrank_new(Distance1, Distance2,'alpha',0.05);
        if flag == 1 && (r1-r2 <0)
            flag = 0;
        end
    end
end
```

### `KTA2.m`
```matlab
classdef KTA2 < ALGORITHM
% <2021> <multi/many> <real/integer> <expensive>
% Kriging-assisted Two_Arch2
% tau  --- 0.75 --- Proportion of one type noninfluential points in training data
% phi  ---  0.1 --- Number of randomly selected individuals
% wmax ---   10 --- Number of generations before updating CA and DA 
% mu   ---    5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% Z. Song, H. Wang, C. He, and Y. Jin. A Kriging-assisted two-archive
% evolutionary algorithm for expensive many-objective optimization. IEEE
% Transactions on Evolutionary Computation, 2021, 25(6): 1013-1027.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song
% Email:zssong@stu.xidian.edu.cn

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [tau,phi,wmax,mu] = Algorithm.ParameterSet(0.75,0.1,10,5);
            
            %% Initialization 
            p      = 1/Problem.M;
            CAsize = Problem.N;
            N      = Problem.N;
            P      = UniformPoint(N, Problem.D, 'Latin');
            Population     = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*P+repmat(Problem.lower,N,1));
            All_Population = Population;
            Ho_Population  = All_Population;
            CA       = UpdateCA([],Population,CAsize);
            DA       = Population;
            THETA_S  = 5.*ones(Problem.M,Problem.D);
            THETA_IS =  5.*ones(2,Problem.M,Problem.D);
            Model_sensitive   = cell(1,Problem.M);
            Model_insensitive = cell(2,Problem.M);

            %% Optimization
            while Algorithm.NotTerminated(All_Population)
                %***** Building influential point-insensitive model********
                % build sensitive model
                Dec = All_Population.decs;
                Obj = All_Population.objs;
                for i = 1 : Problem.M
                    dmodel             = dacefit(Dec,Obj(:,i),'regpoly0','corrgauss',THETA_S(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    Model_sensitive{i} = dmodel;
                    THETA_S(i,:)       = dmodel.theta;
                end
                % build insensitive models 
                Centers = zeros(Problem.M,2);
                for i = 1 : Problem.M
                    [~,N1] = sort(Obj(:,i));
                    num    = ceil(length(All_Population).*tau);
                    mean_index{1} = N1(1:num);
                    mean_index{2} = N1(end-num:end);
                    for j = 1 : 2
                        Centers(i,j) = mean(Obj(mean_index{j},i));  % lambda and miu
                    end
                    for j = 1 : 2
                        train_X = Dec(mean_index{j},:);
                        train_Y = Obj(mean_index{j},i);
                        dmodel  = dacefit(train_X,train_Y,'regpoly0','corrgauss',THETA_IS(j,i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                        Model_insensitive{j,i} = dmodel;
                        THETA_IS(j,i,:)        = dmodel.theta;
                    end
                end
                % Set the CCA and CDA as the current CA and DA
                CAobj = CA.objs;
                CAdec = CA.decs;
                DAobj = DA.objs;
                DAdec = DA.decs;
                w     = 1;
                while w <= wmax   % this part is same as Two_Arch2 
                    [~,ParentCdec,~,ParentMdec] = MatingSelection_KTA2(CAobj,CAdec,DAobj,DAdec,Problem.N);
                    OffspringDec = [OperatorGA(Problem,ParentCdec,{1,20,0,0});OperatorGA(Problem,ParentMdec,{0,0,1,20})];
                    PopDec = [DAdec;CAdec;OffspringDec];
                    N      = size(PopDec,1);
                    PopObj = zeros(N,Problem.M);
                    MSE    = zeros(N,Problem.M);
                    %****** Using influential point-insensitive model *****
                    for i = 1 : N
                        for j = 1 : Problem.M
                            [PopObj(i,j),~,~] = predictor(PopDec(i,:),Model_sensitive{j});
                            if abs(PopObj(i,j)- Centers(j,1)) <= abs(PopObj(i,j)- Centers(j,2))
                                model = Model_insensitive{1,j};
                            else
                                model = Model_insensitive{2,j};
                            end
                            [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),model);
                        end
                    end
                    [CAobj,CAdec,~]     = K_UpdateCA(PopObj,PopDec,MSE,CAsize);
                    [DAobj,DAdec,DAvar] = K_UpdateDA(PopObj,PopDec,MSE,Problem.N,p);
                    w                   = w + 1;
                end
                
                % Adaptive sampling 
                Offspring01 = Adaptive_sampling(CAobj,DAobj,CAdec,DAdec,DAvar,DA,mu,p,phi);
                
                [~,index]   = unique(Offspring01 ,'rows');
                PopNew      = Offspring01(index,:);
                Offspring02 = [];
                for i = 1 : size(PopNew,1)
                    dist2 = pdist2(real( PopNew(i,:)),real(All_Population.decs));
                    if min(dist2) > 1e-5
                        Offspring02 = [Offspring02;PopNew(i,:)];
                    end
                end
                if ~isempty(Offspring02)
                    Offspring = Problem.Evaluation(Offspring02);

                    temp = All_Population.decs;
                    for i = 1 : size(Offspring,2)
                        dist2 = pdist2(real(Offspring(i).decs),real(temp));
                        if min(dist2) > 1e-5
                            All_Population = [All_Population,Offspring(i)];
                        end
                        temp = All_Population.decs;
                    end
                    CA = UpdateCA(CA,Offspring,CAsize);
                    DA = UpdateDA(DA,Offspring,Problem.N,p);
                    Ho_Population = [Ho_Population,Offspring];
                end
            end
        end
    end
end
```

### `K_UpdateCA.m`
```matlab
function [CAobj,CAdec,CAvar] = K_UpdateCA(CAobj,CAdec,CAvar,MaxSize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    N  = size(CAobj,1);
    if N <= MaxSize
        return;
    end
    
    %% Calculate the fitness of each solution
    CAobj1 = (CAobj-repmat(min(CAobj),N,1))./(repmat(max(CAobj)-min(CAobj),N,1));
    I = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(CAobj1(i,:)-CAobj1(j,:));
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
    CAobj = CAobj(Choose,:);
    CAdec = CAdec(Choose,:);
    CAvar = CAvar(Choose,:);
end
```

### `K_UpdateDA.m`
```matlab
function [DAobj,DAdec,DAvar] = K_UpdateDA(DAobj,DAdec,DAvar,MaxSize,p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    DA_Nor_pre = (DAobj - repmat(min(DAobj,[],1),size(DAobj,1),1))./repmat(max(DAobj,[],1) - min(DAobj,[],1),size(DAobj,1),1);

    %% Find the non-dominated solutions
    ND = NDSort(DAobj,1);
    DAobj = DAobj((ND==1),:);
    DAdec = DAdec((ND==1),:);
    DAvar = DAvar((ND==1),:);
    DA_Nor_pre = DA_Nor_pre((ND==1),:);
    N  = size(DAobj,1);
    if N <= MaxSize
        return;
    end
    
    %% Select the extreme solutions first
    Choose = false(1,N);
    M      = size(DA_Nor_pre,2);
    select = randperm(M);
    Choose(select(1)) = true;

    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > MaxSize
        % Randomly delete several solutions
        Choosed = find(Choose);
        k = randperm(sum(Choose),sum(Choose)-MaxSize);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < MaxSize
        % Add several solutions by truncation strategy
        Distance = inf(N);
        for i = 1 : N-1
            for j = i+1 : N
                Distance(i,j) = norm(DA_Nor_pre(i,:)-DA_Nor_pre(j,:),p);
                Distance(j,i) = Distance(i,j);
            end
        end
        while sum(Choose) < MaxSize
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
    end
    DAobj = DAobj(Choose,:);
    DAdec = DAdec(Choose,:);
    DAvar = DAvar(Choose,:);
end
```

### `MatingSelection_KTA2.m`
```matlab
function [ParentCobj,ParentCdec,ParentMobj,ParentMdec] = MatingSelection_KTA2(CAobj,CAdec,DAobj,DAdec,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    CAParent1 = randi(size(CAobj,1),1,ceil(N/2));
    CAParent2 = randi(size(CAobj,1),1,ceil(N/2));
    Dominate  = any(CAobj(CAParent1,:)<CAobj(CAParent2,:),2) - any(CAobj(CAParent1,:)>CAobj(CAParent2,:),2);  
    ParentCobj   = [CAobj([CAParent1(Dominate==1),CAParent2(Dominate~=1)],:);...
                 DAobj(randi(size(DAobj,1),1,ceil(N/2)),:)];
    ParentCdec = [CAdec([CAParent1(Dominate==1),CAParent2(Dominate~=1)],:);...
                 DAdec(randi(size(DAdec,1),1,ceil(N/2)),:)];
    ParentMobj   = CAobj(randi(size(CAobj,1),1,N),:);
    ParentMdec   = CAdec(randi(size(CAdec,1),1,N),:);
end
```

### `UpdateCA.m`
```matlab
function CA = UpdateCA(CA,New,MaxSize)
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
    I = zeros(N);
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
function DA = UpdateDA(DA,New,MaxSize,p)
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
    ND = NDSort(DA.objs,1);
    DA = DA(ND==1);
    N  = length(DA);
    if N <= MaxSize
        return;
    end
    
    %% Select the extreme solutions first
    Choose = false(1,N);
    [~,Extreme1] = min(DA.objs,[],1);
    [~,Extreme2] = max(DA.objs,[],1);
    Choose(Extreme1) = true;
    Choose(Extreme2) = true;
    
    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > MaxSize
        % Randomly delete several solutions
        Choosed = find(Choose);
        k = randperm(sum(Choose),sum(Choose)-MaxSize);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < MaxSize
        % Add several solutions by truncation strategy
        Distance = inf(N);
        for i = 1 : N-1
            for j = i+1 : N
                Distance(i,j) = norm(DA(i).obj-DA(j).obj,p);
                Distance(j,i) = Distance(i,j);
            end
        end
        while sum(Choose) < MaxSize
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
    end
    DA = DA(Choose);
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
    if min(sum(abs(D),2) ) == 0
        error('Multiple design sites are not allowed')
    end
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

### `signrank_new.m`
```matlab
function [p, h, stats,r1,r2] = signrank_new(x,y,varargin)
%SIGNRANK Wilcoxon signed rank test for zero median.
%   P = SIGNRANK(X) performs a two-sided signed rank test of the hypothesis
%   that the data in the vector X come from a distribution whose median
%   (and mean, if it exists) is zero, and returns the p-value from the
%   test.  P is the probability of observing the given result, or one more
%   extreme, by chance if the null hypothesis ("median is zero") is true.
%   Small values of P cast doubt on the validity of the null hypothesis.
%   The data are assumed to come from a continuous distribution, symmetric
%   about its median.
%
%   P = SIGNRANK(X,M) performs a two-sided test of the hypothesis that the
%   data in the vector X come from a distribution whose median is M.  M
%   must be a scalar.
%
%   P = SIGNRANK(X,Y) performs a paired, two-sided test of the hypothesis
%   that the difference between the matched samples in the vectors X and Y
%   comes from a distribution whose median is zero.  The differences X-Y
%   are assumed to come from a continuous distribution, symmetric about its
%   median.  X and Y must be the same length.  The two-sided p-value is
%   computed by doubling the most significant one-sided value.
%
%   SIGNRANK treats NaNs in X or Y as missing values, and removes them.
%
%   [P,H] = SIGNRANK(...) returns the result of the hypothesis test,
%   performed at the 0.05 significance level, in H.  H=0 indicates that
%   the null hypothesis ("median is zero") cannot be rejected at the 5%
%   level. H=1 indicates that the null hypothesis can be rejected at the
%   5% level.
%
%   [P,H] = SIGNRANK(...,'alpha',ALPHA) returns the result of the hypothesis
%   test performed at the significance level ALPHA.
%
%   [P,H] = SIGNRANK(...,'method',METHOD) computes the p-value using an
%   exact algorithm if METHOD is 'exact', or a normal approximation if
%   METHOD is 'approximate'.  The default is to use an exact method for
%   small samples.
%
%   [P,H,STATS] = SIGNRANK(...) returns STATS, a structure with one or two
%   fields.  The field 'signedrank' contains the value of the signed rank
%   statistic.  If P is calculated using a normal approximation, then the
%   field 'zval' contains the value of the normal (Z) statistic.
%
%   See also SIGNTEST, RANKSUM, TTEST, ZTEST.

%   For the two-sample case, SIGNRANK uses a tolerance based on the
%   values EPSD=EPS(X)+EPS(Y). Any pair of values of D=X-Y that differ by
%   no more than the sum of their two EPSD values are treated as ties.

%   References:
%      [1] Hollander, M. and D. A. Wolfe.  Nonparametric Statistical
%          Methods. Wiley, 1973.
%      [2] Gibbons, J.D.  Nonparametric Statistical Inference,
%          2nd ed.  M. Dekker, 1985.


%   Copyright 1993-2011 The MathWorks, Inc. 
%   $Revision: 1.1.8.3 $  $Date: 2011/05/09 01:26:51 $

% Check most of the inputs now
alpha = 0.05;
if nargin>2 && isnumeric(varargin{1})
   % Grandfathered syntax:  signrank(x,y,alpha)
   alpha = varargin{1};
   varargin(1) = [];
end
oknames = {'alpha' 'method'};
dflts   = {alpha   ''};
[alpha,method] = internal.stats.parseArgs(oknames,dflts,varargin{:});

if ~isscalar(alpha)
   error(message('stats:signrank:BadAlpha'));
end
if ~isnumeric(alpha) || isnan(alpha) || (alpha <= 0) || (alpha >= 1)
   error(message('stats:signrank:BadAlpha'));
end

onesample = false;
if nargin < 2 || isempty(y)
    y = zeros(size(x));
    onesample = true;
elseif isscalar(y)
    y = repmat(y, size(x));
end

if ~isvector(x) || ~isvector(y)
    error(message('stats:signrank:InvalidData'));
elseif numel(x) ~= numel(y)
    error(message('stats:signrank:InputSizeMismatch'));
end

diffxy = x(:) - y(:);
if onesample
    epsdiff = zeros(size(x(:)));
else
    epsdiff = eps(x(:)) + eps(y(:));
end

% Remove missing data
t = isnan(diffxy);
diffxy(t) = [];
epsdiff(t) = [];
if isempty(diffxy)
   error(message('stats:signrank:NotEnoughData'));
end

t = (abs(diffxy) < epsdiff);
diffxy(t) = [];
epsdiff(t) = [];

n = length(diffxy);

if (n == 0)         % degenerate case, all ties
    p = 1;
    if (nargout > 1)
        h = 0;
        if (nargout > 2)
            stats.signedrank = 0;
        end
    end
%     return
end

% Now deal with the method argument
if isempty(method)
   if n<=15
      method = 'exact';
   else
      method = 'approximate';
   end
elseif isequal(lower(method),'oldexact')
    % OK
else
   method = internal.stats.getParamVal(method,{'exact' 'approximate'},'''method''');
end

% Find negative differences and ranks of absolute differences
neg = (diffxy<0);
[tierank, tieadj] = tiedrank(abs(diffxy),0,0,epsdiff);

% Compute signed rank statistic (most extreme version)
w = sum(tierank(neg));
r1=w;
r2=n*(n+1)/2-w;
w = min(w, n*(n+1)/2-w);

if isequal(method,'approximate')
    z = (w-n*(n+1)/4) / sqrt((n*(n+1)*(2*n+1) - tieadj)/24);
    p = 2*normcdf(z,0,1);
    if (nargout > 2)
        stats.zval = z;
    end
elseif isequal(method,'oldexact')
    % Enumerates all possibilities and does not adjust for ties
    allposs = (ff2n(n))';
    idx = (1:n)';
    idx = idx(:,ones(2.^n,1));
    pranks = sum(allposs.*idx,1);
    tail = 2*length(find(pranks <= w)); % two side.

    % Avoid p>1 if w is in the middle and is double-counted
    p = min(1, tail./(2.^n));
else  %isequal(method,'exact')
    p = statsrexact(tierank,w);

    p = min(1, 2*p);   % two-sided, don't double-count the middle value
end

if nargout > 1
    h = (p<=alpha);
    if (nargout > 2)
        stats.signedrank = w;
    end
end
```

### `statsrexact.m`
```matlab
function [pval,P] = statsrexact(v,w)
%STATSREXACT Compute exact tail probability for signed rank statistic.
%   [PVAL,ALLP]=STATSREXACT(V,W) computes the tail probability PVAL
%   for the statistic W with the vector V of ranks.  ALLP is a matrix
%   containing the probabilities (col. 2) for each W value (col. 1).
%
%   Private function used by the SIGNRANK function.

%   Copyright 2003-2006 The MathWorks, Inc. 


n = length(v);
v = sort(v(:)'); % make sure it's a row

% For convenience we can just compute the lower tail.  If w is
% in the upper tail, compute its equivalent lower tail value.
maxw = n*(n+1)/2;
folded = (w>maxw/2);
if folded
   w = maxw-w;
end

% We would like to use the elements of w and v as indexes into
% arrays that enumerate possible values.  If there are ties causing
% non-integer ranks, multiply by 2 to force everything to integer.
doubled = any(v~=floor(v));
if doubled
   v = round(2*v);
   w = round(2*w);
end

C = zeros(w+1,1);  % C(w+1) will be the number of combinations adding
                   % to w at each step
C(1) = 1;          % just one combination includes nothing
top = 1;           % top entry currently in use in C vector

% Look at all combinations of ranks that could contribute
% to the observed value of W
for vj=v(v<=w)

   % C now enumerates combinations not including v(j).  Now update the
   % elements that could include v(j).
   newtop = min(top+vj,w+1);
   hi = min(vj,w+1)+1:newtop;
   lo = 1:length(hi);

   C(hi) = C(hi) + C(lo);

   top = newtop;
end

% Convert to probabilities
C = C / (2^n);

% Get tail probability
pval = sum(C);

if nargout>1
   allw = 0:w;
   if doubled
      allw = allw/2;
   end
   if folded
      allw = n*(n+1)/2 - allw;
   end
   
   P = [allw(:), C(:)];
end
```
