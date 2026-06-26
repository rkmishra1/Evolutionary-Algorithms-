# ESB-CEO

**Tags**: <2023> <multi> <real> <expensive>

## Description
Bayesian co-evolutionary optimization based entropy search

## Reference
H. Bian, J. Tian, J. Yu, and H. Yu. Bayesian co-evolutionary optimization based entropy search for high-dimensional many-objective optimization. Knowledge-Based Systems, 2023, 274: 110630.

## Source Code

### `EGOSelect.m`
```matlab
function PopDec = EGOSelect(Problem,Population,L1,L2,Ke,delta,nr)
% Selecting Points for Function Evaluation with the assistance of the
% Gaussian models

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    %% Fuzzy clustering the solutions
    [model,centers] = FCMmodel(Problem,Population,L1,L2);
    %% MOEA/D-DE optimization, where the popsize is set to N, the maximum evaluations is maxeva
	Z       = min(Population.objs,[],1);
    maxeva  = 20;
    eva=1;
    
	%% Generate the weight vectors
    [W, N]  = UniformPoint(Problem.N,Problem.M);
    T       = ceil(N/10);
    B       = pdist2(W,W);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);
    duplicated = randi(length(Population),N,1);
    NewPop  = Population(duplicated);  % Sample N individuals from the initial population
    PopDec  = NewPop.decs; PopObj = NewPop.objs;
    gmin    = inf;

    [V0,N1] = UniformPoint(length(Population),Problem.M);
    V             = V0;
    alpha=2;
    fr=0.1;
    Population2=Population.decs;
    ObjValue2=Population.objs;
    while (eva<=maxeva)
    %%MOEAD/DE
        for i = 1 : N
            if rand < delta
                P = B(i,randperm(size(B,2)));
            else
                P = randperm(N);
            end
            OffDec = OperatorDE(Problem,PopDec(i,:),PopDec(P(1),:),PopDec(P(2),:));
            OffObj = Evaluate(Problem,OffDec,model,centers);
            Z = min(Z,OffObj);
            g_old = max(abs(PopObj(P,:) - repmat(Z,length(P),1)).*W(P,:),[],2);
            g_new = max(repmat(abs(OffObj-Z),length(P),1).*W(P,:),[],2);
            gmin = min([gmin,min(g_old),min(g_new)]);
            offindex = P(find(g_old>g_new,nr));
            if ~isempty(offindex)
                PopDec(offindex,:) = repmat(OffDec,length(offindex),1);
                PopObj(offindex,:) = repmat(OffObj,length(offindex),1);
            end
        end
        Pop1 = PopDec;
        Obj1 = PopObj;
        
        
       %% RVEA
        MatingPool = randi(length(Population2),1,N1);
        OffDec  = GABound(Population2(MatingPool',:),[Problem.upper;Problem.lower]);
        OffObj = Evaluate(Problem,OffDec,model,centers);
        PopDec=[Population2;OffDec];
        PopObj=[ObjValue2;OffObj];
        index = EnvironmentalSelection(ObjValue2,V,(Problem.FE/Problem.maxFE)^alpha);
        PopDec = PopDec(index,:);
        PopObj = PopObj(index,:);
        if ~mod(ceil(Problem.FE/N1),ceil(fr*Problem.maxFE/N1))
            V(1:N1,:) = ReferenceVectorAdaptation(PopObj,V0);
        end

        Pop2 = PopDec;
        Obj2 = PopObj;        
       
        if mod(eva,3)==0
           [FrontNo1,~]=NDSort(Obj1,inf);
           [FrontNo2,~]=NDSort(Obj2,inf);
            tag=1;
            PopDec=[Pop1(FrontNo1==1,:);Pop2(FrontNo2==1,:)];
            PopObj=[Obj1(FrontNo1==1,:);Obj2(FrontNo2==1,:)];
            if size(PopDec,1)>N
                d=zeros(size(PopDec,1),1);
                for i=1:size(PopDec,1)
                    for j=1:Problem.M
                        d(i)=d(i)+abs(PopObj(i,j))^(1/Problem.M);
                    end
                end
                dist=d.^(Problem.M);
                [dist,Index]=sort(dist);
                Dec=[];
                Obj=[];
                for j=1:N
                    Dec =[Dec;PopDec(Index==j,:)];
                    Obj =[Obj;PopObj(Index==j,:)];
                end
                PopDec=Dec;
                PopObj=Obj;

            else
                while size(PopDec,1)+size(Obj1(FrontNo1==tag+1,:),1)+size(Obj2(FrontNo2==tag+1,:),1)<N
                    tag=tag+1;
                    PopDec = [PopDec;Pop1(FrontNo1==tag,:);Pop2(FrontNo2==tag,:)];
                    PopObj = [PopObj;Obj1(FrontNo1==tag,:);Obj2(FrontNo2==tag,:)];
                end
                Dec=[Pop1(FrontNo1==tag+1,:);Pop2(FrontNo2==tag+1,:)];
                Obj=[Obj1(FrontNo1==tag+1,:);Obj2(FrontNo2==tag+1,:)];
                [Data,~]=ShanNonx([Dec Obj],[Dec Obj],size(Obj,2),10,10);
                [~,Index]=sort(Data,'descend');
                for j=1:N-size(PopDec,1)
                    PopDec =[PopDec;Dec(Index(j),:)];
                    PopObj =[PopObj;Obj(Index(j),:)];
                end
            end
        else
             PopDec=Pop1;
             PopObj=Obj1;  
        end
        Population2=PopDec;
        ObjValue2=PopObj;
        eva = eva +1;
    end
    
    %% Kmeans cluster the solutions into Ke clusters and select the solutions with the maximum EI in each cluster
    cindex  = kmeans(real(PopDec),Ke);
    Q = [];
    q=-0.5*cos(Problem.FE/Problem.maxFE*pi)+0.5;
    for i = 1 : Ke
        index = find(cindex == i); 
        temp = PopDec(index,:);
       
        [tempObj,~] = Evaluate(Problem,temp,model,centers);
        K = length(index);
        EI = zeros(K,1);
        [Data,~]=ShanNonx([temp tempObj],[temp tempObj],size(tempObj,2),10,10);
        for j = 1 : K
            EI(j) = EICal(tempObj(j,:),Data(j),1/size(tempObj,2),q);
        end
        [~,best] = max(EI);
        Q = [Q,index(best)];
    end
    PopDec = PopDec(Q,:);
end

function EI = EICal(Obj,s,p,q)
% Calculate the expected improvement

    M = size(Obj,2);
    d=0;
    for j=1:M
         d=d+abs(Obj(j))^p;
    end
    dist=d^(1/p);
    EI=(1-q)*dist-q*s;
end

function [y,x] = GPcal(lamda,mu,sig2)
% Calculate the mu (x) and sigma^2 (y) of the aggregation function

    tao = sqrt(lamda(1)^2*sig2(1) + lamda(2)^2*sig2(2));
    alpha = (mu(1)-mu(2))/tao;
    y = mu(1)*normcdf(alpha) + mu(2)*normcdf(-alpha) + tao*normpdf(alpha);
    x = (lamda(1)^2 + sig2(1))*normcdf(alpha) + ...
        (lamda(2)^2 + sig2(2))*normcdf(-alpha) + sum(lamda)*normpdf(alpha);
end

function [PopObj,MSE] = Evaluate(Problem,PopDec,model,centers)
% Predict the objective vector of the candidate solutions accodring to the
% Euclidean distance from each candidate solution to evaluated solutions
    
    D = pdist2(real(PopDec),real(centers));
    [~,index] = min(D,[],2);
    N = size(PopDec,1);
    PopObj = zeros(N,Problem.M);
    MSE = zeros(N,Problem.M);
    for i = 1 : N
        for j = 1 : Problem.M
            [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),model{index(i),j});
        end
    end
end

function [idx,dist] = nbselect(fitness,part,varargin)
    if varargin{1} == 'K'
        k = varargin{2};
        [idx,dist] = knnsearch(fitness(:,1:end),part(:,1:end),'Distance','euclidean','NSMethod','kdtree','K',k);  
    end
end
```

### `ESBCEO.m`
```matlab
classdef ESBCEO < ALGORITHM
% <2023> <multi> <real> <expensive>
% Bayesian co-evolutionary optimization based entropy search
% Ke    ---   5 --- The number of function evaluations at each generation
% delta --- 0.9 --- The probability of choosing parents locally
% nr    ---   2 --- Maximum number of solutions replaced by each offspring
% L1    ---  80 --- The maximal number of points used for building a local model
% L2    ---  20 --- The maximal number of points used for building a local model

%------------------------------- Reference --------------------------------
% H. Bian, J. Tian, J. Yu, and H. Yu. Bayesian co-evolutionary optimization
% based entropy search for high-dimensional many-objective optimization.
% Knowledge-Based Systems, 2023, 274: 110630.
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
            [Ke,delta,nr,L1,L2] = Algorithm.ParameterSet(5,0.9,2,80,20);

            %% Generate random population
            NI = 100+Problem.D/10;
            P  = UniformPoint(NI,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            L1 = min(L1,length(Population));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                PopDec     = EGOSelect(Problem,Population,L1,L2,Ke,delta,nr);
                Offspring  = Problem.Evaluation(PopDec);
                Population = [Population,Offspring];
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function index= EnvironmentalSelection(PopObj,V,theta)
% The environmental selection of RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    NV    = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);
    
    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current = find(associate==i);
        % Calculate the APD value of each solution
        APD = (1+M*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
        % Select the one with the minimum APD value
        [~,best] = min(APD);
        Next(i)  = current(best);
    end
    % Population for next generation
    index = Next(Next~=0);
end
```

### `FCMmodel.m`
```matlab
function [model,centers] = FCMmodel(Problem,Population,L1,L2)
% Fuzzy clustering-based method for modeling c_size* M models, where c_size
% is the number of clusters and M the number of objectives.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopDec      = Population.decs;
    csize       = 1 + ceil((length(Population)-L1)/L2);
    [centers,~] = fcm(PopDec,csize,[2 NaN 0.05 false]);
    dis         = pdist2(PopDec,centers);
    [~,index]   = sort(-dis);
    group       = index(1:L1,:);

    %% Build GP model of each objective for each cluster 
    model   = cell(csize,Problem.M);
    THETA   = 5.*ones(csize,Problem.M,Problem.D);
    for i   = 1 : csize
        temp = Population(group(:,i));
        PopDec = temp.decs;
        PopObj = temp.objs;
        for j = 1 : Problem.M
            dmodel = dacefit(PopDec,PopObj(:,j),...
                     'regpoly0','corrgauss',...
                     squeeze(THETA(i,j,:)),...
                     1e-5.*ones(1,Problem.D),...
                     100.*ones(1,Problem.D));
            model{i,j}   = dmodel;
            THETA(i,j,:) = dmodel.theta;
        end
    end
end
```

### `GABound.m`
```matlab
function Offspring = GABound(Parent,Boundary)
% Genetic operators for real encoding with specified bounds

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    proC = 1; disC = 20; proM = 1; disM = 20;
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
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
    % Polynomial mutation
    Upper = repmat(Boundary(1,:),N,1);
    Lower = repmat(Boundary(2,:),N,1);
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

### `LOF.m`
```matlab
function lof = LOF(dist)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    K         = 6;
    m         = size(dist,1);
    distance  = zeros(m,m);
    num       = zeros(m,m);
    kdistance = zeros(m,1);
    count     = zeros(m,1);
    reachdist = zeros(m,m);
    lrd       = zeros(m,1);
    lof       = zeros(m,1);

    for i = 1 : m 
        [distance(i,:),num(i,:)] = sort(dist(i,:),'ascend');
        kdistance(i)             = distance(i,K+1); 
        count(i)                 = -1;
        for j = 1 : m
            if dist(i,j) <= kdistance(i)
                count(i) = count(i)+1;
            end
        end
    end

    for i = 1 : m
        for j = 1 : i-1
            reachdist(i,j) = max(dist(i,j),kdistance(j));
            reachdist(j,i) = reachdist(i,j);
        end
    end

    for i = 1 : m
        sum_reachdist = 0;
        for j = 1 : count(i)
            sum_reachdist = sum_reachdist+reachdist(i,num(j+1));
        end
        lrd(i) = count(i)/sum_reachdist;
    end

    for i = 1 : m
        sumlrd = 0;
        for j = 1 : count(i)
            sumlrd = sumlrd+lrd(num(j+1))/lrd(i);
        end
        lof(i) = sumlrd/count(i);
    end
end
```

### `ReferenceVectorAdaptation.m`
```matlab
function V = ReferenceVectorAdaptation(PopObj,V)
% Reference vector adaption strategy

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    V = V.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V,1),1);
end
```

### `ShanNonx.m`
```matlab
function [H,dist]=ShanNonx(train,data,m,D,k)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    H = zeros(size(data,1),1);
    for i = 1 : size(data,1)
        [knn,dist] = nbselect(train(:,D+1:D+m),data(i,D+1:D+m),'K',k,m);
        sum        = zeros(1,D);
        Px         = zeros(size(knn,2),D);
        for p = 1 : D
            for q = 1 : size(knn,2)
                sum(p) = sum(p)+train(knn(q),p);
            end
        end
        for p = 1 : D
            for q = 1 : size(knn,2)                
                if sum(p) == 0
                   Px(q,p) = 1/size(knn,2);
                else
                   Px(q,p) = train(knn(q),p)/sum(p);
                end
                if p == D
                    if q == size(knn,2)
                        Px = NdSort(Px,size(knn,2),D);
                    end
                end
                if p == D
                    if Px(q,p) == 0
                        h = 0;
                    else
                        h = Px(q,p)*log2(Px(q,p));
                    end
                    H(i) = H(i) - h;
                end
            end
        end
    end
end

function P = NdSort(Px,q,D)
    [~,I] = sort(Px);
    for i = 1 : q/2
        for j = 1 : D
            Px = swap(Px,I,j,i,q-i+1);
        end
    end
    P = Px;
end

function Px = swap(Px,I,j,i,p)
    t            = Px(I(i,j),j);
    Px(I(i,j),j) = Px(I(p,j),j);
    Px(I(p,j),j) = t;
end

function [idx,dist] = nbselect(fitness,part,varargin)
    if varargin{1} == 'K'
        k          = varargin{2};
        [idx,dist] = knnsearch(fitness(:,1:end),part(:,1:end),'Distance','euclidean','NSMethod','kdtree','K',k);  
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

### `distfcm.m`
```matlab
function out = distfcm(center, data)
%DISTFCM Distance measure in fuzzy c-mean clustering.
%	OUT = DISTFCM(CENTER, DATA) calculates the Euclidean distance
%	between each row in CENTER and each row in DATA, and returns a
%	distance matrix OUT of size M by N, where M and N are row
%	dimensions of CENTER and DATA, respectively, and OUT(I, J) is
%	the distance between CENTER(I,:) and DATA(J,:).
%
%       See also FCMDEMO, INITFCM, IRISFCM, STEPFCM, and FCM.

%	Roger Jang, 11-22-94, 6-27-95.
%       Copyright 1994-2016 The MathWorks, Inc. 

out = zeros(size(center, 1), size(data, 1));

% fill the output matrix

if size(center, 2) > 1
    for k = 1:size(center, 1)
	out(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2), 2));
    end
else	% 1-D data
    for k = 1:size(center, 1)
	out(k, :) = abs(center(k)-data)';
    end
end
```

### `fcm.m`
```matlab
function [center, U, obj_fcn] = fcm(data, cluster_n, options)
%FCM Data set clustering using fuzzy c-means clustering.
%
%   [CENTER, U, OBJ_FCN] = FCM(DATA, N_CLUSTER) finds N_CLUSTER number of
%   clusters in the data set DATA. DATA is size M-by-N, where M is the number of
%   data points and N is the number of coordinates for each data point. The
%   coordinates for each cluster center are returned in the rows of the matrix
%   CENTER. The membership function matrix U contains the grade of membership of
%   each DATA point in each cluster. The values 0 and 1 indicate no membership
%   and full membership respectively. Grades between 0 and 1 indicate that the
%   data point has partial membership in a cluster. At each iteration, an
%   objective function is minimized to find the best location for the clusters
%   and its values are returned in OBJ_FCN.
%
%   [CENTER, ...] = FCM(DATA,N_CLUSTER,OPTIONS) specifies a vector of options
%   for the clustering process:
%       OPTIONS(1): exponent for the matrix U             (default: 2.0)
%       OPTIONS(2): maximum number of iterations          (default: 100)
%       OPTIONS(3): minimum amount of improvement         (default: 1e-5)
%       OPTIONS(4): info display during iteration         (default: 1)
%   The clustering process stops when the maximum number of iterations
%   is reached, or when the objective function improvement between two
%   consecutive iterations is less than the minimum amount of improvement
%   specified. Use NaN to select the default value.
%
%   Example
%       data = rand(100,2);
%       [center,U,obj_fcn] = fcm(data,2);
%       plot(data(:,1), data(:,2),'o');
%       hold on;
%       maxU = max(U);
%       % Find the data points with highest grade of membership in cluster 1
%       index1 = find(U(1,:) == maxU);
%       % Find the data points with highest grade of membership in cluster 2
%       index2 = find(U(2,:) == maxU);
%       line(data(index1,1),data(index1,2),'marker','*','color','g');
%       line(data(index2,1),data(index2,2),'marker','*','color','r');
%       % Plot the cluster centers
%       plot([center([1 2],1)],[center([1 2],2)],'*','color','k')
%       hold off;
%
%   See also FCMDEMO, INITFCM, IRISFCM, DISTFCM, STEPFCM.

%   Roger Jang, 12-13-94, N. Hickey 04-16-01
%   Copyright 1994-2018 The MathWorks, Inc. 

if nargin ~= 2 && nargin ~= 3
	error(message("fuzzy:general:errFLT_incorrectNumInputArguments"))
end

data_n = size(data, 1);

% Change the following to set default options
default_options = [2;	% exponent for the partition matrix U
		100;	% max. number of iteration
		1e-5;	% min. amount of improvement
		1];	% info display during iteration 

if nargin == 2
	options = default_options;
else
	% If "options" is not fully specified, pad it with default values.
	if length(options) < 4
		tmp = default_options;
		tmp(1:length(options)) = options;
		options = tmp;
	end
	% If some entries of "options" are nan's, replace them with defaults.
	nan_index = find(isnan(options)==1);
	options(nan_index) = default_options(nan_index);
	if options(1) <= 1
		error(message("fuzzy:general:errFcm_expMustBeGtOne"))
	end
end

expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);		% Display info or not

obj_fcn = zeros(max_iter, 1);	% Array for objective function

U = initfcm(cluster_n, data_n);			% Initial fuzzy partition
% Main loop
for i = 1:max_iter
	[U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
	if display
		fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
	end
	% check termination condition
	if i > 1
		if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end
	end
end

iter_n = i;	% Actual number of iterations 
obj_fcn(iter_n+1:max_iter) = [];
```

### `initfcm.m`
```matlab
function U = initfcm(cluster_n, data_n)
%INITFCM Generate initial fuzzy partition matrix for fuzzy c-means clustering.
%   U = INITFCM(CLUSTER_N, DATA_N) randomly generates a fuzzy partition
%   matrix U that is CLUSTER_N by DATA_N, where CLUSTER_N is number of
%   clusters and DATA_N is number of data points. The summation of each
%   column of the generated U is equal to unity, as required by fuzzy
%   c-means clustering.
%
%       See also DISTFCM, FCMDEMO, IRISFCM, STEPFCM, FCM.

%   Roger Jang, 12-1-94.
%   Copyright 1994-2002 The MathWorks, Inc. 

U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);
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

### `stepfcm.m`
```matlab
function [U_new, center, obj_fcn] = stepfcm(data, U, cluster_n, expo)
%STEPFCM One step in fuzzy c-mean clustering.
%   [U_NEW, CENTER, ERR] = STEPFCM(DATA, U, CLUSTER_N, EXPO)
%   performs one iteration of fuzzy c-mean clustering, where
%
%   DATA: matrix of data to be clustered. (Each row is a data point.)
%   U: partition matrix. (U(i,j) is the MF value of data j in cluster j.)
%   CLUSTER_N: number of clusters.
%   EXPO: exponent (> 1) for the partition matrix.
%   U_NEW: new partition matrix.
%   CENTER: center of clusters. (Each row is a center.)
%   ERR: objective function for partition U.
%
%   Note that the situation of "singularity" (one of the data points is
%   exactly the same as one of the cluster centers) is not checked.
%   However, it hardly occurs in practice.
%
%       See also DISTFCM, INITFCM, IRISFCM, FCMDEMO, FCM.

%   Copyright 1994-2014 The MathWorks, Inc. 

mf = U.^expo;       % MF matrix after exponential modification
center = mf*data./(sum(mf,2)*ones(1,size(data,2))); %new center
dist = distfcm(center, data);       % fill the distance matrix
obj_fcn = sum(sum((dist.^2).*mf));  % objective function
tmp = dist.^(-2/(expo-1));      % calculate new U, suppose expo != 1
U_new = tmp./(ones(cluster_n, 1)*sum(tmp));
```
