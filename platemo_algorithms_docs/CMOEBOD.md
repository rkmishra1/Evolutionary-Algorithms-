# CMOEBOD

**Tags**: <2025> <multi/many> <real> <expensive> <constrained>

## Description
Constrained multiobjective evolutionary Bayesian optimization based on decomposition

## Reference
Z. Zhang, Y. Wang, G.Sun, and T. Pang. A novel evolutionary Bayesian optimization algorithm based on decomposition for expensive constrained multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2025.

## Source Code

### `CMOEBOD.m`
```matlab
classdef CMOEBOD < ALGORITHM
% <2025> <multi/many> <real> <expensive> <constrained>
% Constrained multiobjective evolutionary Bayesian optimization based on decomposition
% wmax --- 20 --- Number of generations before updating Kriging models
% mu   ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% Z. Zhang, Y. Wang, G.Sun, and T. Pang. A novel evolutionary Bayesian 
% optimization algorithm based on decomposition for expensive constrained 
% multiobjective optimization problems. IEEE Transactions on Systems, Man, 
% and Cybernetics: Systems, 2025.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao.zhang.cn@gmail.com)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [wmax,mu] = Algorithm.ParameterSet(20,5);

            %% Generate the reference points and population
            global V beta NI 
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            NI            = Problem.N;
            P             = UniformPoint(NI,Problem.D,'Latin');
            DB            = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            Population    = DB;           
            THETA_obj     = 5.*ones(Problem.M,Problem.D);
            THETA_con     = 5.*ones(size(DB.cons,2),Problem.D);
            beta          = 0.1;

            %% Optimization
            while Algorithm.NotTerminated(DB)
                % Model Construction
                [Model_obj,Model_con,THETA_obj,THETA_con] = model_train(DB,THETA_obj,THETA_con);
                
                % Collaborative Evolutionary Optimization
                P = optimizaiton(Population,wmax,Model_obj,Model_con,Problem);
                
                % Bilevel Candidate Selection
                DB = NewSelect(P,DB,mu,Problem);   
                
                % Update Population
                Population = PUpdate(DB,NI);           
            end
        end
    end
end

function [Model_obj,Model_con,THETA_obj,THETA_con] = model_train(A2,THETA_obj,THETA_con)
    Dec = A2.decs;
    Obj = A2.objs;
    Con = A2.cons;
    Len_dec = size(Dec,2);
    Len_obj = size(Obj,2);
    Len_con = size(Con,2);
    for i = 1 : Len_obj
        [~,distinct1] = unique(round(Dec*1e100)/1e100,'rows');
        [~,distinct2] = unique(round(Obj(:,i)*1e100)/1e100,'rows');
        distinct      = intersect(distinct1,distinct2);
        
        dmodel         = dacefit(Dec(distinct,:),Obj(distinct,i),'regpoly1','corrgauss',THETA_obj(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
        Model_obj{i}   = dmodel;
        THETA_obj(i,:) = dmodel.theta;
    end
    for i = 1 : Len_con
        [~,distinct1] = unique(round(Dec*1e100)/1e100,'rows');
        [~,distinct2] = unique(round(Con(:,i)*1e100)/1e100,'rows');
        distinct      = intersect(distinct1,distinct2);
        
        dmodel         = dacefit(Dec(distinct,:),Con(distinct,i),'regpoly1','corrgauss',THETA_con(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
        Model_con{i}   = dmodel;
        THETA_con(i,:) = dmodel.theta;
    end
end

function [OffObj,Off_ObjMSE,OffCon,Off_ConMSE] = model_predict(Model_obj,Model_con,OffDec)
    N          = size(OffDec,1);
    Len_obj    = length(Model_obj);
    Len_con    = length(Model_con);
    OffObj     = zeros(N,Len_obj);
    OffCon     = zeros(N,Len_con);
    Off_ObjMSE = zeros(N,Len_obj);
    Off_ConMSE = zeros(N,Len_con);
    for i = 1 : N
        for j = 1 : Len_obj
            [OffObj(i,j),~,Off_ObjMSE(i,j)] = predictor(OffDec(i,:),Model_obj{j});
        end
        for j = 1 : Len_con
            [OffCon(i,j),~,Off_ConMSE(i,j)] = predictor(OffDec(i,:),Model_con{j});
        end
    end
    OffObj     = real(OffObj);
    OffCon     = real(OffCon);
    Off_ObjMSE = abs(real(Off_ObjMSE));
    Off_ConMSE = abs(real(Off_ConMSE));
end

function P = optimizaiton(Population,wmax,Model_obj,Model_con,Problem)
    global V 
    w      = 1;
    P.decs = Population.decs;
    while w <= wmax    
        OffDec   = OperatorGA(Problem,P.decs);
        P.decs   = [P.decs;OffDec];
        [P.objs,P.objmse,P.cons,P.conmse] = model_predict(Model_obj,Model_con,P.decs);
        index    = EnvironmentalSelection(P,V);
        P.decs   = P.decs(index,:);
        P.objs   = P.objs(index,:);
        P.cons   = P.cons(index,:);
        P.objmse = P.objmse(index,:);
        P.conmse = P.conmse(index,:);
        w        = w + 1;
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Next = EnvironmentalSelection(P,V)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao.zhang.cn@gmail.com)

    %% Data Preprocess
    NV     = size(V,1);
    NP     = size(P.objs,1);
    PopObj = P.objs;                                                                                                                                                                                                                                            
    zmin   = min(PopObj,[],1); zmax = max(PopObj,[],1);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    ObjMSE = P.objmse./(max(zmax - zmin,10e-10).^2);
    PopCon = P.cons; 
    ConMSE = P.conmse;

    %% Associate each solution to a reference vector
    Angle  = acos(1-pdist2(PopObj,V,'cosine'));
    Pindex = true(1,NP);
    Vindex = true(1,NV);
    
    %% Select one solution for each reference vector
    while any(Vindex)
        [~,associate] = min(Angle(Pindex,Vindex),[],2);
        Pexist = find(Pindex==1); 
        Vexist = find(Vindex==1); 
        for i = unique(associate)'
            current = find(associate==i);
            if isscalar(current)
                best = 1;
            else
                best = CPoB(PopObj(Pexist(current),:),ObjMSE(Pexist(current),:),...
                    PopCon(Pexist(current),:),ConMSE(Pexist(current),:),V(Vexist(i),:));
            end
            Pindex(Pexist(current(best))) = 0;
            Vindex(Vexist(i)) = 0;
        end
    end
    Next = Pindex==0;
end

function best = CPoB(PopObj,ObjMSE,PopCon,ConMSE,lamda)
    % Approximate Scalar
    PoF    = Feasible_Probability(PopCon,ConMSE);
    [N,M]  = size(PopObj);
    u      = lamda.*PopObj;
    sigma2 = (lamda.^2).*ObjMSE;
    sigma2 = abs(real(sigma2));
    mu     = u(:,1:2); sig2 = sigma2(:,1:2);
    [y,x]  = GPcal(mu,abs(sig2));
    if M >= 3
        for i = 3 : M
            mu = [y,u(:,i)]; sig2 = [x,sigma2(:,i)];
            [y,x] = GPcal(mu,abs(sig2));
        end
    end  
    y1 = sum(lamda.*PopObj,2);
    x1 = sum((lamda.^2).*ObjMSE,2);
    y  = y + 0.05*y1;
    x  = x + 0.05^2*x1;
   
    % Probabilistic Sorting
    sigma    = sqrt(x(reshape(ones(N,1)*(1:N),N*N,1),:) + repmat(x,N,1));
    mean     = y(reshape(ones(N,1)*(1:N),N*N,1),:) - repmat(y,N,1);
    mean(all(mean==0,2),:) = ones(sum(all(mean==0,2)),1);
    x_PD     = normcdf((0-mean)./abs(real(sigma)));
    y_PD     = 1 - x_PD;
    x_PD     = - x_PD.*PoF(reshape(ones(N,1)*(1:N),N*N,1),:);
    y_PD     = - y_PD.*repmat(PoF,N,1);
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
    dominate = sum(dominate,1);
    best     = find(dominate==min(dominate));
    if length(best) > 1
       best = best(randperm(length(best),1));
    end
end

function [y,x] = GPcal(mu,sig2)
    % Calculate the mu (x) and sigma^2 (y) of the aggregation function
    tao   = sqrt(sig2(:,1)+sig2(:,2));
    alpha = (mu(:,1)-mu(:,2))./tao;
    y     = mu(:,1).*normcdf(alpha) + mu(:,2).*normcdf(-alpha) + tao.*normpdf(alpha);
    x     = (mu(:,1).^2+sig2(:,1)).*normcdf(alpha) + (mu(:,2).^2+sig2(:,2)).*normcdf(-alpha)...
        + sum(mu,2).*tao.*normpdf(alpha) - y.^2;
end

function PoF = Feasible_Probability(PopCon,ConMSE)
    [N,M] = size(PopCon);
    PoF   = ones(N,1);
    for i = 1 : N
        for j = 1 : M
            PoF(i) = PoF(i) * normcdf((0-PopCon(i,j))/sqrt(ConMSE(i,j)));
        end
    end
end
```

### `NewSelect.m`
```matlab
function DB = NewSelect(P,DB,mu,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao.zhang.cn@gmail.com)

    %% Preparing Data  
    index = [];
    for i = 1 : size(P.decs,1)
        dist2 = pdist2(real(P.decs(i,:)),real(DB.decs));
        if min(dist2) > 1e-5
            index =[index,i];
        end
    end
    if isempty(index)
        return; 
    elseif length(index) <= mu
       PopNew = P.decs(index,:);
       DB     = [DB,Problem.Evaluation(PopNew)];
       return; 
    end
    
    PopDec = P.decs(index,:);
    PopObj = P.objs(index,:);ObjMSE = P.objmse(index,:);
    PopCon = P.cons(index,:);ConMSE = P.conmse(index,:);
    
    A2Obj = DB.objs;
    A2Obj = unique(A2Obj,'rows');
    
    zmin   = min([A2Obj;PopObj],[],1); zmax = max([A2Obj;PopObj],[],1);
    A2Obj  = (A2Obj - zmin)./max(zmax - zmin,10e-10);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    ObjMSE = ObjMSE./(max(zmax - zmin,10e-10).^2);
    
    %% Selection
    global beta NI
    Pindex = true(1,size(PopObj,1));
    while length(find(Pindex==0)) < mu 
        % Upper Level: Diversity Contribution
        Pexist   = find(Pindex==1);
        Angle    = acos(1-pdist2(PopObj(Pindex,:),A2Obj,'cosine'));
        Angle    = sort(Angle,2);
        Angle    = Angle(:,1);
        [~,Rank] = sort(Angle,'descend');
        index    = Rank(1:min([ceil(beta*NI),length(find(Pindex==1))]));
        
        % Lower Level: Convergence and Feasibility Contributions
        L       = Pexist(index);
        PopDec_ = PopDec(L,:);
        PopObj_ = PopObj(L,:);ObjMSE_ = ObjMSE(L,:);
        PopCon_ = PopCon(L,:);ConMSE_ = ConMSE(L,:);
        PoF     = Feasible_Probability(PopCon_,ConMSE_);
        Pro = zeros(1,size(PopDec_,1));
        for i = 1 : size(PopDec_,1)
            mean   = PopObj_(i,:) - A2Obj;
            sigma  = sqrt(ObjMSE_(i,:));
            Pro(i) = -sum(prod(normcdf((0-mean)./sigma),2))/size(A2Obj,1).*PoF(i);
        end
        [~,Rank] = sort(Pro);
        DB       = [DB,Problem.Evaluation(PopDec(L(Rank(1)),:))];
        A2Obj    = (DB.objs - zmin)./max(zmax - zmin,10e-10);
        Pindex(L(Rank(1))) = 0; 
    end
end

function PoF = Feasible_Probability(PopCon,ConMSE)
    [N,M] = size(PopCon);
    PoF   = ones(N,1);
    for i = 1 : N
        for j = 1 : M
            PoF(i) = PoF(i) * normcdf((0-PopCon(i,j))/sqrt(ConMSE(i,j)));
        end
    end
end
```

### `PUpdate.m`
```matlab
function Population = PUpdate(Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiyao Zhang (email: zhiyao.zhang.cn@gmail.com)

    %% Data Preprocess
    global V
    PopObj = Population.objs ; PopCon = Population.cons;
    zmin   = min(PopObj,[],1); zmax = max(PopObj,[],1);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    NV     = size(V,1);
    NP     = size(PopObj,1);

    %% Associate each solution to a reference vector
    Angle  = acos(1-pdist2(PopObj,V,'cosine'));
    Pindex = true(1,NP);
    Vindex = true(1,NV);
    
    %% Select one solution for each reference vector
    while any(Vindex)
        [~,associate] = min(Angle(Pindex,Vindex),[],2);
        Pexist = find(Pindex==1); 
        Vexist = find(Vindex==1); 
        for i = unique(associate)'
            current = find(associate==i);
            if isscalar(current)
                best = 1;
            else
                best = scalar_select(PopObj(Pexist(current),:),PopCon(Pexist(current),:),V(Vexist(i),:));
            end
            Pindex(Pexist(current(best))) = 0;
            Vindex(Vexist(i)) = 0;
        end
    end

    %% Select remain solutions
    if N > NV
        Pexist = find(Pindex==1);
        index  = randperm(length(Pexist),N-NV);
        Pindex(Pexist(index)) = 0;
    end
    Next = Pindex==0;
    
    %% Population for next generation
    Population = Population(Next);
end

function best = scalar_select(PopObj,PopCon,lamda)
    g    = max(PopObj.*lamda,[],2);
    CV   = sum(max(0,PopCon),2);
    best = find(CV==min(CV,[],1));
    if length(best) > 1
        [~,best_] = min(g(best));
        best = best(best_);
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
