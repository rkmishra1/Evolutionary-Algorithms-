# AB-SAEA

**Tags**: <2020> <multi/many> <real/integer> <expensive>

## Description
Adaptive Bayesian based surrogate-assisted evolutionary algorithm

## Reference
X. Wang, Y. Jin, S. Schmitt S, and M. Olhofer. An adaptive Bayesian approach to surrogate-assisted evolutionary multi-objective optimization. Information Sciences, 2020, 519: 317-331.

## Source Code

### `ABSAEA.m`
```matlab
classdef ABSAEA < ALGORITHM
% <2020> <multi/many> <real/integer> <expensive>
% Adaptive Bayesian based surrogate-assisted evolutionary algorithm
% alpha ---  2 --- The parameter controlling the rate of change of penalty
% wmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% X. Wang, Y. Jin, S. Schmitt S, and M. Olhofer. An adaptive Bayesian
% approach to surrogate-assisted evolutionary multi-objective optimization.
% Information Sciences, 2020, 519: 317-331.
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
            [alpha,wmax,mu] = Algorithm.ParameterSet(2,20,5);

            %% Generate the reference points and population
            [V0,Problem.N] = UniformPoint(Problem.N,Problem.M);
            V     = V0;
            V1    = V0;
            NI    = 11*Problem.D-1;
            P     = UniformPoint(NI,Problem.D,'Latin');
            A1    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));  
            L     = 11*Problem.D-1+25; % L = 300;
            THETA = 5.*ones(Problem.M,Problem.D);
            Model = cell(1,Problem.M);

            %% Optimization
            while Algorithm.NotTerminated(A1)
                % Refresh the model and generate promising solutions delete the multiple data
                [~,index]  = unique(A1.decs,'rows');
                A1Dec = A1.decs;
                A1Dec = A1Dec(index,:);                
                A1Obj = A1.objs;
                A1Obj = A1Obj(index,:);
                Numdata=size(A1Dec,1);
                % Limit the size of the data
                if Numdata > L
                    [FrontNo,~] = NDSort(A1Obj,Numdata);
                    [~,index] = sort(FrontNo);
                    A1Dec1 = A1Dec(index(1:floor(L/2)), :);
                    A1Obj1 = A1Obj(index(1:floor(L/2)), :);                    
                    A1Dec2 = A1Dec(index(1:L-floor(L/2)), :);
                    A1Obj2 = A1Obj(index(1:L-floor(L/2)), :);
                    index = randperm(size(A1Dec2,1));
                    A1Dec = [A1Dec1;A1Dec2(index(1:L-floor(L/2)),:)];
                    A1Obj = [A1Obj1;A1Obj2(index(1:L-floor(L/2)),:)];
                end
                for i = 1 : Problem.M
                    % The parameter 'regpoly1' refers to one-order polynomial
                    % function, and 'regpoly0' refers to constant function. The
                    % former function has better fitting performance but lower
                    % efficiency than the latter one  
                    [mS, mY]   = dsmerge(A1Dec, A1Obj(:,i));
                    dmodel     = dacefit(mS, mY,'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    Model{i}   = dmodel;
                    THETA(i,:) = dmodel.theta;                  
                end
                PopDec = A1Dec;
                w      = 1;
                while w <= wmax
                    drawnow('limitrate');
                    OffDec = OperatorGA(Problem,PopDec);
                    PopDec = [PopDec;OffDec];
                    [N,~]  = size(PopDec);
                    PopObj = zeros(N,Problem.M);
                    MSE    = zeros(N,Problem.M);
                    for i = 1: N
                        for j = 1 : Problem.M
                            [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Model{j});
                        end
                    end
                    Selection = FSelection(PopObj,V,(w/wmax)^alpha,2);
                    PopDec = PopDec(Selection,:);
                    PopObj = PopObj(Selection,:);
                    MSE=MSE(Selection,:);
                    % Adapt referece vectors
                    if(mod(w, ceil(wmax*0.1)) == 0)
                        V = V0.*repmat(max(A1Obj,[],1)-min(A1Obj,[],1),size(V0,1),1);
                    end
                    w = w + 1;
                end

                % Select mu solutions for re-evaluation acquisiion function
                a=-0.5*cos(Problem.FE*pi/Problem.maxFE)+0.5;
                b=0.5*cos(Problem.FE*pi/Problem.maxFE)+0.5;
                [MMSE,~]=max(MSE,[],1);
                [MPopObj,~]=max(PopObj,[],1);
                fit=PopObj./MPopObj*b+MSE./MMSE*a;
                % Select by the reference vectors
                if a>0.5
                    Flag=2;
                else
                    Flag=1;
                end                
                Selection = FSelection(fit,V1,(Problem.FE/Problem.maxFE)^alpha,Flag);
                PopNew1 = PopDec(Selection,:);
                if size(PopNew1,1)<mu
                    PopNew=PopNew1(:,1:Problem.D);
                elseif size(PopNew1,1)>=mu
                    PopNew=PopNew1(randperm(size(PopNew1,1),5),1:Problem.D);
                end
                
                % Adapt referece vectors
                if(mod(Problem.FE, ceil(Problem.maxFE*0.1)) == 0)
                    V1 = V0.*repmat(max(A1Obj,[],1)-min(A1Obj,[],1),size(V0,1),1);
                end
                PopNew1 = Problem.Evaluation(PopNew);
                A1 = [A1 PopNew1];
            end
        end
    end
end
```

### `FSelection.m`
```matlab
function index = FSelection(FunctionValue,V,theta0,Flag)
% The environmental selection of K-RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(FunctionValue);
    NV    = size(V,1);
    
    %% Translate the population
    FunctionValue = FunctionValue - repmat(min(FunctionValue,[],1),N,1);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(FunctionValue,V,'cosine'));
    [~,associate] = min(Angle,[],2);
    
    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current = find(associate==i);
        if Flag==2
            % Calculate the APD value of each solution
             AD = (1+M*theta0*Angle(current,i)/gamma(i)).*sqrt(sum(FunctionValue(current,:).^2,2));
        else
             % Calculate the Angle value of each solution 
             AD = M*Angle(current,i)/gamma(i);   
        end
        % Select the one with the minimum AD value
        [~,best] = min(AD);
        Next(i)  = current(best);
    end
    % Population for next generation
    index = Next(Next~=0);
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

### `dsmerge.m`
```matlab
function  [mS, mY] = dsmerge(S, Y, ds, nms, wtds, wtdy)
%DSMERGE  Merge data for multiple design sites.
%
% Call
%   [mS, mY] = dsmerge(S, Y)
%   [mS, mY] = dsmerge(S, Y, ds)
%   [mS, mY] = dsmerge(S, Y, ds, nms)
%   [mS, mY] = dsmerge(S, Y, ds, nms, wtds)
%   [mS, mY] = dsmerge(S, Y, ds, nms, wtds, wtdy)
%
% Input
% S, Y : Data points (S(i,:), Y(i,:)), i = 1,...,m
% ds   : Threshold for equal, normalized sites. Default is 1e-14.
% nms  : Norm, in which the distance is measured.
%        nms =  1 : 1-norm (sum of absolute coordinate differences)
%               2 : 2-norm (Euclidean distance) (default)
%        otherwise: infinity norm (max coordinate difference)      
% wtds : What to do with the S-values in case of multiple points.
%        wtds = 1 : return the mean value (default)
%               2 : return the median value
%               3 : return the 'cluster center'
% wtdy : What to do with the Y-values in case of multiple points.
%        wtdy = 1 : return the mean value (default)
%               2 : return the median value
%               3 : return the 'cluster center' value
%               4 : return the minimum value
%               5 : return the maximum value    
%
% Output
% mS : Compressed design sites, with multiple points merged
%      according to wtds
% mY : Responses, compressed according to wtdy

% hbn@imm.dtu.dk  
% Last update July 3, 2002

% Check design points
[m n] = size(S);  % number of design sites and their dimension
sY = size(Y);
if  min(sY) == 1,  Y = Y(:);   lY = max(sY);  sY = size(Y);
else,              lY = sY(1); end
if m ~= lY
  error('S and Y must have the same number of rows'), end

% Threshold
if  nargin < 3
  ds = 1e-14;
elseif  (ds < 0) | (ds > .5)
  error('ds must be in the range [0, 0.5]'), end

% Which measure
if  nargin < 4
  nms = 2;
elseif  (nms ~= 1) & (nms ~= 2)
  nms = Inf;
end

% What to do
if  nargin < 5
  wtds = 1;
else
  wtds = round(wtds);
  if  (wtds < 1) | (wtds > 3)
    error('wtds must be in the range [1, 3]'), end
end
if  nargin < 6
  wtdy = 1;
else
  wtdy = round(wtdy);
  if  (wtdy < 1) | (wtdy > 5)
    error('wtdy must be in the range [1, 5]'), end
end

% Process data
more = 1;
ladr = zeros(1,ceil(m/2));
while more
  m = size(S,1);
  D = zeros(m,m);
  
  % Normalize sites
  mS = mean(S);   sS = std(S);
  scS = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);
  
  % Calculate distances D (upper triangle of the symetric matrix)
  for k = 1 : m-1
    kk = k+1 : m;
    dk = abs(repmat(scS(k,:), m-k, 1) - scS(kk,:));
    if  nms == 1,      D(kk,k) = sum(dk,2);
    elseif  nms == 2,  D(kk,k) = sqrt(sum(dk.^2,2));
    else,              D(kk,k) = max(dk,[],2);      end
  end
  D = D + D'; % make D symetric
  
  % Check distances
  mult = zeros(1,m);
  for  j = 1 : m
    % Find the number of multiple sites in each column of D
    mult(j) = length(find(D(:,j) < ds));
  end
  % Find the first column with the maximum number of multiple sites
  [mmult jj] = max(mult);
  
  if  mmult == 1,  more = 0;
  else
    nm = 0;
    while  mmult > 1
      nm = nm + 1;  % no. of points to merge
      ladr(nm) = jj;
      
      % Merge point no jj and its neighbours, note that jj is the center
      % of the cluster, as it has the most neighbors (among the multiple sites)
      ngb = find(D(:,jj) < ds);
      
      switch  wtds
      case 1,  S(jj,:) = mean(S(ngb,:));
      case 2,  S(jj,:) = median(S(ngb,:));
      case 3,  S(jj,:) = S(jj,:);
      end
      
      switch  wtdy
      case 1,  Y(jj,:) = mean(Y(ngb,:));
      case 2,  Y(jj,:) = median(Y(ngb,:));
      case 3,  Y(jj,:) = Y(jj,:);
      case 4,  Y(jj,:) = min(Y(ngb,:));
      case 5,  Y(jj,:) = max(Y(ngb,:));
      end
      
      % Delete from list
      mult(ngb) = 0;
      [mmult jj] = max(mult);
    end

    % Reduced data set
    act = [find(mult > 0)  ladr(1:nm)];
    S = S(act,:);    Y = Y(act,:);
  end % multiple
end % loop

% Return reduced set
mS = S;   mY = Y;
```
