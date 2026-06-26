# CI-EMO

**Tags**: <2026> <multi/many> <real> <expensive>

## Description
Composite indicator-guided infilling sampling for expensive multi-objective optimization

## Reference
H. Zhen, X. Li, W. Gong, and X. Hu. Composite indicator-guided infilling sampling for expensive multi-objective optimization. Swarm and Evolutionary Computation, 2026, 101: 102312.

## Source Code

### `CI.m`
```matlab
function [Popreal_Dec1, Popreal_Obj1, index] = CI(ND_TSDec,ND_TSObj,ND_PopDec,ND_PopObj,TSDec,TSObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    % I1 
    angle     = acos(1-pdist2((max(ND_PopObj,0)),max(ND_TSObj,0),'cosine'));   
    [Angle,~] = min(angle,[],2);                                            
        
    % Objective Normalization
    ND_PopObj = (ND_PopObj - min(TSObj,[],1))./(max(TSObj,[],1)-min(TSObj,[],1));
    TSObj     = (TSObj- min(TSObj,[],1))./(max(TSObj,[],1)-min(TSObj,[],1));
    Z         = min(TSObj,[],1);

    % I2 
    ddt         = pdist2(ND_PopObj, TSObj,'euclidean');
    ddt(ddt==0) = inf;
    CN          = min(ddt,[],2);

    % I3
    DC = pdist2(ND_PopObj, Z,'euclidean');

    % Indicators Normalization
    N_angle = (Angle - min(Angle,[],1))./(max(Angle,[],1)-min(Angle,[],1));
    N_CN    = (CN - min(CN,[],1))./(max(CN,[],1)-min(CN,[],1));
    N_DC    = (DC - min(DC,[],1))./(max(DC,[],1)-min(DC,[],1));
    
    % CI
    r1 = rand;
    r2 = rand;
    r3 = rand;
    CI = r1 * N_CN + r2 * N_angle - r3 * N_DC;
    
    % Selection
    [~, index]   = max(CI);
    Popreal_Dec1 = ND_PopDec(index,:);
    Popreal_Obj1 = ND_PopObj(index,:);
end
```

### `CIEMO.m`
```matlab
classdef CIEMO < ALGORITHM
% <2026> <multi/many> <real> <expensive>
% Composite indicator-guided infilling sampling for expensive multi-objective optimization

%------------------------------- Reference --------------------------------
% H. Zhen, X. Li, W. Gong, and X. Hu. Composite indicator-guided infilling
% sampling for expensive multi-objective optimization. Swarm and
% Evolutionary Computation, 2026, 101: 102312.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    methods
       function main(Algorithm,Problem)
            %% Population initialization
            Problem.N = 11*Problem.D-1;                                     % Initial sample size NI
            if Problem.N > 100
                Problem.N = 100;
            end
            [W, Problem.N] = UniformPoint(Problem.N,Problem.M);             % Generate reference point W
            P              = UniformPoint(Problem.N,Problem.D,'Latin');     % LHS initializes the population P, where normalized samples are generated
            Achieve        = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Problem.N,1).*P+repmat(Problem.lower,Problem.N,1)); % Population evaluation, get archive Achieve
            THETA          = 5.*ones(Problem.M,Problem.D);                  % GP Parameters
            
            %% Optimization       
            while Algorithm.NotTerminated(Achieve)
                % End
                if length(Achieve) >= Problem.maxFE
                    break;
                end

                % Train GP models
                DATA  = Achieve;
                TSDec = DATA.decs;
                TSObj = DATA.objs; 
                maxnumData = 1000;                                          % 11*Problem.D-1+5;
                numTS      = size(TSDec,1);
                if size(TSDec,1) >= maxnumData
                   trainX = TSDec(numTS-maxnumData+1:end,:);
                   trainY = TSObj(numTS-maxnumData+1:end,:);
                else
                   trainX = TSDec;
                   trainY = TSObj;
                end
                [trainX, trainY] = dsmerge(trainX, trainY);                 % Combining repetitive training points
                for i = 1 : Problem.M
                    dmodel     = dacefit(trainX,trainY(:,i),'regpoly1','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    Model{i}   = dmodel;
                    THETA(i,:) = dmodel.theta;
                end 

                % NSGAIII search
                [PopDec,PopObj,MSE] = NSGAIII_opt(Achieve,Model,W,Problem);

                % Candidate Seletction
                PopNew = CS(PopDec,PopObj,Achieve.decs,Achieve.objs);
                
                % Expensive evaluation
                New = [];
                if ~isempty(PopNew)                                         % Avoiding empty samples and removing duplicate points
                    [~,ib] = intersect(PopNew,Achieve.decs,'rows');
                    PopNew(ib,:) = [];
                    if ~isempty(PopNew)
                        New = Problem.Evaluation(PopNew);                   % Evaluation of sampling points obtained
                    else
                        [~,ib]       = intersect(PopDec,Achieve.decs,'rows');
                        PopDec(ib,:) = [];
                        [a1, a2]     = size(PopDec);
                        index        = randi(a2);
                        PopNew       = PopDec(index,:);
                        New          = Problem.Evaluation(PopNew);  
                    end
                end
                Achieve = cat(2,Achieve,New);                               % Adding evaluation points to the archive
            end
       end
    end
end
```

### `CS.m`
```matlab
function Popreal = CS(PopDec,PopObj,TSDec,TSObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

     % PopDec: candidate solutions, TSDec: modeling data
     % Non-dominated sorting of archived data
     [FrontNo,~] = NDSort(TSObj,inf);
     ND_TSObj    = TSObj(FrontNo==1,:);
     ND_TSDec    = TSDec(FrontNo==1,:);
     
     % Non-dominated Sorting of Candidate Solution Data
     [FrontNo1,~] = NDSort(PopObj,inf);
     ND_PopObj    = PopObj(FrontNo1==1,:);
     ND_PopDec    = PopDec(FrontNo1==1,:);   
     
     % parameters
     Popreal = [];
     rand1   = 1;

     %% Sampling Strategy
     if rand < rand1
         if ~isempty(PopDec)
             [Popreal_Dec1, Popreal_Obj1, index] = CI(ND_TSDec,ND_TSObj,ND_PopDec,ND_PopObj,TSDec,TSObj); % Composite indicator sampling
             Popreal = [Popreal; Popreal_Dec1];
         end
     end
end
```

### `GP_estimate.m`
```matlab
function [PopObj,PopObj_b,MSE,MSE_b] = GP_estimate(X,Model,numOBJ,beta)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    [N,~]  = size(X);
    PopObj = zeros(N,numOBJ);
    MSE    = zeros(N,numOBJ);
    for i = 1: N
        for j = 1 : numOBJ
            [PopObj(i,j),~,MSE(i,j)] = predictor(X(i,:),Model{j});
        end
    end
    MSE = max(MSE,0);
    S_  = sqrt(MSE);

    MSE      = S_; % AUCB
    PopObj_b = PopObj;
    MSE_b    = MSE;
    PopObj   = PopObj+beta*MSE; % AUCB
end
```

### `NSGAIII_opt.m`
```matlab
function [PopDec,PopObj,MSE] = NSGAIII_opt(Achieve,Model,W,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    PopDec = Achieve.decs;
    PopObj = Achieve.objs;
    Zmin   = min(PopObj,[],1);
    if size(PopDec,1) >= Problem.N
        Next   = nsga3EnvironmentalSelection(PopDec,PopObj,Problem.N,W,Zmin);
        PopDec = PopDec(Next,:);
        PopObj = PopObj(Next,:);
    end
    g    = 1;
    gmax = 20;
    beta = 0;
    while g <= gmax
        OffDec = generateOffing(PopDec,Problem,Achieve);
        PopDec = cat(1,PopDec,OffDec);
        [PopObj,~,MSE,~] = GP_estimate(PopDec,Model,Problem.M,beta);
        Zmin  = min([Zmin; PopObj],[],1);
        Choose = nsga3EnvironmentalSelection(PopDec,PopObj,Problem.N,W,Zmin);
        PopDec = PopDec(Choose,:);
        PopObj = PopObj(Choose,:);
        MSE    = MSE(Choose,:);
        g      = g + 1;
    end
end
```

### `dacefit.m`
```matlab
function  [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
%DACEFIT Constrained non-linear least-squares fit of a given correlation
% model to the provided data set and regression model
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
[m,n]   = size(S);  % number of design sites and their dimension
sY      = size(Y);
if  min(sY) == 1,
    Y = Y(:);  
    lY  = max(sY);  
    sY  = size(Y);
else       
    lY  = sY(1);
end
if m ~= lY
  error('S and Y must have the same number of rows')
end

% Check correlation parameters if it is given
lth     = length(theta0);
if  nargin > 5  % optimization case
  if  length(lob) ~= lth || length(upb) ~= lth
    error('theta0, lob and upb must have the same length')
  end
  if  any(lob <= 0) || any(upb < lob)
    error('The bounds must satisfy  0 < lob <= upb')
  end
else  % given theta
  if  any(theta0 <= 0)
    error('theta0 must be strictly positive')
  end
end

% Normalize data
mS = mean(S);   sS = std(S);
mY = mean(Y);   sY = std(Y);
% 02.08.27: Check for 'missing dimension'
j   = find(sS == 0);
if  ~isempty(j),  sS(j) = 1; end
j   = find(sY == 0);
if  ~isempty(j),  sY(j) = 1; end
S   = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);
Y   = (Y - repmat(mY,m,1)) ./ repmat(sY,m,1);

% Calculate distances D between points
mzmax = m*(m-1) / 2;        % number of non-zero distances
ij  = zeros(mzmax, 2);       % initialize matrix with indices
D   = zeros(mzmax, n);        % initialize matrix with distances
LL  = 0;
for k = 1 : m-1
  LL        = LL(end) + (1 : m-k);
  ij(LL,:)  = [repmat(k,m-k,1) (k+1:m)']; % indices for sparse matrix
  D(LL,:)   = repmat(S(k,:),m-k,1)-S(k+1:m,:); % differences between points
end
if  min(sum(abs(D),2) ) == 0
  error('Multiple design sites are not allowed'), end

% Regression matrix
    F       = feval(regr, S);  
    [mF,p]  = size(F);
if  mF ~= m
    error('number of rows in  F  and  S  do not match')
end
if  p > mF 
    error('least squares problem is underdetermined')
end

% parameters for objective function
par = struct('corr',corr, 'regr',regr, 'y',Y, 'F',F, ...
  'D', D, 'ij',ij, 'scS',sS);

% Determine theta
if  nargin > 5
  % Bound constrained non-linear optimization
  [theta, f, fit, perf] = boxmin(theta0, lob, upb, par);
  if  isinf(f)
    error('Bad parameter region.  Try increasing  upb'), end
else
  % Given theta
  theta = theta0(:);   
  [f,fit] = objfunc(theta, par);
  perf = struct('perf',[theta; f; 1], 'nv',1);
  if  isinf(f)
    error('Bad point.  Try increasing theta0'), end
end

% Return values
dmodel = struct('regr',regr, 'corr',corr, 'theta',theta.', ...
  'beta',fit.beta, 'gamma',fit.gamma, 'sigma2',sY.^2.*fit.sigma2, ...
  'S',S, 'Ssc',[mS; sS], 'Ysc',[mY; sY], ...
  'C',fit.C, 'Ft',fit.Ft, 'G',fit.G);

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================

function  [obj, fit] = objfunc(theta, par)
% Initialize
obj = inf; 
fit = struct('sigma2',NaN, 'beta',NaN, 'gamma',NaN, ...
    'C',NaN, 'Ft',NaN, 'G',NaN);
m = size(par.F,1);
% Set up  R
r = feval(par.corr, theta, par.D);
idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([par.ij(idx,1); o], [par.ij(idx,2); o], ...
  [r(idx); ones(m,1)+mu]);  
% Cholesky factorization with check for pos. def.
[C,rd] = chol(R);
if  rd,  return, end % not positive definite

% Get least squares solution
C = C';   Ft = C \ par.F;
[Q,G] = qr(Ft,0);
if  rcond(G) < 1e-10
  % Check   F  
  if  cond(par.F) > 1e15 
    T = sprintf('F is too ill conditioned\nPoor combination of regression model and design sites');
    error(T)
  else  % Matrix  Ft  is too ill conditioned
    return 
  end 
end
Yt = C \ par.y;   beta = G \ (Q'*Yt);
rho = Yt - Ft*beta;  sigma2 = sum(rho.^2)/m;
detR = prod( full(diag(C)) .^ (2/m) );
obj = sum(sigma2) * detR;
if  nargout > 1
  fit = struct('sigma2',sigma2, 'beta',beta, 'gamma',rho' / C, ...
    'C',C, 'Ft',Ft, 'G',G');
end

% --------------------------------------------------------

function  [t, f, fit, perf] = boxmin(t0, lo, up, par)
%BOXMIN  Minimize with positive box constraints

% Initialize
[t, f, fit, itpar] = start(t0, lo, up, par);
if  ~isinf(f)
  % Iterate
  p = length(t);
  if  p <= 2
      kmax = 2;
  else kmax = min(p,4);
  end
  for  k = 1 : kmax
    th = t;
    [t, f, fit, itpar] = explore(t, f, fit, itpar, par);
    [t, f, fit, itpar] = move(th, t, f, fit, itpar, par);
  end
end
perf = struct('nv',itpar.nv, 'perf',itpar.perf(:,1:itpar.nv));

% --------------------------------------------------------

function  [t, f, fit, itpar] = start(t0, lo, up, par)
% Get starting point and iteration parameters

% Initialize
t = t0(:);  lo = lo(:);   up = up(:);   p = length(t);
D = 2 .^((1:p)'/(p+2));
ee = find(up == lo);  % Equality constraints
if  ~isempty(ee)
  D(ee) = ones(length(ee),1);   t(ee) = up(ee); 
end
ng = find(t < lo | up < t);  % Free starting values
if  ~isempty(ng)
  t(ng) = (lo(ng) .* up(ng).^7).^(1/8);  % Starting point
end
ne = find(D ~= 1);

% Check starting point and initialize performance info
[f,fit] = objfunc(t,par);   nv = 1;
itpar = struct('D',D, 'ne',ne, 'lo',lo, 'up',up, ...
  'perf',zeros(p+2,200*p), 'nv',1);
itpar.perf(:,1) = [t; f; 1];
if  isinf(f)    % Bad parameter region
  return
end

if  length(ng) > 1  % Try to improve starting guess
  d0 = 16;  d1 = 2;   q = length(ng);
  th = t;   fh = f;   jdom = ng(1);  
  for  k = 1 : q
    j = ng(k);    fk = fh;  tk = th;
    DD = ones(p,1);  DD(ng) = repmat(1/d1,q,1);  DD(j) = 1/d0;
    alpha = min(log(lo(ng) ./ th(ng)) ./ log(DD(ng))) / 5;
    v = DD .^ alpha;   tk = th;
    for  rept = 1 : 4
      tt = tk .* v; 
      [ff, fitt] = objfunc(tt,par);  nv = nv+1;
      itpar.perf(:,nv) = [tt; ff; 1];
      if  ff <= fk 
        tk = tt;  fk = ff;
        if  ff <= f
          t = tt;  f = ff;  fit = fitt; jdom = j;
        end
      else
        itpar.perf(end,nv) = -1;   break
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

% --------------------------------------------------------

function  [t, f, fit, itpar] = explore(t, f, fit, itpar, par)
% Explore step

nv = itpar.nv;   ne = itpar.ne;
for  k = 1 : length(ne)
  j = ne(k);   tt = t;   DD = itpar.D(j);
  if  t(j) == itpar.up(j)
    atbd = 1;   tt(j) = t(j) / sqrt(DD);
  elseif  t(j) == itpar.lo(j)
    atbd = 1;  tt(j) = t(j) * sqrt(DD);
  else
    atbd = 0;  tt(j) = min(itpar.up(j), t(j)*DD);
  end
  [ff, fitt] = objfunc(tt,par);  nv = nv+1;
  itpar.perf(:,nv) = [tt; ff; 2];
  if  ff < f
    t = tt;  f = ff;  fit = fitt;
  else
    itpar.perf(end,nv) = -2;
    if  ~atbd  % try decrease
      tt(j) = max(itpar.lo(j), t(j)/DD);
      [ff,fitt] = objfunc(tt,par);  nv = nv+1;
      itpar.perf(:,nv) = [tt; ff; 2];
      if  ff < f
        t = tt;  f = ff;  fit = fitt;
      else
        itpar.perf(end,nv) = -2;
      end
    end
  end
end % k

itpar.nv = nv;

% --------------------------------------------------------

function  [t, f, fit, itpar] = move(th, t, f, fit, itpar, par)
% Pattern move

nv = itpar.nv;   ne = itpar.ne;   p = length(t);
v = t ./ th;
if  all(v == 1)
  itpar.D = itpar.D([2:p 1]).^.2;
  return
end

% Proper move
rept = 1;
while  rept
  tt = min(itpar.up, max(itpar.lo, t .* v));  
  [ff,fitt] = objfunc(tt,par); 
  nv = nv+1;
  itpar.perf(:,nv) = [tt; ff; 3];
  if  ff < f
    t = tt;  f = ff;  fit = fitt;
    v = v .^ 2;
  else
    itpar.perf(end,nv) = -3;
    rept = 0;
  end
  if  any(tt == itpar.lo | tt == itpar.up)
      rept = 0;
  end
end

itpar.nv = nv;
itpar.D = itpar.D([2:p 1]).^.25;
```

### `generateOffing.m`
```matlab
function OffDec = generateOffing(PopDec,Problem,A1)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    if size(PopDec,1) < Problem.N
        MatingPool = randi(size(PopDec,1),1,Problem.N);
        OffDec     = OperatorGA(Problem,PopDec(MatingPool,:));
    else
        OffDec = OperatorGA(Problem,PopDec);
    end
    pop_candi = [];
    NP        = size(OffDec,1);
    for ii = 1 : NP
        if min(sqrt(sum((OffDec(ii,:) - [A1.decs;pop_candi]).^2,2)))>1E-6
            pop_candi = cat(1,pop_candi,OffDec(ii,:));
        end
    end
    OffDec = pop_candi;
end
```

### `nsga3EnvironmentalSelection.m`
```matlab
function Next = nsga3EnvironmentalSelection(PopDec,PopObj,N,Z,Zmin)
% The environmental selection of NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(PopObj(Next,:),PopObj(Last,:),N-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
end

function Choose = LastSelection(PopObj1,PopObj2,K,Z,Zmin)
% Select part of the solutions in the last front

    PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
    [N,M]  = size(PopObj);
    N1     = size(PopObj1,1);
    N2     = size(PopObj2,1);
    NZ     = size(Z,1);

    %% Normalization
    % Detect the extreme points
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));
    end
    
    if size(unique(Extreme),1)~=M
        a = max(PopObj,[],1)';
    else
        Hyperplane = PopObj(Extreme,:)\ones(M,1);
        a = 1./Hyperplane;       
    end 

    % Normalization
    PopObj = PopObj./repmat(a',N,1);
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    % Associate each solution with its nearest reference point
    [d,pi]   = min(Distance',[],1);

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = hist(pi(1:N1),1:NZ);
    
    %% Environmental selection
    Choose  = false(1,N2);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);
        Jmin = find(rho(Temp)==min(rho(Temp)));
        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose==0 & pi(N1+1:end)==j);
        % Then select one solution associated with this reference point
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(d(N1+I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end
```

### `predictor.m`
```matlab
function  [y, or1, or2, dmse] = predictor(x, dmodel)
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
 
  or1 = NaN;   or2 = NaN;  dmse = NaN;  % Default return values
  if  isnan(dmodel.beta)
    y = NaN;   
    error('DMODEL has not been found')
  end

  [m n] = size(dmodel.S);  % number of design sites and number of dimensions
  sx = size(x);            % number of trial sites and their dimension
  if  min(sx) == 1 & n > 1 % Single trial point 
    nx = max(sx);
    if  nx == n 
      mx = 1;  x = x(:).';
    end
  else
    mx = sx(1);  nx = sx(2);
  end
  if  nx ~= n
    error(sprintf('Dimension of trial sites should be %d',n))
  end
  
  % Normalize trial sites  
  x = (x - repmat(dmodel.Ssc(1,:),mx,1)) ./ repmat(dmodel.Ssc(2,:),mx,1);
  q = size(dmodel.Ysc,2);  % number of response functions
  y = zeros(mx,q);         % initialize result
  
  if  mx == 1  % one site only
    dx = repmat(x,m,1) - dmodel.S;  % distances to design sites
    if  nargout > 1                 % gradient/Jacobian wanted
      [f df] = feval(dmodel.regr, x);
      [r dr] = feval(dmodel.corr, dmodel.theta, dx);
      % Scaled Jacobian
      dy = (df * dmodel.beta).' + dmodel.gamma * dr;
      % Unscaled Jacobian
      or1 = dy .* repmat(dmodel.Ysc(2, :)', 1, nx) ./ repmat(dmodel.Ssc(2,:), q, 1);
      if q == 1
        % Gradient as a column vector
        or1 = or1';
      end
      if  nargout > 2  % MSE wanted
        
        rt = dmodel.C \ r;
        u = dmodel.Ft.' * rt - f.';
        v = dmodel.G \ u;
        or2 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + sum(v.^2) - sum(rt.^2))',1,q);
        
        if  nargout > 3  % gradient/Jacobian of MSE wanted
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
    dx = zeros(mx*m,n);  kk = 1:m;
    for  k = 1 : mx
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
    
    if  nargout > 1   % MSE wanted
      rt = dmodel.C \ r;
      u = dmodel.G \ (dmodel.Ft.' * rt - f.');
      or1 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + colsum(u.^2) - colsum(rt.^2))',1,q);
      if  nargout > 2
        disp('WARNING from PREDICTOR.  Only  y  and  or1=mse  are computed')
      end
    end
    
  end % of several sites
  
% >>>>>>>>>>>>>>>>   Auxiliary function  ====================

function  s = colsum(x)
% Columnwise sum of elements in  x
if  size(x,1) == 1,  s = x; 
else,                s = sum(x);  end
```
