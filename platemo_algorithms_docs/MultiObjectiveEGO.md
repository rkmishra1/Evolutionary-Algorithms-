# MultiObjectiveEGO

**Tags**: <2016> <multi> <real/integer> <constrained/none> <expensive>

## Description
Multi-objective efficient global optimization

## Reference
R. Hussein and K. Deb. A generative Kriging surrogate model for constrained and unconstrained multi-objective optimization. Proceedings of the Genetic and Evolutionary Computation Conference, 2016, 573-580.

## Source Code

### `Infill_Maximal_Distance.m`
```matlab
function obj = Infill_Maximal_Distance(x, sample_x)
obj = - min(pdist2(x, sample_x),[],2);
end
```

### `Infill_Standard_EI.m`
```matlab
function obj=Infill_Standard_EI(x, Kriging_model, f_min)
%--------------------------------------------------------------------------
% the DACE toolbox of  Lophaven et al. (2002)  is used to predict value
%--------------------------------------------------------------------------
% REFERENCES
%Lophaven SN, Nielsen HB, and Sodergaard J, DACE - A MATLAB Kriging
%Toolbox, Technical Report IMM-TR-2002-12, Informatics and Mathematical
%Modelling, Technical University of Denmark, 2002.
%Available at: http://www2.imm.dtu.dk/~hbn/dace/.
%--------------------------------------------------------------------------
% get the Kriging prediction and variance
[y,mse] = predictor(x,Kriging_model);
s=sqrt(max(0,mse));
% calcuate the EI value
EI=(f_min-y).*Gaussian_CDF((f_min-y)./s)+s.*Gaussian_PDF((f_min-y)./s);

% the genetic algorithm tries to minimize the objective
obj=-EI;

end

function res=Gaussian_PDF(x)
res=1/sqrt(2*pi)*exp(-x.^2/2);
end

function x=Gaussian_CDF(x)
  x=0.5*(1+erf(x/sqrt(2)));
end
```

### `MultiObjectiveEGO.m`
```matlab
classdef MultiObjectiveEGO < ALGORITHM
% <2016> <multi> <real/integer> <constrained/none> <expensive>
% Multi-objective efficient global optimization
% alpha --- 0.7 --- portion of samples for Kriging construction
% num_k ---   5 --- number of infill points per iteration
% H     ---  21 --- number of reference directions

%------------------------------- Reference --------------------------------
% R. Hussein and K. Deb. A generative Kriging surrogate model for
% constrained and unconstrained multi-objective optimization. Proceedings
% of the Genetic and Evolutionary Computation Conference, 2016, 573-580.
%--------------------------------------------------------------------------

% This function is written by Youwei He (email: 1554748356@qq.com)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha,num_k,H] = Algorithm.ParameterSet(0.7,5,21);
            % parameter for AASF in equation (10)
            rho = 1e-3;

            %% Generate the initial design points
            % number of design variables
            D = Problem.D;
            % number of initial design points 11*D-1
            N = Problem.N;
            % step1-1: generate initial design points using Latin Hypercube sampling
            PopDec = repmat(Problem.upper-Problem.lower,N,1).*UniformPoint(N,D,'Latin') ...
                     +repmat(Problem.lower,N,1);
            % step1-2: evaluate initial design points
            Population = Problem.Evaluation(PopDec);
            % step 2: Generate the reference direction set
            [R,N_R] = UniformPoint(H,Problem.M);
            R       = R./sqrt(sum(R.^2,2));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% step 2 diversity_preserver procedure: neighborhood approach
                i_direction = linspace(1,N_R,N_R);
                for i = i_direction
                    for j = 1 : num_k
                        Algorithm.NotTerminated(Population);
                        PopDecT = Population.decs;
                        PopObjT = Population.objs;
                        PopConT = Population.cons;% cons>=0: meet constraints
                        num_con = size(PopConT,2);
                        N_Pop   = length(PopDecT(:,1));
                        % determine whether the problem is constrained or not
                        if all(all(PopConT==0)) && N_Pop==1
                            constrained=1;
                        else
                            constrained=0;
                        end
                        %% step 3: Points_Selector procedure
                        normW   = sqrt(sum(R(i,:).^2,2));
                        normP   = sqrt(sum(PopObjT.^2,2));
                        CosineP = sum((PopObjT).*R(i,:),2)./normP./normW;
                        % orthogonal distance of each point to the given reference direction
                        distB    = normP.*sqrt(1-CosineP.^2);
                        [~,indx] = sort(distB);
                        N        = ceil(alpha*N_Pop);
                        PopObj   = PopObjT(indx(1:N),:);
                        PopDec   = PopDecT(indx(1:N),:);
                        PopCon   = PopConT(indx(1:N),:);
                        %% step 4-1
                        index = sum(PopCon >= 0, 2) == num_con;% feasible solutions
                        PopCon(index,:)  = 0;
                        PopCon(~index,:) = -PopCon(~index,:);     
                        PopObjScaled = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj),N,1)-repmat(min(PopObj),N,1));  
                        if constrained
                            PopConScaled = (PopCon-repmat(min(PopCon),N,1))./(repmat(max(PopCon),N,1)-repmat(min(PopCon),N,1));
                        end
                        %% step 4-2
                        PopSmetric  = zeros(N,1);
                        PopSmetricT = PopObjScaled(index,:)./repmat(R(i,:),sum(index),1);
                        PopSmetric(index,:) = max(PopSmetricT,[],2)+rho*sum(PopSmetricT,2);
                        if constrained
                            ASF_max = max(PopSmetric(index,1));
                            PopSmetric(~index,:) = ASF_max.*ones(sum(~index),1) + sum(PopConScaled(~index,:),2);
                        end
                        %% step 4-3
                        kriging_obj = dacefit(PopDec,PopSmetric,'regpoly0','corrgauss',1*ones(1,D),0.001*ones(1,D),1000*ones(1,D));
                        f_min       = min(PopSmetric);
                        %% step 5: Optimization
                        infill_criterion = @(x)Infill_Standard_EI(x, kriging_obj, f_min);
                        best_x           = rGA(infill_criterion,Problem);
                        % infill point too close (not used in the paper, 
                        % but this happens sometimes especially for low dimensional problem)
                        if min(sqrt(sum((PopDecT-best_x).^2,2)))<1E-8
                            best_x = rGA(@(x)Infill_Maximal_Distance(x, PopDecT),Problem);
                        end
                        Population = [Population,Problem.Evaluation(best_x)];
                    end   
                end  
            end
        end
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

### `rGA.m`
```matlab
function PopDec=rGA(fhandle,Problem)
% real-parameter genetic algorithm to find minimal objective value

%--------------------------------------------------------------------------
% This function is written by Youwei He (email: 1554748356@qq.com)

    % the number of design variables
    D = Problem.D;
    % population size of GA
    GAPopulationSize = 10*D;
    % Generation of GA
    GAGeneration = 100;
    obj_max      = Inf;
    % the first GA generation, randomly generated
    Offspring = repmat(Problem.upper-Problem.lower,GAPopulationSize,1).*...
                UniformPoint(GAPopulationSize,D,'Latin')+repmat(Problem.lower,GAPopulationSize,1);
    % the GA process for optimizing the objective function
    for gen = 1 :  GAGeneration
        obj_Offspring = feval(fhandle, Offspring);
        [~,index] = sort(obj_Offspring,'ascend');
        if obj_Offspring(index(1)) < obj_max
            Best    = Offspring(index(1),:);
            obj_max = obj_Offspring(index(1));
        end
        Parent    = Offspring(index(1:ceil(GAPopulationSize/2)),:);
        Offspring = [OperatorGA(Problem,Parent(TournamentSelection(2,size(Parent,1), ...
            obj_Offspring(index(1:ceil(GAPopulationSize/2)))),:));OperatorGA(Problem,Parent,{0.9,2,1/D,20})];
    end
    PopDec = Best;
end
```
