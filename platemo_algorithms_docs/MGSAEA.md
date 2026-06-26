# MGSAEA

**Tags**: <2024> <multi> <real> <expensive> <constrained>

## Description
Multigranularity surrogate-assisted constrained evolutionary algorithm

## Reference
Y. Zhang, H. Jiang, Y. Tian, H. Ma, and X. Zhang. Multigranularity surrogate modeling for evolutionary multiobjective optimization with expensive constraints. IEEE Transactions on Neural Networks and Learning Systems, 2024, 35(3): 2956-2968.

## Source Code

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EnvironmentalSelection.m`
```matlab
function [PopDec,PopObj,Fitness] = EnvironmentalSelection(PopDec,PopObj,NI,M,status)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete the duplicated points
    [~, Unduplicated] = unique(PopObj(:,1:M),'rows');
    PopDec            = PopDec(Unduplicated,:);
    PopObj            = PopObj(Unduplicated,:);
    
    %% Calculate the fitness of each solution
    if nargin == 4
        Fitness = CalFitness(PopObj);
    else
        if status == 1
            RealObj = PopObj(:,1:M);
            CV      = PopObj(:,end);           
            Fitness = CalFitness(RealObj,max(0,CV)); 
        elseif status == 2
            RealObj = PopObj(:,1:M);
            PopCon  = PopObj(:,M+1:end);
            CV      = sum(max(0,PopCon),2);            
            Fitness = CalFitness([RealObj,CV]);
        elseif status == 3
            Fitness = CalFitness(PopObj);
        end
    end
    
    %% Environmental selection
    if nargin == 4
        Next = Fitness < 1;
        if sum(Next) < NI
            [~,Rank] = sort(Fitness);
            Next(Rank(1:NI)) = true;
        elseif sum(Next) > NI
            Del  = Truncation(PopObj(Next,:),sum(Next)-NI);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
    else
        Next = Fitness < 1;
        if sum(Next) < NI
            [~,Rank] = sort(Fitness);
            Next(Rank(1:NI)) = true;
        elseif sum(Next) > NI
            if status ~=3
                RealObj = PopObj(:,1:M);
            else
                RealObj = PopObj;
            end
            Del  = Truncation(RealObj(Next,:),sum(Next)-NI);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
    end
    PopDec  = PopDec(Next,:);
    PopObj  = PopObj(Next,:);   
    Fitness = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `MGSAEA.m`
```matlab
classdef MGSAEA < ALGORITHM
% <2024> <multi> <real> <expensive> <constrained>
% Multigranularity surrogate-assisted constrained evolutionary algorithm
% wmax   ---    20 --- Number of generations before updating surrogate models
% mu     ---     5 --- Number of real evaluated solutions at each iteration
% gap    ---    20 --- Parameter calculating the change rate of ideal points
% lambda --- 0.001 --- Parameter determining the evolving stages 

%------------------------------- Reference --------------------------------
% Y. Zhang, H. Jiang, Y. Tian, H. Ma, and X. Zhang. Multigranularity
% surrogate modeling for evolutionary multiobjective optimization with
% expensive constraints. IEEE Transactions on Neural Networks and Learning
% Systems, 2024, 35(3): 2956-2968.
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
            [wmax,mu,gap,lambda] = Algorithm.ParameterSet(20,5,20,1e-3);
            
            %% Generate the initial population 
            NI          = 11*Problem.D-1;
            P           = UniformPoint(NI,Problem.D,'Latin');
            Population  = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));                      
            flag        = 0;  % 0 for the first stage, 1 for the second stage
            Iter        = 1;
            IdealPoints = [];
            Archive     = UpdateArchive(Population,Problem.N);
            THETA_OBJ   = 5.*ones(Problem.M,Problem.D);
            THETA_CV    = 5.*ones(1,Problem.D);
            THETA_CON   = 5.*ones(size(Population.cons,2),Problem.D);
            
            %% Optimization
            while Algorithm.NotTerminated(Archive)
                IdealPoints(Iter,:) = min(Population.objs,[],1);
                if Iter > gap && flag == 0
                    max_change = calc_maxchange(IdealPoints,Iter,gap);
                    if max_change <= lambda
                        flag = 1;
                    end
                end
                if flag ==0 
                    PopDec  = Population.decs;
                    PopObj  = Population.objs;
                    M       = Problem.M;
                    Model   = cell(1,M);
                    % Construct M surrogate models for M objectives
                    for i = 1 : M
                        dmodel     = dacefit(PopDec,PopObj(:,i),'regpoly0','corrgauss',THETA_OBJ(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                        Model{i}   = dmodel;
                        THETA_OBJ(i,:) = dmodel.theta;
                    end
                    % Use the surrogates
                    Fitness = CalFitness(PopObj);
                    for w = 1:wmax
                        drawnow();
                        MatingPool = TournamentSelection(2,NI,Fitness);
                        OffDec     = OperatorGA(Problem,PopDec(MatingPool,:));
                        PopDec     = cat(1,PopDec,OffDec);
                        [N,~]      = size(PopDec);
                        PopObj     = zeros(N,M);
                        MSE        = zeros(N,M);
                        for i = 1: N
                            for j = 1 : M
                                [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Model{j});
                            end
                        end
                        [PopDec,PopObj,Fitness] = EnvironmentalSelection(PopDec,PopObj,NI,Problem.M);
                    end
                    % Select mu solutions for real evaluation in the first stage
                    [NewDec,~,~] = EnvironmentalSelection(PopDec,PopObj,mu,Problem.M);
                    New          = Problem.Evaluation(NewDec);
                    % Update Population and Archive with mu new solutions
                    Population   = UpdatePopulation(Population,New,NI-mu);
                    Archive      = cat(2,Archive,New);
                    Archive      = UpdateArchive(Archive,Problem.N);
                else  
                    MaxCon = max(max(0,Population.cons));
                    Ninf   = length(find(MaxCon>0)); % The number of not fully staisfied constraints
                    if Ninf == size(Population.cons,2)
                        status  = 1;
                        CV      = NormalizeCV(Population.cons); 
                        PopDec  = Population.decs;
                        PopObj  = [Population.objs,CV];
                        M       = Problem.M+1;
                        Model   = cell(1,M);
                        Fitness = CalFitness(Population.objs,Population.cons);
                        THETA   = [THETA_OBJ;THETA_CV];
                        % Construct M+1 surrogate models for M objectives and the CV function
                        for i = 1 : M
                            dmodel     = dacefit(PopDec,PopObj(:,i),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                            Model{i}   = dmodel;
                            THETA(i,:) = dmodel.theta;
                        end
                        THETA_OBJ = THETA(1:Problem.M,:);
                        THETA_CV  = THETA(end,:);
                    elseif Ninf > 0 && Ninf < size(Population.cons,2)
                        status   = 2;
                        PopCon   = max(0,Population.cons);
                        MaxCon   = max(PopCon);
                        index    = find(MaxCon>0);
                        PopCon(:,MaxCon==0) = [];
                        PopCon   = (PopCon-min(PopCon,[],1))./(max(PopCon,[],1)-min(PopCon,[],1));
                        PopDec   = Population.decs;
                        PopObj   = [Population.objs,PopCon];
                        M        = size(PopObj,2);
                        Model    = cell(1,M);
                        Fitness  = CalFitness([Population.objs,sum(max(0,Population.cons),2)]);
                        THETA    = [THETA_OBJ;THETA_CON(index,:)];
                        % Construct surrogate models for M objectives and the not
                        % fully satisfied constraints
                        for i = 1 : M
                            dmodel     = dacefit(PopDec,PopObj(:,i),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                            Model{i}   = dmodel;
                            THETA(i,:) = dmodel.theta;
                        end
                        THETA_OBJ = THETA(1:Problem.M,:);
                        THETA_CON(index,:) = THETA(Problem.M+1:end,:);
                    elseif Ninf  == 0
                        status   = 3;
                        PopDec   = Population.decs;
                        PopObj   = Population.objs;
                        M        = Problem.M;
                        Model    = cell(1,M);
                        Fitness  = CalFitness(PopObj);
                        % Construct M surrogate models for M objectives
                        for i = 1 : M
                            dmodel     = dacefit(PopDec,PopObj(:,i),'regpoly0','corrgauss',THETA_OBJ(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                            Model{i}   = dmodel;
                            THETA_OBJ(i,:) = dmodel.theta;
                        end
                    end
                    % Use the surrogates
                    for w = 1: wmax
                        drawnow();
                        MatingPool = TournamentSelection(2,NI,Fitness);
                        OffDec     = OperatorGA(Problem,PopDec(MatingPool,:));
                        PopDec     = cat(1,PopDec,OffDec);
                        [N,~]      = size(PopDec);
                        PopObj     = zeros(N,M);
                        MSE        = zeros(N,M);
                        for i = 1: N
                            for j = 1 : M
                                [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Model{j});
                            end
                        end
                        [PopDec,PopObj,Fitness] = EnvironmentalSelection(PopDec,PopObj,NI,Problem.M,status);
                    end
                    % Select mu solutions for real evaluation in the second stage
                    [NewDec,~,~] = EnvironmentalSelection(PopDec,PopObj,mu,Problem.M,status);
                    New          = Problem.Evaluation(NewDec);
                    % Update Population and Archive with mu new solutions
                    Population   = UpdatePopulation(Population,New,NI-mu,status);
                    Archive      = cat(2,Archive,New);
                    Archive      = UpdateArchive(Archive,Problem.N);
                end
                Iter = Iter +1;
            end
        end
    end
end

function max_change = calc_maxchange(ideal_points,Iter,gap)
    % Calculate the maximum change rate of ideal points
    delta = 1e-6 * ones(1,size(ideal_points,2));
    rz    = abs((ideal_points(Iter,:) - ideal_points(Iter - gap,:)) ./ max(ideal_points(Iter - gap,:),delta));  
    max_change = max(rz);
end

function CV = NormalizeCV(PopCon)
    % Calculate the normalized overall constraints violation
    PopCon = max(0,PopCon);
    PopCon = (PopCon-min(PopCon,[],1))./(max(PopCon,[],1)-min(PopCon,[],1));
    PopCon(:,isnan(PopCon(1,:))) = 0;
    CV = sum(PopCon,2);
end
```

### `UpdateArchive.m`
```matlab
function Archive = UpdateArchive(Archive,N)
% Update the archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete the duplicated solutions
    [~, Unduplicated] = unique(Archive.objs,'rows');
    Archive           = Archive(Unduplicated);
    if length(Archive) > N
        % Calculate the fitness of each solution
        Fitness = CalFitness(Archive.objs, Archive.cons);          
        % Environmental selection
        Next = Fitness < 1;
        if sum(Next) < N
            [~,Rank] = sort(Fitness);
            Next(Rank(1:N)) = true;
        elseif sum(Next) > N
            Del  = Truncation(Archive(Next).objs,sum(Next)-N);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
        Archive = Archive(Next);
    end
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `UpdatePopulation.m`
```matlab
function Population = UpdatePopulation(Population,New,N,status)
% Select NI-mu solutions from Population and combine them with mu new solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete the solutions in Population that may duplicate with new solutions
    [~,index]  = setdiff(Population.objs,New.objs,'rows');
    Population = Population(index);
    
    %% Delete the duplicated solutions in current Population
    [~, Unduplicated] = unique(Population.objs,'rows');
    Population        = Population(Unduplicated);

    %% Calculate the fitness of each solution  
    if nargin == 3
        Fitness = CalFitness(Population.objs);
    else
        if status ==1
            Fitness = CalFitness(Population.objs,Population.cons);
        elseif status ==2
            CV      = sum(max(0,Population.cons),2);         
            Fitness = CalFitness([Population.objs,CV]);
        elseif status ==3
            Fitness = CalFitness(Population.objs,Population.cons);
        end
    end
    
    %% Environmental selection 
    if length(Population)>N
        Next = Fitness < 1;
        if sum(Next) < N
            [~,Rank] = sort(Fitness);
            Next(Rank(1:N)) = true;
        elseif sum(Next) > N
            Del  = Truncation(Population(Next).objs,sum(Next)-N);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
        Population = Population(Next);
    end
    Population = cat(2,Population,New);  
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
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
