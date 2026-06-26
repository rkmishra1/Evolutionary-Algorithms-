# NSGAIII-EHVI

**Tags**: <2023> <multi/many> <real> <expensive>

## Description
NSGA-III with expected hypervolume improvement

## Reference
Y. Pang, Y. Wang, S. Zhang, X. Lai, W. Sun, and X. Song. An expensive many-objective optimization algorithm based on efficient expected hypervolume improvement. IEEE Transactions on Evolutionary Computation, 2023, 27(6): 1822-1836.

## Source Code

### `CalEHVI.m`
```matlab
function EI = CalEHVI(RealFirstObj,Obj,MSE,S,S_S)
% EHVI calculation using improtance sampling

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    nSample    = size(S,1);
    NRealFront = size(RealFirstObj,1);
    
    PopObj   = [RealFirstObj;Obj];
    [NPop,M] = size(Obj);
    Zmin     = min(PopObj,[],1);
    Zmax     = max(PopObj,[],1);
    
    % Normalization
    a   = Zmax - Zmin;
    Obj = Obj - repmat(Zmin,size(Obj,1),1);
    RealFirstObj = RealFirstObj - repmat(Zmin,size(RealFirstObj,1),1);
    Obj = Obj./repmat(a,NPop,1);
    RealFirstObj = RealFirstObj ./repmat(a,NRealFront,1);
    
    PopMSE = MSE./repmat((a.^2),size(MSE,1),1);
    sigma  = PopMSE.^0.5;
    R_S    = zeros(NRealFront,nSample);
    for i = 1 : NRealFront
        x        = sum(repmat(RealFirstObj(i,:),nSample,1)-S<=0,2) == M;
        R_S(i,x) = 1;
    end
    index     = (sum(R_S,1) == 0);
    NonDomS   = S(index,:);
    nNonDomS  = size(NonDomS,1);
    NonDomS_S = S_S(index,index);

    % Measurement of improvement
    I  = sum(NonDomS_S,2);
    EI = ones(NPop,1);
    for i = 1 : NPop
        if any(sigma(i,:)==0)
            EI(i) = 0;
        else
            P = ones(nNonDomS,1);
            for j = 1 : M
                p = normpdf(NonDomS(:,j) ,Obj(i,j),sigma(i,j));
                P = P.*p;
            end
            EI(i) = sum(I.*P);
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,Z,Zmin)
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
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,N-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
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
    [d,pi] = min(Distance',[],1);

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

### `LastSelection.m`
```matlab
function Choose = LastSelection(PopObj1,PopObj2,K,Z,Zmin)
% Select part of the solutions in the last front

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

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
    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes
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
    [d,pi] = min(Distance',[],1);

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

### `NSGAIIIEHVI.m`
```matlab
classdef NSGAIIIEHVI < ALGORITHM
% <2023> <multi/many> <real> <expensive>
% NSGA-III with expected hypervolume improvement
% wmax    ---    15 --- Number of generations before updating Kriging models
% LB      ---  -0.5 --- Low bound for EHVI calculation
% UB      ---   1.2 --- Upper bound for EHVI calculation
% nSample --- 10000 --- number of samples in importance sampling for EHVI
% randp   ---   0.3 --- The parameter in generating a random number

%------------------------------- Reference --------------------------------
% Y. Pang, Y. Wang, S. Zhang, X. Lai, W. Sun, and X. Song. An expensive 
% many-objective optimization algorithm based on efficient expected 
% hypervolume improvement. IEEE Transactions on Evolutionary 
% Computation, 2023, 27(6): 1822-1836.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yong Pang (email: pangy@mail.dlut.edu.cn)

    methods
        function main(Algorithm,Problem)
            % Parameter setting
            [wmax,LB,UB,nSample, randp] = Algorithm.ParameterSet(15,-0.5,1.2,10000,0.3);
            
            % Initialization of NSGAIII
            NI            = Problem.N;
            P             = UniformPoint(NI,Problem.D,'Latin');
            Population    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Zmin          = min(Population.objs,[],1); 
            A             = Population;
            theta         = 10.*ones(Problem.M,Problem.D);
            Model         = cell(1,Problem.M);

            % Sample and calculate hypervolume improvement
            Lowb = LB .*ones(1,Problem.M);
            Upb  = UB .*ones(1,Problem.M);
            S    = UniformPoint(nSample,Problem.M,'Latin');
            S    = S.*repmat(Upb-Lowb,nSample,1)+repmat(Lowb,nSample,1);
            S_S  = zeros(nSample,nSample);
            for i = 1 : nSample
                y        = sum(repmat(S(i,:),nSample,1)-S<=0,2) == Problem.M;  
                S_S(i,y) = 1 ;
            end 
            
            % Main loop           
            while Algorithm.NotTerminated(A)
                Dec = Population.decs;
                Obj = Population.objs;
                MSE = zeros(size(Dec,1),Problem.M);
                
                % Train kriging models
                train_X = A.decs;
                train_Y = A.objs;
                [~,distinct] = unique(round(train_X*1e6)/1e6,'rows');  
                train_X      = train_X(distinct,:);
                train_Y      = train_Y(distinct,:);
                for i = 1 : Problem.M 
                    dmodel     = dacefit(train_X,train_Y(:,i),'regpoly0','corrgauss',theta(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    Model{i}   = dmodel;
                    theta(i,:) = dmodel.theta;
                end
                
                % Obtain current real PF
                [RealFrontNo,~] = NDSort(train_Y,1);
                RealFirstObj    = train_Y((RealFrontNo==1),:);        
                w = 1;
                while w <= wmax
                    w = w + 1;
                    OffspringDec = OperatorGA(Problem,Dec(randi(end,1,NI),:));
                    N = size(OffspringDec,1);
                    OffspringObj = zeros(N,Problem.M);
                    OffspringMSE = zeros(N,Problem.M);
                    for i = 1 : N
                        for j = 1 : Problem.M
                            [OffspringObj(i,j),~,OffspringMSE(i,j)] = predictor(OffspringDec(i,:),Model{j});
                        end
                    end
                    Zmin    = min([Zmin; OffspringObj],[],1);
                    all_Obj = [Obj;OffspringObj];
                    all_MSE = [MSE;OffspringMSE];
                    all_Dec = [Dec;OffspringDec];
                    
                    % Nodominated sorting considering the uncertainty           
                    Choose = SelectionMSE(all_Obj,all_MSE,RealFirstObj,NI,Z,Zmin);                                        
                    Dec    = all_Dec(Choose,:);
                    Obj    = all_Obj(Choose,:);
                    MSE    = all_MSE(Choose,:);
                end
     
                % Efficent EHVI calcualtion using importance sampling       
                EHVI = CalEHVI(RealFirstObj,Obj,MSE,S,S_S);
                [~,sortIndex] = sort(EHVI,'descend');

                % Diversity maintain 
                ChoseMax  = max([round(size(EHVI,1)*unifrnd(0,randp)),1]);
                FnewIndex = sortIndex(1:ChoseMax);   
                   
                % Calcualte distance
                DA     = RealFirstObj;
                dist_D = zeros(size(FnewIndex,1),size(DA,1));
                for i = 1 : size(FnewIndex,1 )
                    for j = 1 : size(DA,1)
                        dist_D(i,j) = norm(Obj(FnewIndex(i,1),:)-DA(j,:),2);
                    end
                end
                
                % Diversity Indicator
                DI = min(dist_D,[],2);  
                [ ~,SnewIndex] = max(DI);
                newIndex = [FnewIndex(SnewIndex,1)];
                PnewDec  = Dec(newIndex,:);
                PnewDec  = unique(PnewDec,'rows');               
                New  = Problem.Evaluation(PnewDec);
                A    = [A,New];
                A2   = [Population,New];
                Zmin = min(A2.objs,[],1);
                Population = EnvironmentalSelection([Population,New],Problem.N,Z,Zmin);  
            end      
        end
    end
end
```

### `SelectionMSE.m`
```matlab
function Choose = SelectionMSE(Obj,MSE,RealFirstObj,N,Z,Zmin)
% Nodominated sorting considering the uncertainty of the points, the
% sorting rule is K1,L1,K2,L2....

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    MSE2 = prod(MSE,2);
    MSE3 = MSE2.^(1/size(MSE,2));
    all  = [Obj,MSE3];
    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    [NRealFront,M] = size(RealFirstObj);
    NPop = size(Obj,1);
    
    R_S = zeros(NRealFront,NPop);      
    for i = 1 : NRealFront
        x        = sum(repmat(RealFirstObj(i,:),NPop,1)-Obj<=0,2) == M;  
        R_S(i,x) = 1;  
    end      
    
    index=(sum(R_S,1)==0);
    NonDomPopAll = all(index,:);
    OtherPopAll  = all(index==0,:);
   
    [FrontNo1,~] = NDSort(NonDomPopAll,N);
    OtherPopAll(:,M+1:end) = -1.*OtherPopAll(:,M+1:end);
    [FrontNo2,~] = NDSort(OtherPopAll,N);
    nonDom    = find(index==1);
    other     = find(index==0);
    Choose    = false(1,size(index,2));
    NonDomPop = Obj(index,:);
    OtherPop  = Obj(index==0,:);
    tnum      = 0;
    i = 0;
    while tnum < N
        i = i + 1;
        current1 = FrontNo1 == i;
        if tnum+sum(current1) <= N
            Choose(nonDom(current1)) = true ;
            tnum = tnum+sum(current1);
        else
            temp1 = false(1,size(FrontNo1,2));
            Last1 = find(FrontNo1==i);
            C1    = LastSelection(Obj(Choose,:),NonDomPop(current1,:),N-sum(Choose),Z,Zmin);
            temp1(Last1(C1)) = true;
            Choose(nonDom(temp1)) = true ;           
            tnum = tnum + sum(C1);
        end
        if tnum == N
            break;
        elseif tnum > N
            fprintf('error')
        else
            current2 = FrontNo2 == i;
            if tnum+sum(current2) <= N
                Choose(other(current2)) = true ;
                tnum = tnum + sum(current2);
            else
                temp2 = false(1,size(FrontNo2,2));
                Last2 = find(FrontNo2==i);
                C2    = LastSelection(Obj(Choose,:),OtherPop(current2,:),N-sum(Choose),Z,Zmin);
                temp2(Last2(C2))     = true;
                Choose(other(temp2)) = true ;           
                tnum = tnum+sum(C2);
            end    
        end
    end    
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
    [d,pi] = min(Distance',[],1);

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
