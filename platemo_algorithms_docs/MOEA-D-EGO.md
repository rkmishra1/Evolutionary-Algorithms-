# MOEA-D-EGO

**Tags**: <2010> <multi> <real/integer> <expensive>

## Description
MOEA/D with efficient global optimization

## Reference
Q. Zhang, W. Liu, E. Tsang, and B. Virginas. Expensive multiobjective optimization by MOEA/D with Gaussian process model. IEEE Transactions on Evolutionary Computation, 2010, 14(3): 456-474.

## Source Code

### `GPmodelFCM.m`
```matlab
function [model,centers] = GPmodelFCM(train_x,train_y,L1,L2)
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

% This function was written by Liang Zhao (liazhao5-c@my.cityu.edu.hk).

    [K,M] = size(train_y);
    D     = size(train_x,2);
    if K <= L1
        % all the K evaluated points are directly used for building GP model
        csize       = 1;
        [centers,~] = FCM(train_x,csize,[2 NaN 0.05 false]);
        model       = cell(1,M);
        theta       = (K ^ (-1 ./ K)) .* ones(1, D);
        for j = 1 : M
            model{1,j} = Dacefit(train_x,train_y(:,j),'regpoly0','corrgauss',theta,1e-6*ones(1,D),20*ones(1,D));
        end 
    else
        % FuzzyCM
        csize       = 1 + ceil((K-L1)/L2);
        [centers,~] = FCM(train_x,csize,[2 NaN 0.05 false]);
        dis         = pdist2(train_x,centers);
        [~,index]   = sort(dis);
        theta       = (L1 ^ (-1 ./ L1)) .* ones(1, D);

        %% Build GP model of each objective for each cluster 
        model = cell(csize, M);
        for i = 1 : csize
            for j = 1 :  M
                temp_index = index(1:L1,i);
                model{i,j} = Dacefit(train_x(temp_index,:),train_y(temp_index,j), 'regpoly0','corrgauss', theta, 1e-6.*ones(1,D), 20.*ones(1,D));
            end
        end
    end

end

% >>>>>>>>>>>>>>>>   Auxiliary functions  ==================== 
function [center, U, obj_fcn] = FCM(data, cluster_n, options)
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
end

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
end

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
end

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
end
```

### `MOEADEGO.m`
```matlab
classdef MOEADEGO < ALGORITHM
% <2010> <multi> <real/integer> <expensive>
% MOEA/D with efficient global optimization
% batch_size --- 5 --- Number of true function evaluations per iteration

%------------------------------- Reference --------------------------------
% Q. Zhang, W. Liu, E. Tsang, and B. Virginas. Expensive multiobjective
% optimization by MOEA/D with Gaussian process model. IEEE Transactions on
% Evolutionary Computation, 2010, 14(3): 456-474.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function was written by Liang Zhao (liazhao5-c@my.cityu.edu.hk).
% - The Java Code of MOEA/D-EGO (written by Wudong Liu) is avaliable at 
%   https://sites.google.com/view/moead/resources 
% - The Matlab Code of MOEA/D-EGO without FuzzyCM (written by Liang ZHAO) is 
%   avaliable at https://github.com/mobo-d/MOEAD-EGO

    methods
        function main(Algorithm,Problem)
           %% Parameter setting
            batch_size = Algorithm.ParameterSet(5); 
            
            %% Generate initial design using LHS or other DOE methods
            x_lhs   = lhsdesign(11*Problem.D-1, Problem.D,'criterion','maximin','iterations',1000);
            x_init  = Problem.lower +  (Problem.upper - Problem.lower).*x_lhs;  
            Archive = Problem.Evaluation(x_init);     
            % find non-dominated solutions
            FrontNo = NDSort(Archive.objs,1); 
           
            %% Optimization
            while Algorithm.NotTerminated(Archive(FrontNo==1))     
              %% Maximize ETI using MOEA/D and select q candidate points
                Batch_size = min(Problem.maxFE - Problem.FE,batch_size); % the total budget is  Problem.maxFE 
                train_x    = Archive.decs; train_y = Archive.objs;
                new_x      = Opt_ETI_FCM(Problem.M,Problem.D,Problem.lower,Problem.upper,Batch_size,train_x,train_y);              
 
               %% Expensive Evaluation
                Archive = [Archive,Problem.Evaluation(new_x)];
                FrontNo = NDSort(Archive.objs,1);
            end
        end
    end
end
```

### `Opt_ETI_FCM.m`
```matlab
function new_x = Opt_ETI_FCM(M,D,xlower,xupper,Batch_size,train_x,train_y)
% Maximizing N Subproblems and Selecting Batch of Points 
% Expected Tchebycheff Improvement (ETI)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function was written by Liang Zhao (liazhao5-c@my.cityu.edu.hk).

   %% Fuzzy clustering the solutions
   L1 = 80;
   L2 = 20;
   [GPModels,centers] = GPmodelFCM(train_x,train_y,L1,L2);

   %% Generate the initial weight vectors
    % # of weight vectors：M = 2,  3,  4,  5,  6  
    num_weights = [200,210,295,456,462]; 
    if M <= 3
        [ref_vecs, ~] = UniformPoint(num_weights(M-1),M);       % simplex-lattice design 
    elseif M <= 6
        [ref_vecs, ~] = UniformPoint(num_weights(M-1),M,'ILD'); % incremental lattice design
    else
        [ref_vecs, ~] = UniformPoint(500,M);                    % Two-layered SLD
    end
  
    %% Estimate the Utopian point z
    z    = get_estimation_z(D,xlower,xupper,GPModels,centers,ref_vecs,min(train_y,[],1)); 
    gmin = get_gmin(train_y,ref_vecs,z); 
   
    %% Using MOEA/D-DE to Maximize ETI
   [pop_ETI,candidate_x,~,~] = MOEAD_ETI(D,xlower,xupper,GPModels,centers,ref_vecs,gmin,z);

    %% Select the unsimilar candidate solutions and build candidate pool Q
    Q     = [];
    Q_ETI = []; 
    temp  = train_x;
    for i = 1 : size(candidate_x,1)
        if min(pdist2(real(candidate_x(i,:)),real(temp))) > 1e-5
            if pop_ETI(i) > 0
                Q    = [Q;candidate_x(i,:)]; Q_ETI = [Q_ETI;pop_ETI(i)];
                temp = [temp;candidate_x(i,:)];
            end
        end
    end
 
    new_x = K_means_Batch_Select(Q,Batch_size,candidate_x,Q_ETI) ;
end
 
% >>>>>>>>>>>>>>>>   Auxiliary functions  ==================== 
function gmin = get_gmin(D_objs,ref_vecs,z)
% calculate the minimum of  Tch for each ref_vec
% g(x|w,z) = max{w1(f1-z1),w2(f2-z2),...}
    Objs_translated = D_objs-z; % n*M
    G = ref_vecs(:,1)*Objs_translated(:,1)';  % N*n, f1
    for j = 2:size(ref_vecs,2)
        G = max(G,ref_vecs(:,j)*Objs_translated(:,j)'); % N*n, max(fi,fj)
    end
    gmin = min(G,[],2);  % N*1  one for each weight vector 
end 

function  [pop_ETI,pop_x,pop_mean,pop_std] = MOEAD_ETI(D,xlower,xupper,GPModels,centers,ref_vecs,gmin,z) 
%% using MOEA/D-DE to solve subproblems
    % In order to find the maximum value of ETI for each sub-problem, 
    % it is recommended to set the maximum number of iterations to at least 50.
    %% Parameter setting for MOEA/D-DE
    delta=0.9; nr = 2; maxIter = 50;
    pop_size = size(ref_vecs,1);

    %% neighbourhood   
    T       = ceil(pop_size/10); % size of neighbourhood
    B       = pdist2(ref_vecs,ref_vecs);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);

     %% the initial population for MOEA/D
    pop_x = (xupper-xlower).*lhsdesign(pop_size, D) + xlower; 
    [pop_mean,pop_std] = GPEvaluate_FCM(pop_x,GPModels,centers);
    pop_ETI = get_ETI(pop_mean,pop_std,ref_vecs,gmin,z);   

    for gen = 1 : maxIter-1
       for i = 1 : pop_size
           if rand < delta
               P = B(i,randperm(size(B,2)));
           else
               P = randperm(pop_size);
           end
           %% generate an offspring and calculate its predictive mean and s
           off_x = operator_DE(pop_x(i,:),pop_x(P(1),:),pop_x(P(2),:), xlower,xupper);          
           [off_mean,off_std]= GPEvaluate_FCM(off_x,GPModels,centers);  
            
            ETI_new = get_ETI(repmat(off_mean,length(P),1),repmat(off_std,length(P),1),ref_vecs(P,:),gmin(P),z);
            
            offindex =  find(pop_ETI(P)<ETI_new,nr) ;
            if ~isempty(offindex)
               pop_x(P(offindex),:) = repmat(off_x,length(offindex),1); 
               pop_mean(P(offindex),:) = repmat(off_mean,length(offindex),1);
               pop_std(P(offindex),:) = repmat(off_std,length(offindex),1);
               pop_ETI(P(offindex)) = ETI_new(offindex);
            end
       end      
    end
end

function  ETI = get_ETI(u,sigma,ref_vecs,Gbest,z)
%     g(x|w,z) = max{w1(f1-z1),w2(f2-z2)}  
% calculate the ETI(x|w) at multiple requests, e.g., N  
% u       : N*M  predictive mean
% sigma   : N*M  square root of the predictive variance
% ref_vecs: N*M  weight vectors 
% Gbest   : N*1  
% z       : 1*M  reference point   
    g_mu = ref_vecs.*(u - repmat(z,size(u,1),1));% N*M
    g_sig = ref_vecs.*sigma; % N*M
    % Moment Matching Approximation (MMA)
    g_sig(g_sig<0) = 0; g_sig2 = g_sig.^2; % N*M
     
     % Eq. 18 & Eq. 19 in MOEA/D-EGO
	[mma_mean,mma_sigma2] = app_max_of_2_Gaussian(g_mu(:,1:2),g_sig2(:,1:2)); % f1 & f2
    for i = 3 : size(g_mu,2)
        mu_temp = [mma_mean,g_mu(:,i)]; sig2_temp = [mma_sigma2,g_sig2(:,i)];
        [mma_mean,mma_sigma2] = app_max_of_2_Gaussian(mu_temp,sig2_temp);
    end
    
    mma_std = (sqrt(mma_sigma2));
    Gbest_minus_u = Gbest-mma_mean;
    tau = Gbest_minus_u./mma_std; % n*1

    % Precompute the normal distributions
    normcdf_tau = normcdf(tau);
    normpdf_tau = normpdf(tau);

    ETI = Gbest_minus_u.*normcdf_tau + mma_std.*normpdf_tau;
end

function [u,s] = GPEvaluate_FCM(X,model,centers)
% Predict the objective vector of the candidate solutions accodring to the
% Euclidean distance from each candidate solution to evaluated solutions
    D = pdist2(X,centers);
    [~,index] = min(D,[],2);
    N = size(X,1); % number of samples
    M = size(model,2);% number of objectives
    u = zeros(N,M); % predictive mean
    MSE = zeros(N,M); % predictive MSE
    for i = 1 : N
        for j = 1 : M
            [u(i,j),~,MSE(i,j)] = Predictor(X(i,:),model{index(i),j}); % DACE Kriging toolbox
        end  
    end
     MSE(MSE<0) = 0;
     s = sqrt(MSE);% square root of the predictive variance
end

function [u] = GPEvaluate_mean_FCM(X,model,centers)
% Predict the objective vector of the candidate solutions accodring to the
% Euclidean distance from each candidate solution to evaluated solutions
    D = pdist2(X,centers);
    [~,index] = min(D,[],2);
    N = size(X,1); % number of samples
    M = size(model,2); % number of objectives
    u = zeros(N,M); % predictive mean
    for i = 1 : N
        for j = 1 : M
            [u(i,j)] = Predictor(X(i,:),model{index(i),j}); % DACE Kriging toolbox
        end
    end
end

function [y,s2] = app_max_of_2_Gaussian(mu,sig2)
% Calculate  Eq. 18 & Eq. 19 in MOEA/D-EGO 
% n requests
% mu is N*2
% sig2 is N*2
    tao = sqrt(sum(sig2,2));  % N*1
    alpha = (mu(:,1)-mu(:,2))./tao;  % N*1
    % Eq. 16 / Eq. 18
    y = mu(:,1).*normcdf(alpha) + mu(:,2).*normcdf(-alpha) + tao.*normpdf(alpha);  % N*1
    % There is a typo in Eq. 17.  See Appendix B of MOEA/D-EGO.
    % It should be $$ +(\mu_1+\mu_2) \tau \varphi(\alpha)$$
    s2 = (mu(:,1).^2 + sig2(:,1)).*normcdf(alpha) + ...
        (mu(:,2).^2 + sig2(:,2)).*normcdf(-alpha) + (sum(mu,2)).*tao.*normpdf(alpha); 
%     s2 = (mu(:,1).^2 + sig2(:,1)).*normcdf(alpha) + ...
%         (mu(:,2).^2 + sig2(:,2)).*normcdf(-alpha) + (sum(mu,2)).*normpdf(alpha); 
    s2 = s2 - y.^2;
    s2(s2<0) = 0;
end

function  z = get_estimation_z(D, xlower,xupper,GPModels,centers,ref_vecs,z) 
% min(\mu_1(x),...,\mu_m(x))^T
% Utilize MOEA/D to minimize the GP posterior mean and determine the utopian
% point during optimization. Alternatively, other multi-objective optimization 
% algorithms such as NSGA-II can also be employed.

    delta=0.9; nr = 2; 
    maxIter = 100; 
    pop_size = size(ref_vecs,1);

    %% neighbourhood   
    T       = ceil(pop_size/10); % size of neighbourhood
    B       = pdist2(ref_vecs,ref_vecs);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);

    %% the initial population for MOEA/D
    pop_x = (xupper-xlower).*lhsdesign(pop_size, D) + xlower;
    pop_mean = GPEvaluate_mean_FCM(pop_x,GPModels,centers);
    z       = min(min(pop_mean,[],1),z);
    for gen = 1 : maxIter-1
       for i = 1 : pop_size
           if rand < delta
               P = B(i,randperm(size(B,2)));
           else
               P = randperm(pop_size);
           end
           %% generate an offspring and calculate its predictive mean and s
           off_x = operator_DE(pop_x(i,:),pop_x(P(1),:),pop_x(P(2),:), xlower,xupper);    
           [off_mean]= GPEvaluate_mean_FCM(off_x,GPModels,centers);  
           
            z = min(z,off_mean);        
            g_old = max((pop_mean(P,:) - repmat(z,length(P),1)).*ref_vecs(P,:),[],2);
            g_new = max(repmat((off_mean-z),length(P),1).*ref_vecs(P,:),[],2);
             
           % Update the solutions in P
           offindex = P(find(g_old>g_new,nr));
           if ~isempty(offindex)
               pop_x(offindex,:) = repmat(off_x,length(offindex),1); 
               pop_mean(offindex,:) = repmat(off_mean,length(offindex),1);
           end
       end      
    end
end

function SelectDecs = K_means_Batch_Select(Q,Batch_size,candidate_x,Q_ETI) 
     batch_size = min(Batch_size,size(Q,1));% in case Q is smaller than Batch size
    
    if batch_size == 0
        Qb = randperm(size(candidate_x,1),Batch_size);
        SelectDecs = candidate_x(Qb,:);
    else
        cindex  = kmeans(Q,batch_size);
        Qb = [];
        for i = 1 : batch_size
            index = find(cindex == i); 
            [~,best] = max(Q_ETI(index));
            Qb = [Qb,index(best)];
        end
        SelectDecs = Q(Qb,:);
    end
    % when Q is smaller than batch size
    if size(SelectDecs,1) < Batch_size
        Qb = randperm(size(candidate_x,1), Batch_size - size(SelectDecs,1));
        SelectDecs = [SelectDecs;candidate_x(Qb,:)];
    end
end

% >>>>>>>>>>>>>>>>    functions in PlatEMO ====================
function Offspring = operator_DE(Parent1,Parent2,Parent3, xlower,xupper)
%OperatorDE - The operator of differential evolution.

    %% Parameter setting
    [CR,F,proM,disM] = deal(1,0.5,1,20);
    [N,D] = size(Parent1);

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    Lower = repmat(xlower,N,1);
    Upper = repmat(xupper,N,1);
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
