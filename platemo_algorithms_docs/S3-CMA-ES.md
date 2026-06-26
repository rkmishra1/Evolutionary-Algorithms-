# S3-CMA-ES

**Tags**: <2020> <multi/many> <real/integer> <large/none>

## Description
Scalable small subpopulations based covariance matrix adaptation

## Reference
H. Chen, R. Cheng, J. Wen, H. Li, and J. Weng. Solving large-scale many-objective optimization problems by covariance matrix adaptation evolution strategy with scalable small subpopulations. Information Sciences, 2020, 509: 457-469.

## Source Code

### `ControlVariableAnalysis.m`
```matlab
function [PV,DV] = ControlVariableAnalysis(Problem,NCA)
% Control variable analysis

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen
    
    Fno = zeros(1,Problem.D);
    for i = 1 : Problem.D
        x          = 0.2*ones(1,Problem.D).*(Problem.upper-Problem.lower) + Problem.lower;
        S          = repmat(x,NCA,1);
        inter      = (0.95-0.05)/(NCA-1);
        tempA      = 0.05:inter:0.96;
        S(:, i)    = tempA'*(Problem.upper(i)-Problem.lower(i)) + Problem.lower(i);
        S          = Problem.Evaluation(S);
        [~,MaxFNo] = NDSort(S.objs,inf);
        Fno(i)     = MaxFNo;      
    end
    [~,I] = sort(Fno);
    PV    = sort(I(1:Problem.M-1));
    DV    = sort(I(Problem.M:end));
end
```

### `GenerateBigPopulation.m`
```matlab
function [BigPopulation,tempPara] = GenerateBigPopulation(PV,groups,Archive)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

   cDecs  = Archive.decs;
   PVDecs = cDecs(:,PV);
   
   BigPopulation = cell(1,size(PVDecs,1));	% initialize the big population to contain all the population
   cPopSize      = 16;                      % the population size of each population
   
   para.lambda = cPopSize;                                      % population size, offspring number
   tempPara    = repmat(para,size(PVDecs,1),length(groups));	% Initialize the parameters for CMA-ES
   
   for ci = 1 : size(PVDecs,1)
       % Construct the population
       BigPopulation{ci} = repmat(cDecs(ci,:),cPopSize,1);
       
       % Set the CMA-ES parameters
       for g = 1 : length(groups)	% Set the parameters for each variable group
           DVindex = groups{g};     % the variable index of this group
           N = length(DVindex);     % variable number
           
           % set the CMA-ES parameters for each group
           tempPara(ci,g).N = N;
           mu      = cPopSize/2;                % number of parents/points for recombination
           weights = log(mu+1/2) - log(1:mu)';	% muXone array for weighted recombination
           tempPara(ci,g).mu = floor(mu);
           
           tempPara(ci,g).weights = weights/sum(weights);	% normalize recombination weights array
           mueff = sum(weights)^2/sum(weights.^2);          % variance-effectiveness of sum w_i x_i
           tempPara(ci,g).mueff = mueff;
           
           tempPara(ci,g).cc    = (4+mueff/N)/(N+4+2*mueff/N);                                  % time constant for cumulation for C
           tempPara(ci,g).cs    = (mueff+2)/(N+mueff+5);                                        % t-const for cumulation for sigma control
           tempPara(ci,g).c1    = 2/((N+1.3)^2+mueff);                                          % learning rate for rank-one update of C
           tempPara(ci,g).cmu   = min(1-tempPara(ci,g).c1,2*(mueff-2+1/mueff)/((N+2)^2+mueff));	% and for rank-mu update
           tempPara(ci,g).damps = 1 + 2*max(0,sqrt((mueff-1)/(N+1))-1) + tempPara(ci, g).cs;    % damping for sigma
           tempPara(ci,g).chiN  = N^0.5*(1-1/(4*N)+1/(21*N^2));                                 % expectation of ||N(0,I)|| == norm(randn(N,1))
           
           % variable parameters
           paraDecs = BigPopulation{ci};
           tempPara(ci,g).xmean = mean(paraDecs(:,DVindex))';
           tempPara(ci,g).sigma = 0.1;          % coordinate wise standard deviation (step size)
           tempPara(ci,g).pc    = zeros(N,1);	% evolution paths for C and sigma
           tempPara(ci,g).ps    = zeros(N,1);	% evolution paths for C and sigma
           B = eye(N,N);	% B defines the coordinate system
           tempPara(ci,g).B = B;
           D = ones(N,1);	% diagonal D defines the scaling
           tempPara(ci,g).D = D;
           tempPara(ci,g).C = B*diag(D.^2)*B';	% covariance matrix C
           tempPara(ci,g).eigeneval = 0;
           tempPara(ci,g).counteval = 0;
       end
   end
end
```

### `GroupDV.m`
```matlab
function group = GroupDV(Problem,DV,PV,nPerGroup)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

   ub  = Problem.upper(DV);
   lb  = Problem.lower(DV); 
   dim = length(DV);	% the dim of the distance variables
   
   fixPV  = (Problem.lower(PV)+Problem.upper(PV))/2;	% the value vector for position variables
   meanDV = (ub+lb)/2;
   
   f_archiveF = repmat(zeros(dim,dim),1,Problem.M);	% cell(dim, dim);
   
   fhatDVDecs     = repmat(lb,dim,1) + eye(dim).*repmat(meanDV,dim,1);
   fhatDecs       = [repmat(fixPV,dim,1) fhatDVDecs];
   fhatPopulation = Problem.Evaluation(fhatDecs);
   fhat_archiveF  = fhatPopulation.objs;
   
   lambdaF = repmat(zeros(dim,dim),1,Problem.M);	% cell(dim, dim);
   
   p1Dec   = [fixPV lb]; 
   tempFp1 = Problem.Evaluation(p1Dec);
   fp1F    = tempFp1.objs;
   
   for i = 1 : dim-1      
       fp2 = fhat_archiveF(i,:);
       for j = i+1 : dim
           
           fp3 = fhat_archiveF(j,:);

           p4      = lb;
           p4(i)   = meanDV(i);	% temp;
           p4(j)   = meanDV(j);	% temp;
           p4Dec   = [fixPV p4];
           tempFp4 = Problem.Evaluation(p4Dec);
           fp4     = tempFp4.objs;
           
           f_archiveF(i,j:dim:size(f_archiveF,2)) = fp4;

           d1 = fp2 - fp1F;
           d2 = fp4 - fp3;
           
           lambdaF(i,j:dim:size(lambdaF,2)) = abs(d1-d2);
       end
   end
   
   % Check for each objective
   bigTheta = false(length(DV));
   
   for i = 1 : Problem.M
       
       lambda = lambdaF(:,(i-1)*dim+1:i*dim);
       fp1    = fp1F(i);
       
       fhat_archive = fhat_archiveF(:,i);
       
       tempF_archive = f_archiveF(:,(i-1)*dim+1:i*dim);
       f_archive     = tempF_archive + tempF_archive';
       
       F1 = ones(dim,dim)*fp1;
       F2 = repmat(fhat_archive',dim,1);
       F3 = repmat(fhat_archive,1,dim);
       F4 = f_archive;
       
       FS   = cat(3,F1,F2,F3,F4);
       Fmax = max(FS,[],3);
       
       FS       = cat(3,F1+F4,F2+F3);
       Fmax_inf = max(FS,[],3);

       theta = false(dim);

       muM       = eps/2;
       gamma     = @(n)((n.*muM)./(1-n.*muM));
       errub     = gamma(dim^0.5)*Fmax;
       I2        = lambda >= errub;
       theta(I2) = 1;
       
       % add, then find not less than 1
       bigTheta = bigTheta + theta;
   end
   
   bigTheta(bigTheta>0) = true;

   L = size(bigTheta,1);	% number of vertex
   
   % Breadth-first search
   labels = zeros(1,L);     % all vertex unexplored at the begining
   rts    = [];
   ccc    = 0;              % connected components counter
   while true
       ind = find(labels==0);
       if ~isempty(ind)
           fue  = ind(1);   % first unexplored vertex
           rts  = [rts fue];
           list = [fue];
           ccc  = ccc + 1;
           labels(fue) = ccc;
           while true
               list_new = [];
               for lc = 1 : length(list)
                   p   = list(lc);              % point
                   cp  = find(bigTheta(p,:));	% points connected to p
                   cp1 = cp(labels(cp)==0);     % get only unexplored vertecies
                   labels(cp1) = ccc;
                   list_new    = [list_new cp1];
               end
               list = list_new;
               if isempty(list)
                   break;
               end
           end
       else
           break;
       end
   end
   
   group_num = max(labels);
   allgroups = cell(1,group_num);
   for i = 1 : group_num
       allgroups{i} = find(labels==i);
   end
   
   h = @(x)(length(x)==1);
   sizeone = cellfun(h,allgroups);
   
   seps = allgroups(sizeone);
   seps = cell2mat(seps);
   
   allgroups(sizeone) = [];
   nonseps            = allgroups;

   group = {};
   for ns = 1 : length(nonseps)     % the non-seperate variables
       group = {group{1:end} DV(nonseps{ns})};
   end

   for g = 1 : nPerGroup : length(seps)
       index = seps(g:min(g+nPerGroup-1,length(seps)));
       group = {group{1: end} DV(index)};
   end
end
```

### `LoadCMAESparameters.m`
```matlab
function tempPara = LoadCMAESparameters(Problem, groups, popSize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

   para.lambda = popSize;	% population size, offspring number   
   tempPara    = repmat(para,1,length(groups));
   
   for g = 1 : length(groups)	% Set the parameters for each variable group
       
       DVindex = groups{g};     % the variable index of this group
       N = length(DVindex);     % variable number
       
       % set the CMA-ES parameters for each group
       tempPara(g).N = N;
       
       mu      = tempPara(g).lambda/2;      % number of parents/points for recombination
       weights = log(mu+1/2) - log(1:mu)';  % muXone array for weighted recombination
       tempPara(g).mu = floor(mu);
       
       tempPara(g).weights = weights/sum(weights);	% normalize recombination weights array
       mueff = sum(weights)^2/sum(weights.^2);      % variance-effectiveness of sum w_i x_i
       tempPara(g).mueff = mueff;
       
       tempPara(g).cc    = (4+mueff/N)/(N+4+2*mueff/N);     % time constant for cumulation for C
       tempPara(g).cs    = (mueff+2)/(N+mueff+5);           % t-const for cumulation for sigma control
       tempPara(g).c1    = 2/((N+1.3)^2+mueff);             % learning rate for rank-one update of C
       tempPara(g).cmu   = min(1-tempPara(g).c1,2*(mueff-2+1/mueff)/((N+2)^2+mueff));	% and for rank-mu update
       tempPara(g).damps = 1 + 2*max(0,sqrt((mueff-1)/(N+1))-1) + tempPara(g).cs;       % damping for sigma
       tempPara(g).chiN  = N^0.5*(1-1/(4*N)+1/(21*N^2));	% expectation of ||N(0,I)|| == norm(randn(N,1))
       
       % variable parameters
       tempPara(g).xmean = Problem.lower(DVindex)' + (Problem.upper(DVindex)'-Problem.lower(DVindex)').*rand(N,1);	% 10*rand(N, 1); objective variables initial point
       tempPara(g).sigma = 0.5;             % coordinate wise standard deviation (step size)
       tempPara(g).pc    = zeros(N,1);      % evolution paths for C and sigma
       tempPara(g).ps    = zeros(N,1);      % evolution paths for C and sigma
       B = eye(N,N);                        % B defines the coordinate system
       tempPara(g).B = B;
       D = ones(N,1);                       % diagonal D defines the scaling
       tempPara(g).D = D;
       tempPara(g).C = B*diag(D.^2)*B';     % covariance matrix C
       tempPara(g).eigeneval = 0;
       tempPara(g).counteval = 0;
   end
end
```

### `Operator.m`
```matlab
function [para,pop,BestVal,BestIndividual] = Operator(Problem,para,bestmem,dim_index)  
% CMA-ES

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

    N     = para.N;     % number of decision variables/problem dimension
    xmean = para.xmean;	% decision variables initial point
    sigma = para.sigma;	% coordinate wise standard deviation (step size)

    % Strategy parameter setting: Selection  
    lambda  = para.lambda;	% population size, offspring number
    mu      = para.mu;   	% floor(mu);        
    weights = para.weights;	% normalization of recombination weights array
    mueff   = para.mueff;	% variance-effectiveness of sum w_i x_i

    % Strategy parameter setting: Adaptation
    cc    = para.cc;   	% time constant for cumulation for C
    cs    = para.cs;  	% t-const for cumulation for sigma control
    c1    = para.c1;  	% learning rate for rank-one update of C
    cmu   = para.cmu;
    damps = para.damps;	% damping for sigma 
    chiN  = para.chiN;	% expectation of ||N(0,I)|| == norm(randn(N,1))

    % Initialize dynamic (internal) strategy parameters and constants
    pc        = para.pc;                % zeros(N, 1); 
    ps        = para.ps;                % zeros(N, 1); evolution paths for C and sigma
    B         = para.B;                 % eye(N, N); B defines the coordinate system
    D         = para.D;                 % ones(N, 1); diagonal D defines the scaling
    C         = para.C;                 % B * diag(D.^2) * B'; covariance matrix C
    invsqrtC  = B * diag(D.^-1) * B';	% C^-1/2 
    eigeneval = para.eigeneval;         % 0; track update of B and D  
    counteval = para.counteval;

    arx         = repmat(xmean,1,lambda) + sigma*B*(repmat(D,1,lambda).*randn(N,lambda));
    arx(arx<0)  = 0;
    arx(arx>10) = 10;

    tempDecs   = repmat(bestmem,lambda,1);
    pop        = arx';
    tempDecs(:,dim_index) = pop;
    population = Problem.Evaluation(tempDecs);
    objs       = population.objs;

    arfitness = sum(objs.^2,2);
    counteval = counteval + lambda;

    % Sort by fitness and compute weighted mean into xmean
    [arfitness,arindex] = sort(arfitness);	% minimization
    xold  = xmean;
    xmean = arx(:,arindex(1:mu))*weights;	% recombination, new mean value

    % Cumulation: Update evolution paths
    ps   = (1-cs)*ps ...
           + sqrt(cs*(2-cs)*mueff)*invsqrtC*(xmean-xold)/sigma;	% Eq.24
    hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/lambda))/N < 2+4/(N+1);
    pc   = (1-cc)*pc ...
           + hsig*sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma;

    % Adapt covariance matrix C
    artmp = (1/sigma)*(arx(:,arindex(1:mu))-repmat(xold,1,mu));	% mu difference vectors
    C     = (1-c1-cmu)*C ...                 	% regard old matrix
            + c1*(pc*pc' ...                 	% plus rank one update
            +(1-hsig)*cc*(2-cc)*C) ...      	% minor correction if hsig==0
            + cmu*artmp*diag(weights)*artmp';	% Eq.30 plus rank mu update

    % Adapt step size sigma
    sigma = sigma*exp((cs/damps)*(norm(ps)/chiN-1));

    % Update B and D from C
    if counteval - eigeneval > lambda/(c1+cmu)/N/10	% to achieve O(N^2)
        eigeneval = counteval;
        C = triu(C) + triu(C,1)';	% enforce symmetry
        if any(any(isnan(C))) || any(any(C==Inf))
            C = B*diag(D.^2)*B';
            C = triu(C) + triu(C,1)';	% enforce symmetry 
        end
        [B,D] = eig(C);	% eigen decomposition, B==normalized eigenvectors
        D     = sqrt(max(diag(D),0));     
    end
    BestVal        = arfitness(1);
    BestIndividual = population(arindex(1));

    % record the variable parameters
    para.xmean = xmean;     % objective variables initial point
    para.sigma = sigma;     % coordinate wise standard deviation (step size)
    para.pc    = pc;        % evolution paths for C and sigma
    para.ps    = ps;        % evolution paths for C and sigma
    para.B     = B;
    para.D     = D;
    para.C     = C;         % B * diag(D.^2) * B'; covariance matrix C
    para.eigeneval = eigeneval;
    para.counteval = counteval;
end
```

### `S3CMAES.m`
```matlab
classdef S3CMAES < ALGORITHM
% <2020> <multi/many> <real/integer> <large/none>
% Scalable small subpopulations based covariance matrix adaptation
% evolution strategy

%------------------------------- Reference --------------------------------
% H. Chen, R. Cheng, J. Wen, H. Li, and J. Weng. Solving large-scale
% many-objective optimization problems by covariance matrix adaptation
% evolution strategy with scalable small subpopulations. Information
% Sciences, 2020, 509: 457-469.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

    methods
        function main(Algorithm,Problem)
            %% Detect the group of each distance variable
            nPer      = 50;	% Sample size to divide the convergence- and diversity-related variables
            nPerGroup = 35;	% the group size for separative variables

            [PV, DV] = ControlVariableAnalysis(Problem,nPer);	% divide the convergence- and diversity-related variables        
            Groups   = GroupDV(Problem,DV,PV,nPerGroup);     	% divide the convergence-related variables based on correlation

            popN = 5;	% the number of sub-populations
            % V: random unit vectors in diversity space
            V = 0.05 + 0.9*rand(popN,length(PV));
            % extend to the diversity space
            V = repmat(Problem.lower(PV),popN,1)+V.*(repmat((Problem.upper(PV)-Problem.lower(PV)),popN,1));  

            % CMA-ES parameters
            popSize         = 6 + floor(3*log(nPerGroup));	% the population size for CMA-ES
            tempParaOnePopu = LoadCMAESparameters(Problem,Groups,popSize);
            CMAParaMPopu    = repmat(tempParaOnePopu,popN,1);

            BigPopulation = cell(1,popN);	% initialize the big population to contain all the population
            % the diversity-related variables of all the solutions in a sub-population are the same
            for i = 1 : popN
                tempDecs         = zeros(popSize,Problem.D);
                tempDecs(:, PV)  = repmat(V(i, :),popSize,1);
                DVPositionM      = Problem.lower(DV) + (Problem.upper(DV)-Problem.lower(DV)).*rand(1, length(DV));
                tempDecs(:, DV)  = repmat(DVPositionM,popSize,1);
                BigPopulation{i} = tempDecs;
            end

            %% Optimization
            Archive      = Problem.Evaluation(BigPopulation{1});
            lastBestValA = zeros(1,popN);	% Record the best val last iteration
            stopTag      = false(1,popN);
            unUpdateNum  = zeros(1,popN);
            unChangeThr  = 1e-10;
            firstTag     = true;

            ConvergedSolutionSet = [];	% Record the converged solutions
            while Algorithm.NotTerminated(Archive)

                Archive = ConvergedSolutionSet;
                popN    = length(BigPopulation);
                for p = 1 : popN	% evolute each small population
                    if stopTag(p)	% if the subpopulaton has converged, then no evolve it
                        continue;
                    end

                    tempDecs = BigPopulation{p};	% obtain the p-th population
                    PVDecs   = tempDecs(:,PV);
                    bestmem  = tempDecs(1,:);   	% select the first individual as best member

                    tempDecs        = zeros(popSize,Problem.D);	% record the new population
                    tempDecs(:, PV) = PVDecs;                   % repmat(V(p, :), popSize, 1);

                    for g = 1 : length(Groups) % evolute each group of convergence-related variables for a sub-population
                        dim_index = Groups{g};
                        % employ the CMA-ES
                        [CMAParaMPopu(p,g),pop,BestVal,BestIndividual] = Operator(Problem,CMAParaMPopu(p,g),bestmem,dim_index);
                        tempDecs(:,dim_index) = pop;
                    end
                    Archive          = [Archive,BestIndividual];            
                    BigPopulation{p} = tempDecs;	% record the new position for this small population

                    if abs(lastBestValA(p)-BestVal) < unChangeThr	% check whether the sub population has converged
                        unUpdateNum(p)       = unUpdateNum(p) + 1;
                        stopTag(p)           = true;
                        ConvergedSolutionSet = [ConvergedSolutionSet,BestIndividual]; 
                    end
                    lastBestValA(p) = BestVal;
                end

                % generate new sub-populations for next stage
                Tag = Problem.FE > 0.6*Problem.maxFE && firstTag;
                if sum(stopTag) == length(stopTag) || Tag 
                    firstTag = false;	% The first stage has been over

                    % evolute the diversity-related variables
                    for repPV = 1 : 200  % 200 denotes the repeat times for diversity-related variables
                        CR      = 0.2;
                        F       = 0.5;
                        ExiDecs = Archive.decs;
                        ExiPV   = ExiDecs(:, PV);
                        [N,D]   = size(ExiPV);
                        Parent1Dec   = ExiPV;
                        Parent2Dec   = ExiPV(randperm(N), :);
                        Parent3Dec   = ExiPV(randperm(N), :);
                        OffspringDec = Parent1Dec;
                        Site = rand(N,D) < CR;
                        OffspringDec(Site) = OffspringDec(Site) + F*(Parent2Dec(Site)-Parent3Dec(Site));

                        Lower = repmat(Problem.lower(PV),N,1);	% Lower boundary
                        Upper = repmat(Problem.upper(PV),N,1);	% Upper boundary
                        OffspringDec = max(min(OffspringDec,Upper),Lower);

                        newDecs       = ExiDecs;
                        newDecs(:,PV) = OffspringDec;
                        newPop        = Problem.Evaluation(newDecs);

                        % environmental selection
                        Archive = UpdateArchive(Problem.N,[newPop,Archive]);
                    end
                    % generate new sub-populations
                    [BigPopulation,CMAParaMPopu] = GenerateBigPopulation(PV,Groups,Archive);

                    popN         = length(BigPopulation);
                    lastBestValA = 1e+20*ones(1,popN);
                    stopTag      = false(1,popN);
                    unUpdateNum  = zeros(1,popN);            
                    ConvergedSolutionSet = [];
                end

                % Obtain the output solutions
                if Problem.FE >= Problem.maxFE
                    Decs = zeros(length(BigPopulation),Problem.D);
                    for bp = 1 : length(BigPopulation)
                        DecPop      = BigPopulation{bp};
                        Decs(bp, :) = DecPop(1,:);
                    end
                    finalPop = Problem.Evaluation(Decs);
                    Archive  = UpdateArchive(Problem.N,[finalPop,Archive]);
                end
            end
        end
    end
end
```

### `UpdateArchive.m`
```matlab
function Archive = UpdateArchive(N,combinePopulation)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

    % Remove the dominated solutions
    Archive = combinePopulation(NDSort(combinePopulation.objs,1)==1);
    
    % Update the Archive outPopulation, if the number of solutions is larger than the population size
    if length(Archive) > N
        PopObj = Archive.objs;
        
        Choose = false(1,size(PopObj,1));       
        % Select the extreme solutions
        [~,extreme]     = min(pdist2(PopObj,eye(size(PopObj,2)),'cosine'),[],1);
        Choose(extreme) = true;
        
        %% Lp-norm-distances between each two solutions
        LpNormD = pdist2(PopObj,PopObj,'minkowski',0.5);
        while sum(Choose) < N
            Remain   = find(~Choose);
            [~, rho] = max(min(LpNormD(Remain,Choose),[],2));
            Choose(Remain(rho)) = true;
        end
        Archive = Archive(Choose);
    end
end
```
