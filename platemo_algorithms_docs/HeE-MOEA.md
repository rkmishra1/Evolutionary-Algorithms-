# HeE-MOEA

**Tags**: <2019> <multi> <real/integer> <expensive>

## Description
Multiobjective evolutionary algorithm with heterogeneous ensemble based

## Reference
D. Guo, Y. Jin, J. Ding, and T. Chai. Heterogeneous ensemble-based infill criterion for evolutionary multiobjective optimization of expensive problems. IEEE Transactions on Cybernetics, 2019, 49(3): 1012-1025.

## Source Code

### `HeEMOEA.m`
```matlab
classdef HeEMOEA < ALGORITHM
% <2019> <multi> <real/integer> <expensive>
% Multiobjective evolutionary algorithm with heterogeneous ensemble based
% infill criterion
% Ke --- 5 --- Number of the solutions to be revaluated

%------------------------------- Reference --------------------------------
% D. Guo, Y. Jin, J. Ding, and T. Chai. Heterogeneous ensemble-based infill 
% criterion for evolutionary multiobjective optimization of expensive 
% problems. IEEE Transactions on Cybernetics, 2019, 49(3): 1012-1025.
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
            assert(~isempty(ver('nnet')),'The execution of HeE-MOEA requires the Deep Learning Toolbox.');
            
            %% Parameter setting
            Ke = Algorithm.ParameterSet(5);
            NI = 11*Problem.D-1;
            P  = UniformPoint(NI,Problem.D,'Latin');
            A  = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            
            %% Settings of the Ensemble Model
            L    = 11*Problem.D-1+25;
            ADec = A.decs;
            AObj = A.objs;
            
            Selection = PSOMyself(ADec, AObj(:,1),Problem.D);
            str1={'FE', 'FS', 'NONE'};
            str2={'RBF1','SVM', 'RBF2'};%, 'KNN', 'DTree'
            for i=1:length(str1)
                for j=1:length(str2)
                    str{(i-1)*length(str2)+j, 1}=str1{i};
                    str{(i-1)*length(str2)+j, 2}=str2{j};
                end
            end

            while Algorithm.NotTerminated(A)                
                %% Update the model 
              % Select the train data
                ADec = A.decs;
                AObj = A.objs;
                Numdata = size(ADec,1);
                if Numdata <=L
                    % fprintf('No training data decrease\n');                    
                else
                    FrontNo   = NDSort(AObj,Numdata);
                    [~,index] = sort(FrontNo);
                    ADec1 = ADec(index(1:floor(L/2)), :);
                    AObj1 = AObj(index(1:floor(L/2)), :);
                    ADec2 = ADec(index(floor(L/2)+1:end), :);
                    AObj2 = AObj(index(floor(L/2)+1:end), :);
                    index = randperm(size(ADec2,1));
                    ADec  = [ADec1;ADec2(index(1:L-floor(L/2)),:)];
                    AObj  = [AObj1;AObj2(index(1:L-floor(L/2)),:)];
                end
              
                % Train the model
                Models = TrainModel(ADec,AObj,Selection,str,Problem.M,Problem.D);
              
                % Optimization
                New    = NSGA2ESelection(ADec,AObj,Models,str,Problem,Ke);
                PopNew = Problem.Evaluation(New);
                A      = [A PopNew];
            end
        end
    end
end
```

### `NSGA2ESelection.m`
```matlab
function Centers = NSGA2ESelection(tr_x, tr_y, models, str, Problem, Ke)
% NSGA-II based selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    nn = 0;   
    while nn==100 || nn==0
        population = rand(Problem.N,Problem.D); %N*D 
        population = population.*repmat(Problem.upper,Problem.N,1)+(1-population).*repmat(Problem.lower,Problem.N,1);%N*D   
        population(:,Problem.D+1:Problem.D+Problem.M)=Estimate(population(:,1:Problem.D), tr_x, tr_y, models, str, Problem.M);%The value of objective by model
              
        [~,FrontNo,CrowdDis] = EnvironmentalSelection(population,Problem.N,Problem.D,Problem.M);%N*(D+M)

        %% Optimization of NSGA-II
        for i = 1 : floor(Problem.maxFE/Problem.N)
            MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
            parent     = population(MatingPool,:);
            offspring  = OperatorGA(Problem,parent(:,1:Problem.D),{1,20,1,20});
            offspring(:,Problem.D+1:Problem.D+Problem.M)=Estimate(offspring(:,1:Problem.D), tr_x, tr_y, models, str, Problem.M);%The value of objective by model
            [population,FrontNo,CrowdDis] = EnvironmentalSelection([population;offspring],Problem.N,Problem.D,Problem.M);
        end
        [Centers,nn] = Kmean([population,FrontNo',CrowdDis'], tr_x, Problem.N, Problem.D, Ke);
    end
end

function objv_LCB = Estimate(x, tr_x, tr_y, models, str,m)
    alpha      = 2;
    TestSamNum = size(x,1);n=length(models);
    sum        = zeros(TestSamNum, size(tr_y,2));
    for i = 1 : n
        clear y;
        stri = str(i,:);
        M    = models(i).M;
        p    = models(i).p;
        if strcmp(stri(1), 'FE')%PCA
            x_pca = x*M;
            if strcmp(stri(2), 'RBF1')%RBF
                y = sim(p,x_pca');
                y = y';
            elseif strcmp(stri(2), 'SVM')
                y = SVMtest(x_pca, p, TestSamNum,m);
            elseif strcmp(stri(2), 'RBF2')%RBF
                y = RBF2test(x_pca, p, TestSamNum);
            end

        elseif strcmp(stri(1), 'NONE')%NONE
            if strcmp(stri(2), 'RBF1')%RBF
                y = sim(p,x');y=y';
            elseif strcmp(stri(2), 'SVM')
                y = SVMtest(x, p, TestSamNum, m);
            elseif strcmp(stri(2), 'RBF2')%RBF
                y = RBF2test(x, p, TestSamNum);
            end

        elseif strcmp(stri(1), 'FS')%CSO
            x_cso = x(:,M);
            if strcmp(stri(2), 'RBF1')%RBF
                y = sim(p,x_cso');y=y';
            elseif strcmp(stri(2), 'SVM')
                y = SVMtest(x_cso, p, TestSamNum,m);
            elseif strcmp(stri(2), 'RBF2')%RBF
                y = RBF2test(x_cso, p, TestSamNum);
            end 
        end
        result(i).y = y;
        sum         = sum + y;
    end
    me  = sum/n;
    sum = zeros(TestSamNum, size(tr_y,2));
    for i = 1 : n
        y   = result(i).y;
        sum = sum+(y-me).^2;
    end
    s2       = sum/(n-1);
    objv_LCB = me-alpha*sqrt(s2);
end

function y = SVMtest(x, p, N, M)
    ps1    = p{1};
    ps2    = p{2};
    xtest0 = mapminmax('apply',x',ps1);xtest0=xtest0';
    for j = 1 : M
        for k = 1 : N
            py(k,j) = svmpredict(0,xtest0(k,:),p{j+2}, '-q');
        end
    end
    y = mapminmax('reverse',py',ps2);y=y';  
end

function y = RBF2test(x, p, N)
    Centers = p.Centers;
    Spreads = p.Spreads;
    W2      = p.W2;
    B2      = p.B2;
    TestDistance      = dist(Centers,x');
    TestSpreadsMat    = repmat(Spreads,1,N);
    TestHiddenUnitOut = radbas(TestDistance./TestSpreadsMat);
    y = (W2*TestHiddenUnitOut+repmat(B2,1,N))'; 
end

function [population,FrontNo,CrowdDis] = EnvironmentalSelection(population,N,D,M)
% The environmental selection of NSGA-II

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(population(:,D+1:D+M),[],N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(population(:,D+1:D+M),FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    population = population(Next,:);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end

function [Centers, nn]=Kmean(chromosome, tr_x, N, D, Ke)
    % Delete the multiple solutions between chromosome and the train data
    Q = [];
    P = [];
    for i = 1 : N
        PandQ = [tr_x;Q];
        matrix = dist(chromosome(i,1:D),PandQ');%1*size(PandQ,1) array
        if isempty(find(matrix<=10^-06))
            Q = [Q;chromosome(i,1:D)];
            P = [P;chromosome(i,:)];
        end
    end
    if size(Q,1) < Ke
        Centers = [];
        nn      = 100;
    else
        index   = randperm(size(Q,1));
        Centers = Q(index(1:Ke),1:D);
        n       = 1;
        si      = size(Q,1);
        % Cluster by the distance of the decision space
        while n < 100
            NumberInClusters = zeros(Ke,1); % Number of samples in each class ,default is 0
            IndexInClusters  = zeros(Ke,N); % Index of samples in each class
            % Classify all samples by the least distance principle
            for i = 1 : si
                AllDistance = dist(Centers,Q(i,1:D)');          % Calculate the distance between the i-th solution and each clustering center
                [~,Pos]     = min(AllDistance);                 % Minimum distance,training input is the index of clustering center
                NumberInClusters(Pos) = NumberInClusters(Pos) + 1;
                IndexInClusters(Pos,NumberInClusters(Pos)) = i; % Stores the training indexes belonging to this class in turn
            end
            % Store the old clustering centers
            OldCenters = Centers;
            % Recalculate the clustering centers
            for i = 1 : Ke
                Index        = IndexInClusters(i,1:NumberInClusters(i));% Extract the training input index belonging to this class
                Centers(i,:) = mean(Q(Index,1:D),1);                    % Take the average of each class as the new clustering center
            end
            % Judge whether the old and new clustering centers are consistent
            EqualNum = sum(sum(Centers==OldCenters));   % Centers and OldCenters are subtracted from each other to sum up all corresponding bits
            if EqualNum == D*Ke                         % The old and new clustering centers are consistent
                break,
            end
            n = n + 1;
        end
        nn = n;
        fprintf('k-means clustering %d\n',n);
    end
end
```

### `PCAMyself.m`
```matlab
function [x_pca, m] = PCAMyself(x)
% Feature extraction by PCA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [COEFF,~,latent] = pca(x);
    answer = 0;
    s1     = 0;
    s2     = sum(latent);
    count  = 1;
    while answer < 0.95
        s1     = s1+latent(count);
        answer = s1/s2;
        count  = count+1;
    end
    m     = COEFF(:,1:count-1);
    x_pca = x*m;
end
```

### `PSOMyself.m`
```matlab
function index = PSOMyself(x, y, D)  
% Select input features for the generation of different inputs by PSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    m         = 20;     % Population Size
    iter      = 30;     % Number of Generation
    threshold = 0.5;
    c1        = 2;
    c2        = 2;
    w         = 1.05;   % Parameter of CSO
    
    % Limit of Position
    lu     = [zeros(1, D); ones(1, D)];
    XRRmin = repmat(lu(1, :), m, 1);
    XRRmax = repmat(lu(2, :), m, 1);
    VelMax = (XRRmax-XRRmin)/10;

    % Position
    p = rand(m, D);
    p = XRRmin + (XRRmax - XRRmin) .*p;
    p(find(p<threshold))  = 0;
    p(find(p>=threshold)) = 1;
    v = VelMax .* rand(m,D);

    fitness = PSO_CostFunction(p, x, y);

    [bestever, index] = min(fitness);
    zbest             = p(index,:);
    fitnesszbest      = bestever;
    gbest             = p;
    fitnessgbest      = fitness;

    for i = 1 : iter
        v = w*v + c1*rand(m,D).*(gbest-p) + c2*rand(m,D).*(repmat(zbest, m, 1)-p);
        v = max(min(v,VelMax), -VelMax);
        p = p + v;
        p(find(p<threshold))  = 0;
        p(find(p>=threshold)) = 1;

        fitness = PSO_CostFunction(p, x, y); 

        index               = find(fitness < fitnessgbest);
        fitnessgbest(index) = fitness(index);
        gbest(index,:)      = p(index,:);
        [bestever, index]   = min(fitness);
        if  bestever < fitnesszbest
            fitnesszbest = bestever;
            zbest        = p(index, :);
        end
    end
    index = find(zbest==1);
end

function fitness = PSO_CostFunction(p, x, y)
    m       = size(p,1);
    fitness = 100*ones(m,1);
    for i = 1 : m
        index = find(p(i,:)==1);
        if ~isempty(index)
            xx = x(:,index);
            D  = Discor(xx, y);
            R  = Discor(xx,[]);
            fitness(i) = -0.8*D+0.2*R;
        end
    end
end

function D = Discor(x, y)
    [n, x_n] = size(x);
    if ~isempty(y)
        a     = norm(x);
        b     = norm(y);
        A     = a-repmat(mean(a,2),1,n)-repmat(mean(a,1),n,1)+mean(mean(a));
        B     = b-repmat(mean(b,2),1,n)-repmat(mean(b,1),n,1)+mean(mean(b));
        covxy = sum(sum(A.*B))/(n^2);
        covx  = sum(sum(A.^2))/(n^2);
        covy  = sum(sum(B.^2))/(n^2);
        D     = sqrt(covxy)/sqrt(sqrt(covx)*sqrt(covy));
    else
        if x_n == 1
            D = 0;
        else
            for i = 1 : x_n
                xa     = x(:,i);
                index  = setdiff([1:x_n],i);
                xb     = x(:,index);    
                a      = norm(xa);
                b      = norm(xb);
                A      = a-repmat(mean(a,2),1,n)-repmat(mean(a,1),n,1)+mean(mean(a));
                B      = b-repmat(mean(b,2),1,n)-repmat(mean(b,1),n,1)+mean(mean(b));
                covxy  = sum(sum(A.*B))/(n^2);
                covx   = sum(sum(A.^2))/(n^2);
                covy   = sum(sum(B.^2))/(n^2);
                D(1,i) = sqrt(covxy)/sqrt(sqrt(covx)*sqrt(covy));
            end
            D = mean(D);
        end
    end
end
```

### `TrainModel.m`
```matlab
function models = TrainModel(tr_x, tr_y, index, str, M, D)
% Train the heterogeneous ensemble Model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % PCA
    [~, m] = PCAMyself(tr_x);
    trx_fe = tr_x*m;

    % FS
    trx_fs = tr_x(:,index);

    for i = 1 : length(str)
        str3 = str(i,:);
        if strcmp(str3(1), 'FE')
            tr_xx       = trx_fe;
            models(i).M = m;
        elseif strcmp(str3(1), 'FS')
            tr_xx       = trx_fs;
            models(i).M = index;
        elseif strcmp(str3(1), 'NONE')
            tr_xx       = tr_x;
            models(i).M = [];        
        end

        if strcmp(str3(2), 'RBF1')
            models(i).p = RBFMyself1(tr_xx, tr_y);
        elseif strcmp(str3(2), 'SVM')
            models(i).p = SVMMyself(tr_xx, tr_y, M);
        elseif strcmp(str3(2), 'RBF2')
            models(i).p = RBFMyself2(tr_xx, tr_y, M, D);
        end
    end
end

function net = RBFMyself1(tr_x, tr_y)
    goal = sqrt(sum((max(tr_y)-min(tr_y)).^2))*0.05;
    net  = newrb(tr_x', tr_y', goal, 1.0, size(tr_x,2), 1e+06);
end

function models = RBFMyself2(tr_x, tr_y, M, D)
    ClusterNum = round(sqrt(M+D)+3);
    [models.Centers, models.Spreads, models.W2, models.B2] =...
        clusterRBF(tr_x, tr_y,ClusterNum);
end             

function models = SVMMyself(tr_x, tr_y, M)    
    [train_x,ps1] = mapminmax(tr_x',0,1);train_x=train_x';
    [train_y,ps2] = mapminmax(tr_y',0,1);train_y=train_y';
    models{1}     = ps1;
    models{2}     = ps2;
    for i = 1 : M
        models{2+i} = svmtrain(train_y(:,i),train_x,'-s 3 -t 3 -q');
    end
end

function [Centers, Spreads, W2, B2]=clusterRBF(SamIn, SamOut, ClusterNum)
    
    v       = size(SamIn,2);
    Overlap = 1.0;          % Overlap coefficient of hidden node 
    SamNum  = size(SamIn,1);% Number of all the samples
    
    nn = 0;
    while nn==100 || nn==0
        index   = randi([1,SamNum],ClusterNum,1);
        Centers = SamIn(index,:);   % Initialise the clustering center
        n = 1;
        while n < 100
            NumberInClusters = zeros(ClusterNum,1);         % Number of samples in each class ,default is 0?
            IndexInClusters  = zeros(ClusterNum,SamNum);    % Index of samples in each class?
            % Classify all samples by the least distance principle?
            for i = 1 : SamNum
                AllDistance = dist(Centers,SamIn(i,:)');% Calculate the distance between the i-th solution and each clustering center
                [~,Pos]     = min(AllDistance);         % Minimum distance,training input is the index of clustering center
                NumberInClusters(Pos) = NumberInClusters(Pos) + 1;
                IndexInClusters(Pos,NumberInClusters(Pos)) = i; % Stores the training indexes belonging to this class in turn
            end   
            % Store the old clustering centers
            OldCenters = Centers;
            % Recalculate the clustering centers
            for i = 1 : ClusterNum
                Index        = IndexInClusters(i,1:NumberInClusters(i));    % Extract the training input index belonging to this class
                Centers(i,:) = mean(SamIn(Index,:),1);                      % Take the average of each class as the new clustering center
            end
            % Judge whether the old and new clustering centers are consistent
            EqualNum = sum(sum(Centers==OldCenters));   % Centers and Old Centers are subtracted from each other to sum up all corresponding bits
            if EqualNum == v*ClusterNum                 % The old and new clustering centers are consistent?
                break;
            end
            n = n + 1;
        end
        nn = n;
    end
    % Calculate the spread constant (width) of each hidden node?
    AllDistances = dist(Centers,Centers');  % Calculate the distance between hidden node data centers (square matrix of ClusterNum dimension, symmetric matrix)?
    Maximum      = max(max(AllDistances));  % Find the largest distance?
    for i = 1 : ClusterNum                  % Replace the 0 on the diagonal with a larger value?
        AllDistances(i,i) = Maximum + 1;
    end
    Spreads = Overlap*min(AllDistances)';   % The minimum distance between hidden nodes is taken as the expansion constant.And convert it to a column vector?

    % Calculate the output weights of each hidden node
    Distance        = dist(Centers,SamIn');             % Calculate the distance between each sample input and each data center (Clusternum X Samnum matrix)
    SpreadsMat      = repmat(Spreads,1,SamNum);         % Clusternum X Samnum matrix
    HiddenUnitOut   = radbas(Distance./SpreadsMat);     % Calculate the hidden node output matrix;Radbas are radial basis transfer functions
    HiddenUnitOutEx = [HiddenUnitOut' ones(SamNum,1)]'; % Consider offsets (thresholds)
    W2Ex            = SamOut'*pinv(HiddenUnitOutEx);    % Find the generalized output weight
    W2              = W2Ex(:,1:ClusterNum);             % Output weight
    B2              = W2Ex(:,ClusterNum+1);             % Offsets (thresholds)
end
```
