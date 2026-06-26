# SAMOEA-TL2M

**Tags**: <2025> <multi/many> <real/integer> <expensive>

## Description
Surrogate-assisted multiobjective evolutionary algorithm based on two-level model management

## Reference
Y. Liu, J. Ding, Q. Li, F. Li, and J. Liu. A two-level model management-based surrogate-assisted evolutionary algorithm for medium-scale expensive multiobjective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2025, 55(11): 8166-8180.

## Source Code

### `SAMOEATL2M.m`
```matlab
classdef SAMOEATL2M < ALGORITHM
% <2025> <multi/many> <real/integer> <expensive>
% Surrogate-assisted multiobjective evolutionary algorithm based on two-level model management
% G     ---  20 --- Number of generations before updating suoorgate models
% KE    ---   5 --- The maximum number of infill solutions at each generation
% alpha --- 0.4 --- The threshold for the accuracy rate

%------------------------------- Reference --------------------------------
% Y. Liu, J. Ding, Q. Li, F. Li, and J. Liu. A two-level model
% management-based surrogate-assisted evolutionary algorithm for
% medium-scale expensive multiobjective optimization. IEEE Transactions on
% Systems, Man, and Cybernetics: Systems, 2025, 55(11): 8166-8180.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuanchao Liu (email:liuyuanchao@ise.neu.edu.cn)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [G,KE,alpha] = Algorithm.ParameterSet(20,5,0.4);
            stage = 1;

            %% Generate the reference points and population
            [~,Problem.N] = UniformPoint(Problem.N,Problem.M);
            NI  = 100;
            NP  = NI;
            NI  = 11*Problem.D-1;
            P   = UniformPoint(NI, Problem.D, 'Latin');
            A2  = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            Num = randi([1, NI], 1, NP);
            A1  = A2(Num);

            %% Optimization
            while Algorithm.NotTerminated(A2)
                nobj   = Problem.M;
                yyy    = A2.objs;
                x      = A2.decs;
                yy     = (yyy-min(yyy))./(max(yyy)-min(yyy));
                PopObj = A1.objs;
                PopObj = (PopObj-min(yyy))./(max(yyy)-min(yyy));
                for i = 1 : nobj
                    dmodel   = TrainModel2(x,yy(:,i),Problem.M,Problem.D);
                    Model{i} = dmodel;
                end
                Combine = A2;
                fmin    = repmat(min(A2.objs,[],1),length(Combine),1);
                fmax    = repmat(max(A2.objs,[],1),length(Combine),1);
                PopObjA = (Combine.objs-fmin)./(fmax-fmin);
                [N,~]   = size(PopObjA);
                % Calculate the shifted distance between each two solutions
                sde = inf(N);
                for k = 1 : N
                    SPopObj = max(PopObjA,repmat(PopObjA(k,:),N,1));
                    for j = [1:k-1,k+1:N]
                        sde(k,j) = norm(PopObjA(k,:)-SPopObj(j,:));
                    end
                end
                SDE    = min(sde,[],2);
                Cd     = (SDE-min(SDE))./(max(SDE)-min(SDE));
                x      = A2.decs;
                dmodel = TrainModel2(x,Cd,Problem.M,Problem.D);
                ModelC = dmodel;
                % RBF-Assisted Evolutionary Multi-Objective Search
                PopDec = A1.decs;
                w      = 1;
                while w <= G
                    drawnow();
                    OffDec = OperatorGA(Problem,PopDec);
                    PopDec = [PopDec;OffDec];
                    OffObj = [];
                    for i = 1 : nobj
                        TestSamNum  = size(OffDec,1);
                        OffObj(:,i) = RBF(OffDec, Model{i}, TestSamNum);
                    end
                    PopObj = [PopObj;OffObj];
                    [FrontNo,MaxFNo] = NDSort(PopObj,size(OffDec,1));
                    Next       = FrontNo < MaxFNo;
                    TestSamNum = size(PopDec,1);
                    PopD       = RBF(PopDec, ModelC, TestSamNum);
                    Last       = find(FrontNo==MaxFNo);
                    [~,Rank]   = sort(PopD(Last),'descend');
                    Next(Last(Rank(1:size(OffDec,1)-sum(Next)))) = true;
                    PopObj = PopObj(Next,:);
                    PopDec = PopDec(Next,:);
                    w      = w + 1;
                end
                PopDec1 = PopDec;
                % Two-Level Model Management Strategy
                Infill_dataC  = [];
                dist2         = pdist2(real(PopDec1),real(A2.decs));
                DF            = find(min(dist2,[],2)<1e-5);
                PopDec1(DF,:) = [];
                PopObj(DF,:)  = [];
                if ~isempty(PopDec1)
                    if stage == 2
                        NumC = round(KE*(1-alpha));
                        if NumC < 1
                            NumC = 1;
                        end
                        [FrontNo,MaxFNo] = NDSort(PopObj,1);
                        Last = find(FrontNo==MaxFNo);
                        s    = [];
                        Pdec = A2.decs;
                        Pobj = A2.objs;
                        for h = 1 : Problem.M
                            [~, s(:,h)] = idw_prediction_and_uncertainty(PopDec1, Pdec, Pobj(:,h));
                        end
                        PopD         = sum(s,2);
                        [~,Rank]     = sort(PopD(Last),'descend');
                        Infill_dataC = [];
                        if size(PopDec1,1) > NumC
                            h = 1;
                            while size(Infill_dataC,1) < NumC
                                if h == 1
                                    Infill_dataC = PopDec1(Last(Rank(h)),:);
                                else
                                    Infill_TdataC = PopDec1(Last(Rank(h)),:);
                                    dist2 = pdist2(real(Infill_TdataC),real(Infill_dataC));
                                    if min(dist2) > 1e-5
                                        Infill_dataC = [Infill_dataC;Infill_TdataC];
                                    end
                                end
                                h = h +1;
                            end
                        else
                            Infill_dataC = PopDec1;
                        end
                    end
                    if stage ==  1
                        Num = KE;
                    else
                        Num = KE - NumC;
                    end
                    [FrontNo,MaxFNo] = NDSort(PopObj,1);
                    Last = find(FrontNo==MaxFNo);
                    if length(Last) <= Num
                        Infill_data = PopDec1(Last,:);
                    else
                        Next    = FrontNo < MaxFNo;
                        Combine = A2.objs;
                        Combine = [Combine;PopObj];
                        fmin    = repmat(min(Combine,[],1),size(Combine,1),1);
                        fmax    = repmat(max(Combine,[],1),size(Combine,1),1);
                        PopObjA = (Combine-fmin)./(fmax-fmin);
                        [N,~]   = size(PopObjA);
                        % Calculate the shifted distance between each two solutions
                        sde     = inf(N);
                        SPopObj = [];
                        for k = 1 : N
                            SPopObj = max(PopObjA,repmat(PopObjA(k,:),N,1));
                            for j = [1:k-1,k+1:N]
                                sde(k,j) = norm(PopObjA(k,:)-SPopObj(j,:));
                            end
                        end
                        SDE         = min(sde,[],2);
                        Cd          = (SDE-min(SDE))./(max(SDE)-min(SDE));
                        PopD        = Cd(length(A2)+1:end);
                        [~,Rank]    = sort(PopD(Last),'descend');
                        Infill_data = [];
                        if size(PopDec1,1) > Num
                            h = 1;
                            while size(Infill_data,1) < Num
                                if h == 1
                                    Infill_data = PopDec1(Last(Rank(h)),:);
                                else
                                    Infill_Tdata = PopDec1(Last(Rank(h)),:);
                                    dist2 = pdist2(real(Infill_Tdata),real(Infill_data));
                                    if min(dist2) > 1e-5
                                        Infill_data = [Infill_data;Infill_Tdata];
                                    end
                                end
                                h = h +1;
                            end
                        else
                            Infill_data = PopDec1;
                        end
                    end
                    if stage ==  2
                        Infill_data = [Infill_data;Infill_dataC];
                    end
                    % Update population
                    if ~isempty(Infill_data)
                        New     = Problem.Evaluation(Infill_data);
                        A2      = [A2,New];
                        NextA1  = [A1,New];
                        [FrontNoNA1,MaxFNo] = NDSort(NextA1.objs,NextA1.cons,NP);
                        Next    = FrontNoNA1 < MaxFNo;
                        Combine = NextA1.objs;
                        fmin    = repmat(min(Combine,[],1),size(Combine,1),1);
                        fmax    = repmat(max(Combine,[],1),size(Combine,1),1);
                        PopObjA = (Combine-fmin)./(fmax-fmin);
                        [N,~]   = size(PopObjA);
                        % Calculate the shifted distance between each two solutions
                        sde = inf(N);
                        SPopObj = [];
                        for k = 1 : N
                            SPopObj = max(PopObjA,repmat(PopObjA(k,:),N,1));
                            for j = [1:k-1,k+1:N]
                                sde(k,j) = norm(PopObjA(k,:)-SPopObj(j,:));
                            end
                        end
                        SDE      = min(sde,[],2);
                        Cd       = (SDE-min(SDE))./(max(SDE)-min(SDE));
                        Last     = find(FrontNoNA1==MaxFNo);
                        [~,Rank] = sort(Cd(Last),'descend');
                        Next(Last(Rank(1:NP-sum(Next)))) = true; %%Use the shifted distance to select the offspring
                        A1       = NextA1(Next);
                    end
                    % ARI
                    if ~isempty(Infill_data)
                        rp        = sum(Next(NP+1:end))/size(Infill_data,1);
                        FrontNoA1 = NDSort(A1.objs,A1.cons,NP);
                        rn        = length(find(FrontNoA1(NP - sum(Next(NP+1:end))+1:end)==1))/sum(Next(NP+1:end));
                        ARI       = min(rp, rn);
                        if ARI < alpha
                            stage = 2;
                        else
                            stage = 1;
                        end
                    else
                        stage = 2;
                    end
                else
                    NewPop = UniformPoint(1, Problem.D, 'Latin');
                    New    = Problem.Evaluation(NewPop);
                    A2     = [A2,New];
                end
            end
        end
    end
end

function y = RBF(x, p, N)
    Centers = p.Centers;
    Spreads = p.Spreads;
    W2      = p.W2;
    B2      = p.B2;
    TestDistance      = dist(Centers,x');
    TestSpreadsMat    = repmat(Spreads,1,N);
    TestHiddenUnitOut = radbas(TestDistance./TestSpreadsMat);
    y = (W2*TestHiddenUnitOut+repmat(B2,1,N))';
end
```

### `TrainModel2.m`
```matlab
function models = TrainModel2(tr_x, tr_y, M, D)
% Train a RBFN Model by using the least square method to train weight.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuanchao Liu (email:liuyuanchao@ise.neu.edu.cn)

    models = RBFMyself2(tr_x, tr_y, M, D);

    function models = RBFMyself2(tr_x, tr_y, M, D)
        ClusterNum = round(sqrt(M+D)+3);
        [models.Centers, models.Spreads, models.W2, models.B2] = clusterRBF(tr_x, tr_y,ClusterNum);
    end

    function [Centers, Spreads, W2, B2] = clusterRBF(SamIn, SamOut, ClusterNum)
        v       = size(SamIn,2);
        Overlap = 1.0;              % Overlap coefficient of hidden node
        SamNum  = size(SamIn,1);    % Number of all the samples
        nn      = 0;
        while nn==100 || nn==0
            index   = randi([1,SamNum],ClusterNum,1);
            Centers = SamIn(index,:);   %Initialise the clustering center
            n       = 1;
            while n < 100
                NumberInClusters = zeros(ClusterNum,1);         % Number of samples in each class, default is 0
                IndexInClusters  = zeros(ClusterNum,SamNum);    % Index of samples in each class
                % Classify all samples by the least distance principle
                for i = 1 : SamNum
                    AllDistance = dist(Centers,SamIn(i,:)');        % Calculate the distance between the i-th solution and each clustering center
                    [~,Pos]     = min(AllDistance);                 % Minimum distance,training input is the index of clustering center
                    NumberInClusters(Pos) = NumberInClusters(Pos) + 1;
                    IndexInClusters(Pos,NumberInClusters(Pos)) = i; % Stores the training indexes belonging to this class in turn
                end
                % Store the old clustering centers
                OldCenters = Centers;
                % Recalculate the clustering centers
                for i = 1 : ClusterNum
                    Index = IndexInClusters(i,1:NumberInClusters(i));   % Extract the training input index belonging to this class
                    Centers(i,:) = mean(SamIn(Index,:),1);              % Take the average of each class as the new clustering center
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
        W2Ex = SamOut'*pinv(HiddenUnitOutEx);               % Find the generalized output weight
        W2   = W2Ex(:,1:ClusterNum);                        % Output weight
        B2   = W2Ex(:,ClusterNum+1);                        % Offsets (thresholds)
    end
end
```

### `idw_prediction_and_uncertainty.m`
```matlab
function [prediction, s] = idw_prediction_and_uncertainty(x, known_points, values, p)
% IDW_PREDICTION_AND_UNCERTAINTY Calculate IDW prediction and uncertainty
%
%   [PREDICTION, S2] = IDW_PREDICTION_AND_UNCERTAINTY(X, KNOWN_POINTS,
%   VALUES, P) returns the IDW prediction and local variance estimate at
%   point X.
%
%   X is a vector representing the coordinates of the prediction point.
%   KNOWN_POINTS is an NxM matrix where each row represents the coordinates
%   of a known point.
%
%   VALUES is an Nx1 vector of values at the known points.
%   P is the power parameter for IDW, default is 2.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuanchao Liu (email:liuyuanchao@ise.neu.edu.cn)
    
    if nargin < 4
        p = 2; % Default power parameter
    end
    % Number of known points
    n = size(x, 1);
    % Calculate the weights
    for i = 1 : n
        A          = repmat(x(i,:),size(known_points, 1),1) - known_points;
        distance   = sqrt(sum(A.^2, 2));
        weights    = 1 ./ (distance.^p);
        weights    = weights / sum(weights);
        prediction = sum(weights .* values);
        s(i)       = sum(weights .* (values - prediction).^2);
    end
end
```
