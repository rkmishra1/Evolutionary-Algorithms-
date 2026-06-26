# EM-SAEA

**Tags**: <2025> <multi/many> <real> <constrained> <expensive>

## Description
Ensemble-based surrogate model-assisted evolutionary algorithm

## Reference
Y. Li, X. Feng, and H. Yu. Enhancing landscape approximation with ensemble-based surrogate model for expensive constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2025.

## Source Code

### `ArchiveUpdate.m`
```matlab
function [PCDec,PCObj,PCCon] = ArchiveUpdate(PopDec,PopObj,PopCon,N)
% Update archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select feasible solutions
    fIndex  = all(PopCon <= 0,2);
    FPopDec = PopDec(fIndex,:);
    FPopObj = PopObj(fIndex,:);
    FPopCon = PopCon(fIndex,:);
    if isempty(FPopDec)
        PCDec = [];
        PCObj = [];
        PCCon = [];
        return;
    else  
        NFPopDec = FPopDec(NDSort(FPopObj,1)==1,:);
        NFPopObj = FPopObj(NDSort(FPopObj,1)==1,:);
        NFPopCon = FPopCon(NDSort(FPopObj,1)==1,:);
        randind  = randperm(size(NFPopDec,1));
        PCDec    = NFPopDec(randind,:);
        PCObj    = NFPopObj(randind,:);
        PCObj2   = PCObj;
        PCCon    = NFPopCon(randind,:);
        nND      = size(NFPopDec,1);
        %% Population maintenance
        if size(PCDec,1) > N
            % Normalization
            fmax   = max(PCObj2,[],1);
            fmin   = min(PCObj2,[],1);
            PCObj2 = (PCObj2-repmat(fmin,nND,1))./repmat(fmax-fmin,nND,1);
            % Determine the radius of the niche
            d  = pdist2(PCObj2,PCObj2);
            d(logical(eye(length(d)))) = inf;
            sd = sort(d,2);
            r  = median(sd(:,min(size(PCObj2,2),size(sd,2))));
            R  = min(d./r,1);
            % Delete solution one by one
            while size(PCDec,1) > N
                [~,worst]      = max(1-prod(R,2));
                PCDec(worst,:) = [];
                PCObj(worst,:) = [];
                PCCon(worst,:) = [];
                R(worst,:)     = [];
                R(:,worst)     = [];
            end
        end
    end
end
```

### `CDPEnvironmentalSelection.m`
```matlab
function [PopDec,PopObj,PopCon] = CDPEnvironmentalSelection(PopDec,PopObj,PopCon,N,Z,Zmin)
% The environmental selection of NSGA-III using CDP

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
    [FrontNo,MaxFNo] = NDSort(PopObj,PopCon,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(PopObj(Next,:),PopObj(Last,:),N-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
    % Population for next generation
    PopDec = PopDec(Next,:);
    PopObj = PopObj(Next,:);
    PopCon = PopCon(Next,:);
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
    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes
    Hyperplane = PopObj(Extreme,:)\ones(M,1);
    a = 1./Hyperplane;
    if any(isnan(a))
        a = max(PopObj,[],1)';
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

### `EM_SAEA.m`
```matlab
classdef EM_SAEA< ALGORITHM
% <2025> <multi/many> <real> <constrained> <expensive>
% Ensemble-based surrogate model-assisted evolutionary algorithm
% wmax   --- 20 --- Maximum number of generations before updating surrogate models
% lc_num ---  5 --- Number of local models

%------------------------------- Reference --------------------------------
% Y. Li, X. Feng, and H. Yu. Enhancing landscape approximation with
% ensemble-based surrogate model for expensive constrained multiobjective
% optimization. IEEE Transactions on Evolutionary Computation, 2025.
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
            [wmax,lc_num] = Algorithm.ParameterSet(20,5); 

            %% Initialize weight vectors
            [V0,NI] = UniformPoint(Problem.N,Problem.M);
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            [ClW,~] = UniformPoint(lc_num,Problem.M);
            lc_num  = size(ClW,1);
            V       = V0;
            Vbb     = V0;
            T       = ceil(Problem.N/10);
            nr      = ceil(Problem.N/100);
            mu      = 5;
            mu1     = 1;
            kk      = 0.5;
            alpha   = 2;

            %% Detect the neighbours of each solution
            B     = pdist2(W,W);
            [~,B] = sort(B,2);
            B     = B(:,1:T);
            
            %% Initialize population by LHS
            PopDec     = UniformPoint(NI,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*PopDec+repmat(Problem.lower,NI,1));
            
            %% Initialize surrogate model settings
            numCon   = size(Population.cons,2);
            OTHETA   = 5.*ones(Problem.M,Problem.D);
            OModel   = cell(1,Problem.M);
            Model_lc = cell(lc_num,numCon);
            Model_gc = cell(1,numCon);
            THETA_lc = cell(lc_num);
            for i = 1 : lc_num
                THETA_lc{i} = 5.*ones(numCon,Problem.D);
            end
            THETA_gc = 5.*ones(numCon,Problem.D);  
            Z        = min(Population.objs,[],1);

            %% Calculate Angle Values
            angle      = acos(1-pdist2(W,W,'cosine'));
            temp_angle = angle;
            temp_angle(logical(eye(size(temp_angle)))) = inf;
            theta_min  = min(temp_angle');
            theta_min  = theta_min';
            theta      = theta_min.*0.5;
            TrainArc   = Population;
            A1         = TrainArc;
            stage      = 2;
            
            %% Optimization
            while Algorithm.NotTerminated(TrainArc)
                [~,index] = unique(TrainArc.decs,'rows');
                TrainArc  = TrainArc(index);
                TrainDec  = TrainArc.decs;
                TrainObj  = TrainArc.objs;
                TrainCon  = TrainArc.cons;
                [dsTrainDec,dsTrainObj] = dsmerge(TrainDec,TrainObj);
                for i = 1 : Problem.M
                    dmodel      = dacefit(dsTrainDec,dsTrainObj(:,i),'regpoly0','corrgauss',OTHETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    OModel{i}   = dmodel;
                    OTHETA(i,:) = dmodel.theta;
                end
                if stage == 1
                    %% Objective-oriented optimization stage(Code of RVMM from its authors)
                    % Refresh the model and generate promising solutions
                    A1Dec     = A1.decs;
                    A1Obj     = A1.objs;
                    [c,ia,ic] = unique(A1Obj,'rows','stable');
                    A1Obj     = A1Obj(ia,:);
                    A1Dec     = A1Dec(ia,:);
                    A1        = A1(ia);
	                
	                A1(sum(isnan(A1Obj),2)>0)      = [];
	                A1Dec(sum(isnan(A1Obj),2)>0,:) = [];
	                A1Obj(sum(isnan(A1Obj),2)>0,:) = [];
	                
	                A1(sum(isinf(A1Obj),2)>0)      = [];
	                A1Dec(sum(isinf(A1Obj),2)>0,:) = [];
	                A1Obj(sum(isinf(A1Obj),2)>0,:) = [];
                    
                    [c,ia,ic] = unique(A1Obj,'rows');
                    A1Obj     = A1Obj(ia,:);
                    A1Dec     = A1Dec(ia,:);
                    A1        = A1(ia);
                    [c,ib,ic] = unique(A1Dec,'rows');
                    A1Dec     = A1Dec(ib,:);
	                
                    V0         = Vbb;
                    PopDec1    = A1Dec;
                    w          = 1;
                    A2Obj      = TrainArc.objs;
                    [c,ia,ic]  = unique(A2Obj,'rows','stable');
                    A2Obj      = A2Obj(ia,:);
                    A2Obj      = A2Obj(NDSort(A2Obj,1)==1,:);
                    zmin       = min(A2Obj,[],1);
                    A2Obj_temp = A2Obj - repmat(min(zmin,[],1),size(A2Obj,1),1);
                    if size(A2Obj,1) >= 2
                        scale = (max(A2Obj,[],1)-min(A2Obj,[],1));
                        V_1   = V0.*(max(A2Obj,[],1)-min(A2Obj,[],1));
                    else
                        scale = ones(1,Problem.M);
                        V_1   = V0;
                    end
                    
                    Angle         = acos(1-pdist2(A2Obj_temp,V_1,'cosine'));
                    [~,associate] = min(Angle,[],2);
                    active        = unique(associate,'stable');
                    Va            = V_1(active,:);
                    NCluster      = min(5,size(Va,1));
                    [IDX,C]       = kmeans(V0(active,:),NCluster);
                    V1  = [];
                    ids = [];
                    for i = 1 : NCluster
                        EC      = find(IDX==i);
                        id      = EC(randperm(size(EC,1),1),1);
                        V1(i,:) = Va(id,:);
                        ids     = [ids;id];
                    end
                    if size(V1,1) < 5
                        notS = setdiff(1:size(V_1,1),active(ids));
                        V1   = [V1;V_1( notS(randperm(size(notS,2),5-size(V1,1))),:)];
                    end
                    WPopDec = [];
                    WPopObj = [];
                    WMSE    = [];
                    while w <= wmax
                        drawnow();
                        MatingPool = randi(size(PopDec1,1),1,Problem.N);
                        OffDec     = OperatorGA(Problem,PopDec1(MatingPool,:));
                        pop_candi  = [];
                        NP         = size(OffDec,1);
                        for ii = 1 : NP
                            if min(sqrt(sum((OffDec(ii,:) - [A1Dec;pop_candi]).^2,2)))>1E-6
                                pop_candi = [pop_candi;OffDec(ii,:)];
                            end
                        end
                        OffDec  = pop_candi;
                        PopDec1 = [PopDec1;OffDec];
                        [N,~]   = size(PopDec1);
                        PopObj1 = zeros(N,Problem.M);
                        MSE1    = zeros(N,Problem.M);
                        for i = 1 : N
                            for j = 1 : Problem.M
                                [PopObj1(i,j),~,MSE1(i,j)] = predictor(PopDec1(i,:),OModel{j});
                            end
                        end

                        MSE1 = max(MSE1,0);
                        S_   = sqrt(MSE1);
                        MSE1 = S_.*(MSE1<=1)+MSE1.*(MSE1>1);
		                
                        PopObj1_b = PopObj1;
                        MSE1_b    = MSE1;
                        PopObj1   = PopObj1+kk*MSE1;
                        
                        zmin = min([zmin;PopObj1],[],1);
                 
                        index = KEnvironmentalSelection(PopObj1,[V1;[]],(w/wmax)^alpha);
                        
                        PopDec1 = PopDec1(index,:);
                        PopObj1 = PopObj1(index,:);
                        MSE1    = MSE1(index,:);
                        
                        [~,ib]         = intersect(PopDec1,TrainArc.decs,'rows');
                        PopDec1_       = PopDec1;
                        PopDec1_(ib,:) = [];
                        if isempty(PopDec1_)
                            offobj1     = PopObj1_b(size(PopObj1_b,1)-size(OffDec,1)+1:end,:);
                            offmse1     = MSE1_b(size(PopObj1_b,1)-size(OffDec,1)+1:end,:);
                            [frontNo,~] = NDSort(offobj1,size(offobj1,1));
                            solId       = find(frontNo==1);
                            PopDec1     = [PopDec1; OffDec(solId,:)];
                            PopObj1     = [PopObj1; offobj1(solId,:)+kk*offmse1(solId,:)];
                            MSE1        = [MSE1; offmse1(solId,:)];
                        end
                        % Adapt referece vectors
                        if ~mod(w,ceil(wmax*0.1)) && size(unique(PopObj1,'rows'),1)>2
                            V1    = V1./scale;
                            scale = max(PopObj1,[],1)-min(PopObj1,[],1);
                            V1    = V1.*scale;
                        end
                        w = w + 1;
                    end
                    PopObj1 = PopObj1 - kk*MSE1;
                    
                    [c,ia,ic] = unique(PopObj1,'rows','stable');
                    if ~isempty(ia)
                        PopObj1 = PopObj1(ia,:);
                        PopDec1 = PopDec1(ia,:);
                        MSE1    = MSE1(ia,:);
                    end
                    
                    [~,ib]        = intersect(PopDec1,TrainArc.decs,'rows');
                    PopObj1(ib,:) = [];
                    PopDec1(ib,:) = [];
                    MSE1(ib,:)    = [];
                
                    w      = 1;
                    PopDec = A1Dec;
                    
                    while w <= wmax
                        drawnow();
                       
                        if size(PopDec,1) < Problem.N
                            MatingPool = randi(size(PopDec,1),1,Problem.N);
                            OffDec     = OperatorGA(Problem,PopDec(MatingPool,:));
                        else
                            OffDec = OperatorGA(Problem,PopDec);
                        end
                        
                        pop_candi = [];
                        NP        = size(OffDec,1);
                        for ii = 1 : NP
                            if min(sqrt(sum((OffDec(ii,:)-[A1Dec;pop_candi]).^2,2))) > 1E-6
                                pop_candi = [pop_candi;OffDec(ii,:)];
                            end
                        end
                        OffDec = pop_candi;
                        
                        PopDec = [PopDec;OffDec];
                        [N,~]  = size(PopDec);
                        PopObj = zeros(N,Problem.M);
                        MSE    = zeros(N,Problem.M);
                        for i = 1 : N
                            for j = 1 : Problem.M
                                [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),OModel{j});
                            end
                        end

                        MSE = max(MSE,0);
                        S_  = sqrt(MSE);
                        
                        MSE = S_.*(MSE<=1)+MSE.*(MSE>1);
		                
                        PopObj_b = PopObj;
                        PopDec_b = PopDec;
                        MSE_b    = MSE;
                        
                        PopObj = PopObj+kk*MSE;
                        
                        if w == 1
                            PopObj_ = PopObj(NDSort(PopObj,1)==1,:);
                            scale   = (max(PopObj_,[],1)-min(PopObj_,[],1));
                            scale(:,scale==0) = 10^(-6);
                            V = V0.*scale;
                            
                            Angle         = acos(1-pdist2(PopObj_, V,'cosine'));
                            [~,associate] = min(Angle,[],2);
                            active        = unique(associate,'stable');
                            Va            =  V(active,:);
                            if size(Va,1) < Problem.N
                                PopObj_temp = (PopObj_-min(PopObj_,[],1))./scale;
                                Vadd        = PopObj_temp(randperm(size(PopObj_temp,1),min(Problem.N-size(Va,1),size(PopObj_temp,1))),:);
                                V0          = [V0;Vadd];
                                V           =  V0.*scale;
                            end
                        end
                        
                        index   = KEnvironmentalSelection(PopObj,V,(w/wmax)^alpha);
                        PopDec  = PopDec(index,:);
                        PopObj  = PopObj(index,:);
                        MSE     = MSE(index,:);
                        [~,ib]  = intersect(PopDec,TrainArc.decs,'rows');
                        PopDec_ = PopDec;
                        PopDec_(ib,:) = [];
                        if isempty(PopDec_)
                            offobj      = PopObj_b(size(PopObj_b,1)-size(OffDec,1)+1:end,:);
                            offmse      = MSE_b(size(PopObj_b,1)-size(OffDec,1)+1:end,:);
                            [frontNo,~] = NDSort(offobj,size(offobj,1));
                            solId       = find(frontNo==1);
                            PopDec      = [PopDec; OffDec(solId,:)];
                            PopObj      = [PopObj;offobj(solId,:)+kk*offmse(solId,:)];
                            MSE         = [MSE; offmse(solId,:)];
                        end
                        
                        % Adapt referece vectors
                        if ~mod(w,ceil(wmax*0.1)) && size(unique(PopObj,'rows'),1)>2
                            V = V0.*(max(PopObj,[],1)-min(PopObj,[],1));
                        end
                        w       = w + 1;
                        WPopDec = [WPopDec;PopDec];
                        WPopObj = [WPopObj;PopObj];
                        WMSE    = [WMSE;MSE];            
                    end
	                 
                    PopObj = WPopObj;
                    PopDec = WPopDec;
                    MSE    = WMSE;
                    
                    PopObj = PopObj-kk*MSE;
                    
                    [c,ia,ic] = unique(PopObj,'rows','stable');
                    if ~isempty(ia)
                        PopObj = PopObj(ia,:);
                        PopDec = PopDec(ia,:);
                        MSE    = MSE(ia,:);
                    end
                    
                    [~,ib] = intersect(PopDec,TrainArc.decs,'rows');
                    
                    PopObj(ib,:) = [];
                    PopDec(ib,:) = [];
                    MSE(ib,:)    = [];
                    
                    NumVf  = [];
                    PopNew = KrigingSelect_RVMM(PopDec,PopObj,MSE,V,V0,NumVf,0.05*Problem.N,mu1,(w/wmax)^alpha,Problem.FE./Problem.maxFE,PopDec1,PopObj1,MSE1,V1,TrainArc,A2Obj);
                    
                    if ~isempty(PopNew)
                        [~,ib]       = intersect(PopNew,TrainArc.decs,'rows');
                        PopNew(ib,:) = [];
                        if ~isempty(PopNew)
                            New = Problem.Evaluation(PopNew);
                        else
                            New = [];
                        end
                    else
                        New = [];
                    end
                else
                    % Constraint-oriented optimization stage
                    ArcFZ      = min(TrainArc(all(TrainArc.cons<=0,2)).objs,[],1);
                    [PopDec,~] = CDPEnvironmentalSelection(TrainDec,TrainObj,TrainCon,Problem.N,W,ArcFZ);

                    % clustering
                    Zmin     = min(TrainObj,[],1);   
                    [N,~]    = size(TrainDec);
                    NormObj  = (TrainObj - Zmin);
                    Cluster1 = inf(N,lc_num);
                    Cluster2 = inf(N,lc_num);
                    temp     = N;
                    cluster_size = ceil(N/lc_num);
                    for i = 1 : lc_num
                        [~,ind] = sort(acos(1-pdist2(NormObj,ClW(i,:),'cosine')),"ascend");
                        Cluster1(ind(1:cluster_size),i) = i;
                    end
                    while temp > 0
                        for i = 1 : lc_num
                            [~,loc]         = min(acos(1-pdist2(NormObj,ClW(i,:),'cosine')));
                            Cluster2(loc,i) = i;
                            NormObj(loc,:)  = inf;
                            temp            = temp - 1;
                            if temp == 0
                                break;
                            end
                        end
                    end
                    Cluster = min(Cluster1,Cluster2);

                    % Train local models for constraints
                    for i = 1 : lc_num
                        for j = 1 : numCon
                            X_train_lc       = TrainDec(Cluster(:,i)==i,:); Y_train_lc = TrainCon(Cluster(:,i)==i,j);
                            [X_train_lc, Y_train_lc] = dsmerge(X_train_lc, Y_train_lc);
                            model_lc         = dacefit(X_train_lc,Y_train_lc,'regpoly0','corrgauss',THETA_lc{i}(j,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));                   
                            THETA_lc{i}(j,:) = model_lc.theta;
                            Model_lc{i,j}    = model_lc;
                        end
                    end

                    % Train global model for constraints
                    for i = 1 : numCon
                        [X_train_gc,Y_train_gc] = dsmerge(TrainDec,TrainCon(:,i));
                        model_gc      = dacefit(X_train_gc,Y_train_gc,'regpoly0','corrgauss',THETA_gc(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                        THETA_gc(i,:) = model_gc.theta;
                        Model_gc{i}   = model_gc;
                    end

                    ArcZ   = min(TrainArc.objs,[],1);
                    ArcFZ  = min(TrainArc(all(TrainArc.cons<=0,2)).objs,[],1);
                    [N,~]  = size(PopDec);
                    PopObj = zeros(N,Problem.M);
                    MSE    = zeros(N,Problem.M);
                    for k = 1 : N
                        for j = 1 : Problem.M
                            [PopObj(k,j),~,MSE(k,j)] = predictor(PopDec(k,:),OModel{j});
                        end
                    end
                    PopCon = zeros(N,numCon);
                    CMSE   = zeros(N,numCon);
                    for k = 1 : N
                        for j = 1 : numCon
                            [PopCon(k,j),~,CMSE(k,j)] = predictor(PopDec(k,:),Model_gc{j});
                        end
                    end
                    PopCV = sum(max(0,PopCon),2);
                    w     = 0;
                    pf    = length(find(PopCV==0))/size(PopCV,1);
                    while w < wmax
                        for i = 1 : Problem.N
                            % Choose the parents
                            if rand < 0.9
                                P = B(i,randperm(size(B,2)));
                            else
                                P = randperm(Problem.N);
                            end
                            
                            % Generate an offspring
                            OffDec = OperatorGAhalf(Problem,PopDec(P(1:2),:));
                            for j = 1 : Problem.M
                                [OffObj(:,j),~,OffMSE(:,j)] = predictor(OffDec,OModel{j});
                            end
                            
                            % Update the ideal point
                            Zmin        = min(Zmin,OffObj);
                            Z           = min(Z,OffObj);
                            NormObj     = (OffObj - Zmin);
                            [~,loc]     = min(acos(1-pdist2(NormObj,ClW,'cosine')));
                            OffClus     = loc;
                            AnglePopObj = PopObj(P,:)-repmat(Z,length(P),1);
                            Angle0      = acos(1 - pdist2(real(AnglePopObj),W(P,:),'cosine'));
                            Angle       = diag(Angle0);
                            NewAngle    = acos(1-pdist2(real(OffObj-Z),W(P,:),'cosine'));
                            NewAngle    = NewAngle';
    
                            % Predict each constraint function value
                            for j = 1 : numCon
                                [OffCon_lc(:,j),~,OffMSE_lc(:,j)] = predictor(OffDec,Model_lc{OffClus,j});
                                [OffCon_gc(:,j),~,OffMSE_gc(:,j)] = predictor(OffDec,Model_gc{j});
                            end
                            if mean(OffMSE_gc) < mean(OffMSE_lc)
                                OffCon = OffCon_gc;
                            else
                                OffCon = OffCon_lc;
                            end
                        
                            % Calculate the constraint violation of offspring and P
                            CVO = sum(max(0,OffCon));
                            CVP = sum(max(0,PopCon(P,:)),2);
                            
                            % Calculate PBI value
                            normW   = sqrt(sum(W(P,:).^2,2));
                            normP   = sqrt(sum((PopObj(P,:)-repmat(Z,length(P),1)).^2,2));
                            normO   = sqrt(sum((OffObj-Z).^2,2));
                            CosineP = sum((PopObj(P,:)-repmat(Z,length(P),1)).*W(P,:),2)./normW./normP;
                            CosineO = sum(repmat(OffObj-Z,length(P),1).*W(P,:),2)./normW./normO;
                            g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                            g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
   
                            %% Vector-based Constrained Dominance Principle
                                if CVO+CVP == 0
                                    replace = find(g_old>=g_new,nr);
                                else
                                    if ~isempty(NewAngle<= theta(P) & Angle<= theta(P))
                                        replace = find((g_old>=g_new & CVP==CVO)| CVP>CVO,nr);
                                    else
                                        if rand<pf
                                            replace = find(g_old>=g_new,nr);
                                        else
                                            replace = find(CVP>CVO,nr);
                                        end
                                    end
                                end
                            if ~isempty(replace)
                                PopDec(P(replace),:) = OffDec;
                                PopObj(P(replace),:) = OffObj;
                                PopCon(P(replace),:) = OffCon;
                            end
                        end
                        w = w + 1;
                    end
                    % New solutions for re-evaluation
                    [NewDec,~] = ArchiveUpdate(PopDec,PopObj,PopCon,mu);
                    if ~isempty(NewDec)
                        New = Problem.Evaluation(NewDec);
                    else
                        NewDec = KrigingSelect(PopDec,PopObj,W,mu,(Problem.FE/Problem.maxFE)^2,PopCon);
                        New = Problem.Evaluation(NewDec);
                    end
                end
                TrainArc = [TrainArc,New];
                A1       = TrainArc;

                %% Stage switch
                if Problem.FE < ceil(NI+0.5*(Problem.maxFE-NI))
                    stage = 1;
                else
                    stage = 2;
                end
            end
        end
    end
end
```

### `KEnvironmentalSelection.m`
```matlab
function index = KEnvironmentalSelection(PopObj,V,theta,A2)
% The environmental selection of K-RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    NV    = size(V,1);

    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle         = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);
    
    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current = find(associate==i);
        % Calculate the APD value of each solution
        APD = (1+M*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
        % Select the one with the minimum APD value
        [~,best] = min(APD);
        Next(i)  = current(best);
    end
    % Population for next generation
    index = Next(Next~=0);
end
```

### `KrigingSelect.m`
```matlab
function PopNew = KrigingSelect(PopDec,PopObj,V,mu,theta,PopCon)
% Kriging selection in K-RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin == 5
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    [NVa,va] = NoActive(PopObj,V);
    NCluster = min(mu,size(V,1)-NVa);
    Va       = V(va,:);
    [IDX,~]  = kmeans(Va,NCluster);

    PopObj = PopObj - repmat(min(PopObj,[],1),size(PopObj,1),1);
    cosine = 1 - pdist2(Va,Va,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);
    Angle  = acos(1-pdist2(PopObj,Va,'cosine'));
    [~,associate] = min(Angle,[],2);

    APD_S = ones(size(PopObj,1),1);
    for i = unique(associate)'
        current1 = find(associate==i);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD = (1+size(PopObj,2)*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
            % Select the one with the minimum APD value
            APD_S(current1,:) = APD;
        end
    end

    Cindex = IDX(associate); % Solution to cluster

    Next = zeros(NCluster,1);

    for i = unique(Cindex)'
        solution_Best1 = [];
        solution_Best2 = [];
        t = unique(associate(Cindex==i));
        % Calculate the APD value of each solution
        for j = 1 : size(t,1)
            currentS1 = find(associate==t(j)& CV==0);
            currentS2 = find(associate==t(j)& CV~=0);
            if ~isempty(currentS1)
                [~,id]         = min(APD_S(currentS1,:),[],1);
                solution_Best1 = [solution_Best1;currentS1(id)];
            elseif ~isempty(currentS2)
                [~,id]         = min(CV(currentS2));
                solution_Best2 = [solution_Best2;currentS2(id)];
            end
        end
        if ~isempty(solution_Best1)
            [~,best] = min(APD_S(solution_Best1,:),[],1);
            Next(i)  = solution_Best1(best);
        else
            [~,best] = min(CV(solution_Best2));
            Next(i)  = solution_Best2(best);
        end
    end
    index  = Next(Next~=0);
    PopNew = PopDec(index,:);
end
```

### `KrigingSelect_RVMM.m`
```matlab
function PopNew = KrigingSelect_RVMM(PopDec,PopObj,MSE,V,V0,NumV1,delta,mu_,theta,per,PopDec1,PopObj1,MSE1,V1,A2,A2Obj)
% Kriging selection in K-RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % nondominated sorting
    [c,ia,ic] = unique(PopObj1,'rows','stable');
    PopObj1   = PopObj1(ia,:);
    PopDec1   = PopDec1(ia,:);
    num       = 1;
    
    % flagMu denotes the number of solutions for real function evaluations
    flagMu = mu_;

    [c,ia,ic] = unique(PopObj,'rows','stable');
    PopObj    = PopObj(ia,:);
    PopDec    = PopDec(ia,:);
    Noid      = NDSort(PopObj,1)==1;
    PopObj    = PopObj(Noid,:);
    PopDec    = PopDec(Noid,:);
    
    Noid    = NDSort(PopObj1,1)==1;
    PopObj1 = PopObj1(Noid,:);
    PopDec1 = PopDec1(Noid,:);
    
    wholeobj = [A2Obj;PopObj1;PopObj];
    zmin     = min(wholeobj,[],1);
    Zmin     = min(A2Obj,[],1);
    zmin1    = min(wholeobj,[],1);

    scale = (max(A2Obj,[],1)-min(A2Obj,[],1));
    scale(:,scale==0) = 10^(-6);
    cd    = [];
    for i = 1 : size(PopObj1,1)
        mu    = PopObj1(i,:);
        R1{i} = mu;
        zmin1 = min([zmin1;R1{i}],[],1);
    end
    for i = 1 : size(PopObj1,1)
        d = pdist2((R1{i}-zmin1),(A2Obj - zmin1),'euclidean');
        A = R1{i};
        [cd_,id] = sort(d,2);
        id_ = id(:,1);
        k   = any(A<A2Obj(id_,:),2) - any(A>A2Obj(id_,:),2);
        cd_(k~=1,1) = 0;
        cd(i,:)     = sum(cd_(:,1));
    end
    if flagMu == 1
        [~,cbest] = max(cd,[],1);
        cid       = find(cd == cd(cbest,:));
        try
            cbest = cid(randperm(size(cid,1),1),:);
        catch e
            cbest = [];
        end
    elseif flagMu == 5
        [~,id] = sort(cd,1,'descend');
        cbest  = id(1:min(flagMu,size(cd,1)));
    end
    
    % diversity
    dbest = [];
    zmin  = min([zmin;zmin1],[],1);
    if ~isempty(PopObj)
        for i = 1 : size(PopObj,1)
            mu   = PopObj(i,:);
            R{i} = repmat(mu,num,1);
            zmin = min([zmin;R{i}],[],1);
        end
        allR = cell2mat(R(1:end)');
        if RVMM_IGD((max(allR-zmin,0))./scale,max( A2Obj- zmin,0)./scale) < RVMM_IGD((max(allR-zmin,0))./scale,max(A2Obj - Zmin,0)./scale)
            Angle = acos(1-pdist2((max(allR-zmin,0))./scale,max(A2Obj - zmin,0)./scale,'cosine'));
        else
            Angle = acos(1-pdist2((max(allR-zmin,0))./scale,max(A2Obj - Zmin,0)./scale,'cosine'));
        end
        [angle,~] = min(Angle,[],2);
        temp      = reshape(angle,num,length(R));
        dd        = mean(temp,1);
    
        dd = dd';
        dd = dd./max(dd);
    
        if flagMu == 1
            [~,dbest] = max(dd,[],1);
    
        elseif flagMu == 5
            [~,id] = sort(dd,1,'descend');
            dbest  = id(1:min(flagMu,size(dd,1)));
        end
    end

    [frontNo,~] = NDSort([PopObj1(cbest,:);A2Obj],size(A2Obj,1));
    if ~isempty(frontNo(2:end)==2)&&frontNo(1)==1
        PopNew = PopDec1(cbest,:);
    else
        PopNew = PopDec(dbest,:);
    end
    if isempty(PopObj) && isempty(PopObj1)
        PopNew = [];
    end
end
```

### `RVMM_IGD.m`
```matlab
function score = RVMM_IGD(PopObj,optimum)
% Inverted generational distance

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if size(PopObj,2) ~= size(optimum,2)
        score = nan;
    else
        score = mean(min(pdist2(optimum,PopObj),[],2));
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
