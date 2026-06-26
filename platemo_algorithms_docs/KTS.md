# KTS

**Tags**: <2024> <multi/many> <real> <expensive> <constrained>

## Description
Kriging-assisted evolutionary algorithm with two search modes

## Reference
Z. Song, H. Wang, B. Xue, M. Zhang, and Y. Jin. Balancing objective optimization and constraint satisfaction in expensive constrained evolutionary multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2024, 28(5): 1286-1300.

## Source Code

### `Adaptive_sampling.m`
```matlab
function Offspring01 = Adaptive_sampling(CAobj,DAobj,CAdec,DAdec,DAvar,DA,CA1,mu,p,phi)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    Ideal_Point = min([CAobj;DAobj],[],1);

    if  size(DAdec,1) <= mu
        flag = 1;
    else
        flag = Cal_Convergence(CAobj,DAobj,Ideal_Point);
    end

    if flag == 1
        % convergence sampling strategy
        N       = size(CAobj,1);
        CAobj01 = (CAobj-repmat(min(CAobj),N,1))./(repmat(max(CAobj)-min(CAobj),N,1));
        I = zeros(N);
        for i = 1 : N
            for j = 1 : N
                I(i,j) = max(CAobj01(i,:)-CAobj01(j,:));
            end
        end
        C      = max(abs(I));
        F      = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
        Choose = 1 : N;
        while length(Choose) > mu
            [~,x]     = min(F(Choose));
            F         = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
            Choose(x) = [];
        end
        Offspring01 = CAdec(Choose,:);
    else
        if size(DAdec,1) <= mu
            Offspring01 = DAdec;
        else
            if PD(DAobj) < PD(DA.objs)
                % uncertainty sampling strategy
                An     = size(DAvar,1);
                Choose = zeros(1,5);
                for i = 1 : mu
                    A_num       = randperm(size(DAvar,1));
                    Uncertainty = mean(DAvar(A_num(1:ceil(phi*An)),1:size(DAobj,2)),2);
                    [~,best]    = max(Uncertainty);
                    Choose (i)  = A_num(best);
                end
                Offspring01 = DAdec(Choose ,:);
            else
                % diversity sampling strategy
                DA_Nor = (DA.objs - repmat(min([DAobj;DA.objs],[],1),length(DA),1))...
                    ./repmat(max([DAobj;DA.objs],[],1) - min([DAobj;DA.objs],[],1),length(DA),1);
                DA_Nor_pre = (DAobj - repmat(min([DAobj;DA.objs],[],1),size(DAobj,1),1))...
                    ./repmat(max([DAobj;DA.objs],[],1) - min([DAobj;DA.objs],[],1),size(DAobj,1),1);
                N       = size(DA_Nor,1);
                Pop     = [DA_Nor;DA_Nor_pre];
                Pop_dec = [DA.decs;DAdec];
                NN      = size(Pop,1);
                Choose  = false(1,NN);
                Choose(1:N) = true;
                MaxSize     = N+mu;
                Distance    = inf(N);
                for i = 1 : NN-1
                    for j = i+1 : NN
                        Distance(i,j) = norm(Pop(i,:)-Pop(j,:),p);
                        Distance(j,i) = Distance(i,j);
                    end
                end
                Offspring01 = [];
                while sum(Choose) < MaxSize
                    Remain = find(~Choose);
                    [~,x]  = max(min(Distance(~Choose,Choose),[],2));
                    Choose(Remain(x)) = true;
                    Offspring01 = [Offspring01;Pop_dec(Remain(x),:)];
                end
            end
        end
    end
end

function Score = PD(PopObj)
% Pure diversity

    N = size(PopObj,1);
    C = false(N);
    C(logical(eye(size(C)))) = true;
    D = pdist2(PopObj,PopObj,'minkowski',0.1);
    D(logical(eye(size(D)))) = inf;
    Score = 0;
    for k = 1 : N-1
        while true
            [d,J] = min(D,[],2);
            [~,i] = max(d);
            if D(J(i),i) ~= -inf
                D(J(i),i) = inf;
            end
            if D(i,J(i)) ~= -inf
                D(i,J(i)) = inf;
            end
            P = any(C(i,:),1);
            while ~P(J(i))
                newP = any(C(P,:),1);
                if P == newP
                    break;
                else
                    P = newP;
                end
            end
            if ~P(J(i))
                break;
            end
        end
        C(i,J(i)) = true;
        C(J(i),i) = true;
        D(i,:)    = -inf;
        Score     = Score + d(i);
    end
end
```

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

### `Cal_Convergence.m`
```matlab
function flag = Cal_Convergence(PopObj1,PopObj2,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    N1 = size(PopObj1,1);
    N2 = size(PopObj2,1);
    if N1 ~= N2
        flag = 0;
    else
        PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
        PopObj = (PopObj)./repmat(max(PopObj,[],1) - Zmin,size(PopObj,1),1);
        Distance1 = zeros(1,N1);
        Distance2 = zeros(1,N2);
        % calculate the distance sets of CCA and CDA
        for i = 1:N1
            Distance1(i) = sqrt(sum(PopObj(i,:),2));
        end
        for i = 1: N2
            Distance2(i) = sqrt(sum(PopObj(N1+i,:),2));
        end
        % rank-sum test, alpha = 0.05
        [~,flag,~,r1,r2]=signrank_new(Distance1, Distance2,'alpha',0.05);
        if flag == 1 && (r1-r2 <0)
            flag = 0;
        end
    end
end
```

### `KCCMO_sampling.m`
```matlab
function [Population,Fitness1] = KCCMO_sampling(Population,CA1,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CA = CA1.best.objs;
    if isempty(CA)
        CA = CA1.objs;
    end

    Fitness = CalFitness_new(Population.obj,Population.con,CA);

    NCluster = N;
    [IDX,~]  = kmeans(Population.obj,NCluster);
    Next     = false(size(Population.obj,1),1);
    for i = 1 : NCluster
        select    = find(IDX == i);
        Fitness1  = Fitness(select);
        [~,index] = min(Fitness1);
        Next(select(index)) = true;
    end
    Population = givevalue(Population,Next);
end

function Fitness = CalFitness_new(PopObj,PopCon,CCA)

    N = size(PopObj,1);
    if isempty(PopCon)
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

    S = sum(Dominate,2);

    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

    Distance  = pdist2(PopObj,CCA);
    Distance1 = min(Distance,[],2);
    D         = 1./(Distance1+2);
    Fitness   = R + D';
end
```

### `KTS.m`
```matlab
classdef KTS < ALGORITHM
% <2024> <multi/many> <real> <expensive> <constrained>
% Kriging-assisted evolutionary algorithm with two search modes
% tau --- 0.6 --- Threshold value
% phi --- 0.2 --- Threshold value
% mu  ---  20 --- Number of elite solution in A1

%------------------------------- Reference --------------------------------
% Z. Song, H. Wang, B. Xue, M. Zhang, and Y. Jin. Balancing objective
% optimization and constraint satisfaction in expensive constrained
% evolutionary multi-objective optimization. IEEE Transactions on
% Evolutionary Computation, 2024, 28(5): 1286-1300.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song
% Email:zssong@stu.xidian.edu.cn
    
    methods
        function main(Algorithm,Problem)
            %% Parameter setting on KTA2
            phi1  = 0.1;
            wmax1 = 10;
            mu1   = 5;
            %% Parameter setting on KTS
            [tau,phi,mu] = Algorithm.ParameterSet(0.6,0.2,20);
            phi = -phi;
            
            %% Initialization
            p      = 1/Problem.M;
            CAsize = Problem.N;
            N      = Problem.N;
            P      = UniformPoint(N, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*P+repmat(Problem.lower,N,1));
            A1     = Population;
            
            % CA DA in KTA2
            CA = UpdateCA([],Population,CAsize);
            DA1 = UpdateDA(Population,[],Problem.N,p);
            
            % P1 P2 in KCCMO
            P1 = Population;
            P2 = Population;
            DA = DA1;
            
            % build surrogate models for objective functions
            Dec = [DA.decs;CA.decs;P2.decs];
            Obj = [DA.cons;CA.cons;P2.cons];
            [~,index] = unique( round(Dec*1e4)/1e4,'rows');
            Dec   =  Dec(index,:);
            Obj   = Obj(index,:);
            THETA = 5.*ones((Problem.M + size(DA.cons,2)),Problem.D);
            Model = cell(1,(Problem.M + size(DA.cons,2)));
            % build surrogate models for constraint functions
            for i = Problem.M+1: (Problem.M + size(DA.cons,2))
                j = i - Problem.M;
                dmodel     = dacefit(Dec,Obj(:,j),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                Model{i}   = dmodel;
                THETA(i,:) = dmodel.theta;
            end
            
            while Algorithm.NotTerminated(A1)
                Q          = Cal_Q(A1.objs); % calculate the convergence contribution of all evaluated solutions
                [Q, index] = sort(Q,'descend');
                CV         = sum(max(A1.cons,0),2); % calculate the CV of all evaluated solutions
                CV         = CV(index);
                coef = corrcoef(Q(end-mu:end), CV(end-mu:end)); % calculate the correlation coefficient
                r_coef = coef(2);
                % adaptive switching
                if r_coef < phi
                    search_mode = 1;   %  Constrained surrogate-assisted evolutionary search
                else
                    if r_coef <  tau
                        if rand < 0.5
                            search_mode = 0; %  Unconstrained surrogate-assisted evolutionary search
                        else
                            search_mode = 1;
                        end
                    else
                        search_mode = 0;
                    end
                end
                
                
                if search_mode == 0
                    DA = DA1;
                else
                    DA = P1;
                end
                
                % build surrogate models for objectives
                Dec = A1.decs;
                Obj = A1.objs;
                for i = 1:(Problem.M )%+ size(DA.cons,2))
                    dmodel     = dacefit(Dec,Obj(:,i),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                    Model{i}   = dmodel;
                    THETA(i,:) = dmodel.theta;
                end
                
                if search_mode == 1
                    % build surrogate models for constraints
                    Dec = [DA.decs;CA.decs;P2.decs];
                    Obj = [DA.cons;CA.cons;P2.cons];
                    [~,index] = unique( round(Dec,-4) ,'rows');
                    Dec =  Dec(index,:);
                    Obj = Obj(index,:);
                    for i = Problem.M+1: (Problem.M + size(DA.cons,2))
                        j = i - Problem.M;
                        dmodel     = dacefit(Dec,Obj(:,j),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Problem.D),100.*ones(1,Problem.D));
                        Model{i}   = dmodel;
                        THETA(i,:) = dmodel.theta;
                    end
                end
                
                % Set the CCA and CDA as the current CA and DA
                CCA.obj = CA.objs;
                CCA.dec = CA.decs;
                CCA.con = CA.cons;
                CCA.MSE = zeros(size(CCA.con,1),Problem.M+size(CCA.con,2));
                CP2.obj = P2.objs;
                CP2.dec = P2.decs;
                CP2.con = P2.cons;
                CP2.MSE = zeros(size(CP2.con,1),Problem.M+size(CP2.con,2));
                CDA.obj = DA.objs;
                CDA.dec = DA.decs;
                CDA.con = DA.cons;
                CDA.MSE = zeros(size(CDA.con,1),Problem.M+size(CDA.con,2));
                w       = 1;
                while w <= wmax1
                    if search_mode == 0
                        % Solution generation in KTA2
                        [~,ParentCdec,~,ParentMdec] = MatingSelection_KTA2(CCA.obj,CCA.dec,CDA.obj,CDA.dec,Problem.N);
                        OffspringDec = [OperatorGA(Problem,ParentCdec,{1,20,0,0});OperatorGA(Problem,ParentMdec,{0,0,1,20})];
                    else
                        % Solution generation in KCCMO
                        Fitness1 = CalFitness(CP2.obj,CP2.con);
                        Fitness2 = CalFitness(CDA.obj);
                        %
                        MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                        MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                        %
                        Offspring1  = OperatorGA(Problem,CP2.dec(MatingPool1,:));
                        Offspring2  = OperatorGA(Problem,CDA.dec(MatingPool2,:));
                        OffspringDec = [Offspring1;Offspring2];
                    end
                    
                    Pop.dec = OffspringDec;
                    N       = size(Pop.dec,1);
                    Pop.obj = zeros(N,Problem.M);
                    Pop.con = zeros(N,size(DA.cons,2));
                    Pop.MSE = zeros(N,Problem.M+ size(DA.cons,2));
                    PopObj  = zeros(N,Problem.M+ size(DA.cons,2));
                    
                    for i = 1 : N
                        for j = 1 : (Problem.M+size(DA.cons,2))
                            [PopObj(i,j),~,Pop.MSE(i,j)] = predictor(Pop.dec(i,:),Model{j});
                        end
                    end
                    
                    Pop.obj = PopObj(:,1:Problem.M );
                    Pop.con = PopObj(:,Problem.M+1 :end );
                    
                    PopC = cat_struct(CCA,Pop);
                    CCA  = K_UpdateCA(PopC,CAsize);
                    PopD = cat_struct(CDA,Pop);
                    
                    if search_mode == 0
                        CDA = K_UpdateDA(PopD,Problem.N,p);
                    else
                        CDA = K_UpdateP(PopD,Problem.N,false);
                    end
                    
                    PopC1   = cat_struct(CP2,Pop);
                    [CP2,~] = K_UpdateP(PopC1,Problem.N,true);
                    w = w + 1;
                end
                
                % Adaptive sampling
                if search_mode == 0
                    % KTA2
                    % remove the same solution in all_population
                    [~,ia,~] = setxor(CCA.dec,A1.decs,'rows');
                    CCA      = givevalue(CCA,ia);
                    [~,ia,~] = setxor(CDA.dec,A1.decs,'rows');
                    CDA      = givevalue(CDA,ia);
                    
                    Offspring01 = Adaptive_sampling(CCA.obj,CDA.obj,CCA.dec,CDA.dec,CDA.MSE,DA,P2,mu1,p,phi1);
                else
                    % KCCMO
                    [CCA2,~]    = KCCMO_sampling(CP2,P2,mu1);
                    Offspring01 = CCA2.dec;
                end
                
                [~,index]   = unique(Offspring01 ,'rows');
                PopNew      = Offspring01(index,:);
                Offspring02 = PopNew;
                
                if ~isempty(Offspring02)
                    Offspring = Problem.Evaluation(Offspring02);
                    temp      = A1.decs;
                    for i = 1 : size(Offspring,2)
                        dist2 = pdist2(Offspring(i).decs,temp);
                        if min(dist2) > 1e-5
                            A1 = [A1,Offspring(i)];
                        end
                        temp = A1.decs;
                    end
                    
                    % data selection is the same as enviromental selection
                    CA     = UpdateCA(CA,Offspring,CAsize);
                    DA1    = UpdateDA(DA1,Offspring,Problem.N,p);
                    [P1,~] = Update_P([P1,Offspring],Problem.N,false);
                    [P2,~] = Update_P([P2,Offspring],Problem.N,true);
                end
            end
        end
    end
end

function value = Cal_Q(Obj)
    N   = size(Obj,1);
    Obj = (Obj-repmat(min(Obj),N,1))./(repmat(max(Obj)-min(Obj),N,1));
    I   = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(Obj(i,:)-Obj(j,:));
        end
    end
    C     = max(abs(I));
    F     = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
    value = 1./F;
end
```

### `K_UpdateCA.m`
```matlab
function[CCA] = K_UpdateCA(CA,MaxSize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    N  = size(CA.obj,1);
    if N <= MaxSize
        return;
    end
    
    %% Calculate the fitness of each solution
    CAobj1 = (CA.obj-repmat(min(CA.obj),N,1))./(repmat(max(CA.obj)-min(CA.obj),N,1));
    I = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(CAobj1(i,:)-CAobj1(j,:));
        end
    end
    C = max(abs(I));
    F = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
    
    %% Delete part of the solutions by their fitnesses
    Choose = 1 : N;
    while length(Choose) > MaxSize
        [~,x] = min(F(Choose));
        F = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
        Choose(x) = [];
    end
    CCA = givevalue(CA,Choose);
end
```

### `K_UpdateDA.m`
```matlab
function DA = K_UpdateDA(DA,MaxSize,p)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    DA_Nor_pre = (DA.obj - repmat(min(DA.obj,[],1),size(DA.obj,1),1))./repmat(max(DA.obj,[],1) - min(DA.obj,[],1),size(DA.obj,1),1);

    %% Find the non-dominated solutions
    ND = NDSort(DA.obj,1);
    DA = givevalue(DA,(ND==1));

    DA_Nor_pre = DA_Nor_pre((ND==1),:);
    N  = size(DA.obj,1);
    if N <= MaxSize
        return;
    end
    
    %% Select the extreme solutions first
    Choose = false(1,N);
    M      = size(DA_Nor_pre,2);
    select = randperm(M);
    Choose(select(1)) = true;

    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > MaxSize
        % Randomly delete several solutions
        Choosed = find(Choose);
        k       = randperm(sum(Choose),sum(Choose)-MaxSize);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < MaxSize 
        Distance = inf(N);
        for i = 1 : N-1
            for j = i+1 : N
                Distance(i,j) = norm(DA_Nor_pre(i,:)-DA_Nor_pre(j,:),p);
                Distance(j,i) = Distance(i,j);
            end
        end
        while sum(Choose) < MaxSize
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
    end
    DA = givevalue(DA,Choose);
end
```

### `K_UpdateP.m`
```matlab
function [Population,Fitness] = K_UpdateP(Population,N,isOrigin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    if isOrigin
        Fitness = CalFitness(Population.obj,Population.con);
    else
        Fitness = CalFitness(Population.obj);
    end

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population.obj(Next,:),sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    Population = givevalue(Population,Next);
    Fitness    = Fitness(Next);
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

### `MatingSelection_KTA2.m`
```matlab
function [ParentCobj,ParentCdec,ParentMobj,ParentMdec] = MatingSelection_KTA2(CAobj,CAdec,DAobj,DAdec,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhenshou Song

    CAParent1  = randi(size(CAobj,1),1,ceil(N/2));
    CAParent2  = randi(size(CAobj,1),1,ceil(N/2));
    Dominate   = any(CAobj(CAParent1,:)<CAobj(CAParent2,:),2) - any(CAobj(CAParent1,:)>CAobj(CAParent2,:),2);  
    ParentCobj = [CAobj([CAParent1(Dominate==1),CAParent2(Dominate~=1)],:);...
                 DAobj(randi(size(DAobj,1),1,ceil(N/2)),:)];
    ParentCdec = [CAdec([CAParent1(Dominate==1),CAParent2(Dominate~=1)],:);...
                 DAdec(randi(size(DAdec,1),1,ceil(N/2)),:)];
    ParentMobj = CAobj(randi(size(CAobj,1),1,N),:);
    ParentMdec = CAdec(randi(size(CAdec,1),1,N),:);
end
```

### `UpdateCA.m`
```matlab
function CA = UpdateCA(CA,New,MaxSize)
% Update CA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CA = [CA,New];
    N  = length(CA);
    if N <= MaxSize
        return;
    end
    
    %% Calculate the fitness of each solution
    CAObj = CA.objs;
    CAObj = (CAObj-repmat(min(CAObj),N,1))./(repmat(max(CAObj)-min(CAObj),N,1));
    I = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(CAObj(i,:)-CAObj(j,:));
        end
    end
    C = max(abs(I));
    F = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
    
    %% Delete part of the solutions by their fitnesses
    Choose = 1 : N;
    while length(Choose) > MaxSize
        [~,x]     = min(F(Choose));
        F         = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
        Choose(x) = [];
    end
    CA = CA(Choose);
end
```

### `UpdateDA.m`
```matlab
function DA = UpdateDA(DA,New,MaxSize,p)
% Update DA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Find the non-dominated solutions
    DA = [DA,New];
    ND = NDSort(DA.objs,1);
    DA = DA(ND==1);
    N  = length(DA);
    if N <= MaxSize
        return;
    end
    
    %% Select the extreme solutions first
    Choose = false(1,N);
    [~,Extreme1] = min(DA.objs,[],1);
    [~,Extreme2] = max(DA.objs,[],1);
    Choose(Extreme1) = true;
    Choose(Extreme2) = true;
    
    %% Delete or add solutions to make a total of K solutions be chosen by truncation
    if sum(Choose) > MaxSize
        % Randomly delete several solutions
        Choosed = find(Choose);
        k = randperm(sum(Choose),sum(Choose)-MaxSize);
        Choose(Choosed(k)) = false;
    elseif sum(Choose) < MaxSize
        % Add several solutions by truncation strategy
        Distance = inf(N);
        for i = 1 : N-1
            for j = i+1 : N
                Distance(i,j) = norm(DA(i).obj-DA(j).obj,p);
                Distance(j,i) = Distance(i,j);
            end
        end
        while sum(Choose) < MaxSize
            Remain = find(~Choose);
            [~,x]  = max(min(Distance(~Choose,Choose),[],2));
            Choose(Remain(x)) = true;
        end
    end
    DA = DA(Choose);
end
```

### `Update_P.m`
```matlab
function [Population,Fitness] = Update_P(Population,N,isOrigin)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    if isOrigin
        Fitness = CalFitness(Population.objs,Population.cons);
    else
        Fitness = CalFitness(Population.objs);
    end

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population = Population(rank);
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

### `cat_struct.m`
```matlab
function C = cat_struct(A, B)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(A)
        C = B;
    else
        C = struct();
        for f = fieldnames(A)'
            C.(f{1}) = [A.(f{1}); B.(f{1})];
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

### `givevalue.m`
```matlab
function Population = givevalue(Population, N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for f = fieldnames(Population)'
        Population.(f{1}) = Population.(f{1})(N,:);
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
