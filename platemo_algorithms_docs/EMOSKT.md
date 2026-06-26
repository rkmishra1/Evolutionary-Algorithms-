# EMOSKT

**Tags**: <2025> <multi> <real/binary> <large/none> <constrained/none> <sparse> <multitask>

## Description
Evolutionary multi-objective optimization with sparsity knowledge transfer

## Reference
C. Wu, Y. Tian, L. Zhang, X. Xiang, and X. Zhang. A sparsity knowledge transfer-based evolutionary algorithm for large-scale multitasking multi- objective optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2582-2595.

## Source Code

### `EMOSKT.m`
```matlab
classdef EMOSKT < ALGORITHM
% <2025> <multi> <real/binary> <large/none> <constrained/none> <sparse> <multitask>
% Evolutionary multi-objective optimization with sparsity knowledge transfer

%------------------------------- Reference --------------------------------
% C. Wu, Y. Tian, L. Zhang, X. Xiang, and X. Zhang. A sparsity knowledge
% transfer-based evolutionary algorithm for large-scale multitasking multi-
% objective optimization. IEEE Transactions on Evolutionary Computation,
% 2025, 29(6): 2582-2595.
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
            %% Population initialization
            TaskNum    = size(Problem.SubM,2);
            EachN      = ceil(Problem.N/TaskNum);
            [~, Maxid] = max(Problem.SubD);
            Buquan     = zeros(1,TaskNum);
            for i = 1 : TaskNum
                Buquan(i) = Problem.SubD(Maxid)-Problem.SubD(i);
            end
            % Calculate the fitness of each decision variable
            REAL    = all(Problem.encoding~=4);
            TDec    = cell(1+4*REAL,TaskNum);
            TMask   = cell(1+4*REAL,TaskNum);
            TempPop = cell(1+4*REAL,TaskNum);
            Fitness = cell(1,TaskNum);
            for j = 1:TaskNum
                Fitness{j} = zeros(1,Problem.SubD(j));
            end
            FitnessDec = cell(1,TaskNum);
            for j = 1 : TaskNum
                FitnessDec{j} = zeros(1+4*REAL,Problem.SubD(j));
            end
            for i = 1 : 1+4*REAL
                for j = 1 : TaskNum
                    if REAL
                        Dec{i,j} = unifrnd(repmat(Problem.lower(1:Problem.SubD(j))+(Problem.upper(1:Problem.SubD(j))-Problem.lower(1:Problem.SubD(j)))*((i-1)/(1+4*REAL)),Problem.SubD(j),1),...
                                           repmat(Problem.lower(1:Problem.SubD(j))+(Problem.upper(1:Problem.SubD(j))-Problem.lower(1:Problem.SubD(j)))*((i)/(1+4*REAL)),Problem.SubD(j),1));
                    else
                        Dec{i,j} = ones(Problem.SubD(j),Problem.SubD(j));
                    end
                    Mask{i,j}     = eye(Problem.SubD(j));
                    Skill{i,j}    = j*ones(Problem.SubD(j),1);
                    Solution{i,j} = [Dec{i,j}.*Mask{i,j},zeros(size(Dec{i,j},1),Buquan(j)),Skill{i,j}];
                    Initpop{i,j}  = Problem.Evaluation(Solution{i,j});
                    TDec{i,j}     = [TDec{i,j};Dec{i,j}];
                    TMask{i,j}    = [TMask{i,j};Mask{i,j}];
                    TempPop{i,j}  = [TempPop{i,j},Initpop{i,j}];
                    Fitness{j}    = Fitness{j} + NDSort([Initpop{i,j}.objs,Initpop{i,j}.cons],inf);
                    FitnessDec{j}(i,:) = NDSort([Initpop{i,j}.objs,Initpop{i,j}.cons],inf);
                end
            end
            FitnessPop = TempPop;
            for j = 1 : TaskNum
                for i = 1 : Problem.SubD(j)
                    pDecIndex{j,i} = find(FitnessDec{j}(:,i)==min(FitnessDec{j}(:,i)));
                end
            end
            % Generate initial population
            Dec  = cell(1,TaskNum);
            Mask = cell(1,TaskNum);
            Pop  = {};
            SubPopulation = {};
            FrontNo       = {};
            CrowdDis      = {};
            Skill         = {};
            Solution      = {};
            for i = 1 : TaskNum
                if REAL
                    for n = 1 : EachN
                        for d = 1 : Problem.SubD(i)
                            pDecRandIndex = pDecIndex{i,d}(randi(size(pDecIndex{i,d},1)));
                            Dec{i}(n,d)   = unifrnd(Problem.lower(d)+(Problem.upper(d)-Problem.lower(d))*((pDecRandIndex-1)/(1+4*REAL)),...
                                                    Problem.lower(d)+(Problem.upper(d)-Problem.lower(d))*((pDecRandIndex)/(1+4*REAL)));
                        end
                    end
                else
                    Dec{i} = ones(EachN,Problem.SubD(i));
                end
                Mask{i} = zeros(EachN,Problem.SubD(i));
                SamMask   = Mask{i}(1:5,:);
                [~,rank1] = sort(Fitness{i});
                index     = round([0.1,0.2,0.3,0.4,0.5]*Problem.SubD(i));
                for j = 1 : 5
                    SamMask(j,rank1(1:index(j))) = 1;
                end
                Mask{i}(1:5,:) = SamMask;
                for j = 6 : EachN
                    Mask{i}(j,TournamentSelection(2,ceil(rand*Problem.SubD(i)),Fitness{i})) = 1;
                end             
                Skill{i}    = i*ones(EachN,1);
                Solution{i} = [Dec{i}.*Mask{i},zeros(size(Dec{i},1),Buquan(i)),Skill{i}];
                Pop{i}      = Problem.Evaluation(Solution{i});
            end
            % Generate initthetamid
            initthetamid = zeros(1,TaskNum);
            for i = 1 : TaskNum
                [~,~,~,TFrontNo,~] = SparseEA_ESnouni([Pop{i},[TempPop{:,i}]],[Dec{i};vertcat(TDec{:,i})],[Mask{i};vertcat(TMask{:,i})],length([Pop{i},[TempPop{:,i}]]));
                [SubPopulation{i},Dec{i},Mask{i},FrontNo{i},CrowdDis{i}] = SparseEA_EnvironmentalSelection([Pop{i},[TempPop{:,i}]],[Dec{i};vertcat(TDec{:,i})],[Mask{i};vertcat(TMask{:,i})],EachN);
                if size(find(TFrontNo(1:5)==1),2)>0
                    initthetamid(i) = mean(index(TFrontNo(1:5)==1));
                else
                    Theta = sum(Mask{i}(FrontNo{i} ==1,:),2)';
                    initthetamid(i) = mean(Theta);
                end
            end

            %% Optimization
            NumTransUp        = floor(EachN/10)*ones(1,TaskNum);
            ALLTthetamid      = [];
            ALLTthetamid(1,:) = initthetamid;
            [SourceId,TF1]    = SourceTaskrand(Problem,SubPopulation,Dec,Mask,FrontNo,CrowdDis,EachN,Fitness,FitnessDec,pDecIndex,Buquan);
            while Algorithm.NotTerminated([SubPopulation{:}])
                for i = 1 : TaskNum
                    [SubPopulation{i},Dec{i},Mask{i},FrontNo{i},CrowdDis{i}] = OP_SparseEA(i,EachN,Problem,FrontNo{i},CrowdDis{i},Dec{i},Mask{i},Fitness{i},SubPopulation{i},Buquan(i),REAL);
                end
                [SubPopulation,Dec,Mask,FrontNo,CrowdDis,NumTransUp,ALLTthetamid] = Op_Transnodec(Problem,SubPopulation,Dec,Mask,FrontNo,CrowdDis,EachN,TF1,FitnessDec,pDecIndex,FitnessPop,SourceId,Buquan,NumTransUp,ALLTthetamid);
                [SourceId,TF1] = SourceTaskrand(Problem,SubPopulation,Dec,Mask,FrontNo,CrowdDis,EachN,Fitness,FitnessDec,pDecIndex,Buquan);
            end
        end
    end
end
```

### `MOEAPSL_EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis,sRatio] = MOEAPSL_EnvironmentalSelection(Population,Dec,Mask,N,num)
% The environmental selection of MOEA/PSL

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions
    [~,uni] = unique(Population.objs,'rows');
    if isscalar(uni)
        [~,uni] = unique(Population.decs,'rows');
    end
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;    
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Calculate the ratio of successful offsprings
    s1     = sum(Next(N+1:end));
    s2     = num;
    sRatio = (s1+1e-6)./(s2+1e-6);
    sRatio = min(max(sRatio,0),1);
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `OP_SparseEA.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis]=OP_SparseEA(Taskid,EachN,Problem,FrontNo,CrowdDis,Dec,Mask,Fitness,Population,Buquan,REAL)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    MatingPool       = TournamentSelection(2,2*EachN,FrontNo,-CrowdDis);
    [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),Fitness,Taskid,REAL);
                    
    Skill     = Taskid*ones(size(OffDec,1),1);
    Solution  = [OffDec.*OffMask,zeros(size(OffDec,1),Buquan),Skill];
    Offspring = Problem.Evaluation(Solution);

    [Population,Dec,Mask,FrontNo,CrowdDis] = SparseEA_EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],EachN);
end

function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Fitness,Taskid,REAL)
    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        if rand < 0.5
            index = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
            index = index(TS(-Fitness(index)));
            OffMask(i,index) = 0;
        else
            index = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
            index = index(TS(Fitness(index)));
            OffMask(i,index) = Parent2Mask(i,index);
        end
    end
    
    %% Mutation for mask
    for i = 1 : N/2
        if rand < 0.5
            index = find(OffMask(i,:));
            index = index(TS(-Fitness(index)));
            OffMask(i,index) = 0;
        else
            index = find(~OffMask(i,:));
            index = index(TS(Fitness(index)));
            OffMask(i,index) = 1;
        end
    end
    
    %% Crossover and mutation for dec
    if REAL
        OffDec = OperatorGAhalf(Problem,ParentDec,Taskid);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end    
     
end

function index = TS(Fitness)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2,1,Fitness);
    end
end

function Offspring = OperatorGAhalf(Problem,Parent,Taskid)
    [proC,disC,proM,disM] = deal(1,20,1,20);

    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        Parent    = Parent.decs;
    else
        evaluated = false;
    end
    Parent1   = Parent(1:floor(end/2),:);
    Parent2   = Parent(floor(end/2)+1:floor(end/2)*2,:);
    Offspring = GAreal(Parent1,Parent2,Problem.lower(1:Problem.SubD(Taskid)),Problem.upper(1:Problem.SubD(Taskid)),proC,disC,proM,disM);
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end

function Offspring = GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM)
% Genetic operators for real and integer variables

    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
             
    %% Polynomial mutation
    Lower = repmat(lower,N,1);
    Upper = repmat(upper,N,1);
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

### `Op_Transnodec.m`
```matlab
function [SubPopulation,Dec,Mask,FrontNo,CrowdDis,NumTransUp,ALLTthetamid,MeanMaskSourceId]= ...
    Op_Transnodec(Problem,SubPopulation,Dec,Mask,FrontNo,CrowdDis,EachN,TF1,FitnessDec,pDecIndex,FitnessPop,SourceId,Buquan,NumTransUp,ALLTthetamid)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    TaskNum    = size(Problem.SubM,2);
    NSpop      = {};
    NSdec      = {};
    NSmask     = {};
    NSCrowdDis = {};
    MeanMask   = zeros(TaskNum,max(Problem.SubD));
    for i = 1 : TaskNum
        NSpop{i}      = SubPopulation{i}(FrontNo{i} == 1);
        NSdec{i}      = Dec{i}(FrontNo{i} == 1,:);
        NSdec{i}      = [NSdec{i},zeros(size(NSdec{i},1),Buquan(i))];
        NSmask{i}     = Mask{i}(FrontNo{i} == 1,:);
        NSmask{i}     = [NSmask{i},zeros(size(NSmask{i},1),Buquan(i))];
        NSCrowdDis{i} = CrowdDis{i}(FrontNo{i} == 1);
        MeanMask(i,:) = mean(NSdec{i}.*NSmask{i});
    end
    MeanMaskSourceId = [];
    
    OffDec  = {};
    OffMask = {};
    Pop     = {};
    sRatio  = zeros(1,TaskNum);
    CurrentTthetamid = zeros(1,TaskNum);
    for i = 1 : TaskNum
        [OffDec{i},OffMask{i},CurrentNumTransing(i)] = Operator(i,SourceId(i),Problem,NSpop{i},NSdec{i},NSmask{i},NSCrowdDis{i},TF1(i,:),FitnessDec{i},pDecIndex(i,:),FitnessPop(:,i),...
            NSpop{SourceId(i)},NSdec{SourceId(i)},NSmask{SourceId(i)},NSCrowdDis{SourceId(i)},TF1(SourceId(i),:),FitnessDec{SourceId(i)},pDecIndex(SourceId(i),:),FitnessPop(:,SourceId(i)),NumTransUp,ALLTthetamid);
        Skill    = i*ones(size(OffDec{i},1),1);
        Solution = [OffDec{i}.*OffMask{i},Skill];
        if size(Solution,1) > 0
            Pop{i} = Problem.Evaluation(Solution);
        else
            Pop{i} = [];
        end
        [NSpop{i},NSdec{i},NSmask{i},TFrontNo,TCrowdDis,sRatio(i)] = MOEAPSL_EnvironmentalSelection([NSpop{i},Pop{i}],[NSdec{i};OffDec{i}],[NSmask{i};OffMask{i}],length(NSpop{i}),NumTransUp(i));
        NumTransUp(i) = round(NumTransUp(i)*(1+sRatio(i))/2);
        Theta         = sum(NSmask{i}(find(TFrontNo ==1),:),2)';
        CurrentTthetamid(i) = mean(Theta);
    end
    ALLTthetamid = [ALLTthetamid;CurrentTthetamid];
    for i = 1 : TaskNum
        [SubPopulation{i},Dec{i},Mask{i},FrontNo{i},CrowdDis{i}] = SparseEA_EnvironmentalSelection([SubPopulation{i},Pop{i}],[Dec{i};OffDec{i}(:,1:Problem.SubD(i))],[Mask{i};OffMask{i}(:,1:Problem.SubD(i))],EachN);
    end
end

function [OffDec,OffMask,N] = Operator(TargetId,SourceId,Problem,Pop1,Dec1,Mask1,CrowdDis1,Fitness1,FitnessDec1,pDecIndex1,FitnessPop1,Pop2,Dec2,Mask2,CrowdDis2,Fitness2,FitnessDec2,pDecIndex2,FitnessPop2,NumTransUp,ALLTthetamid)
    [N,D] = deal(size(Dec1,1),Problem.SubD(TargetId));
    if N>NumTransUp(TargetId)
        N = NumTransUp(TargetId);
        TMatingPool = TournamentSelection(2,N,-CrowdDis1);
        TDec        = Dec1(TMatingPool,:);
        TMask       = Mask1(TMatingPool,:);
    else
        TDec  = Dec1;
        TMask = Mask1;
    end
    Tthetamid = ALLTthetamid(end,TargetId);

    SMatingPool = TournamentSelection(2,N,-CrowdDis2);
    SDec        = Dec2(SMatingPool,:);
    SMask       = Mask2(SMatingPool,:);
    
    S1index = sum(SMask)>0;

    OffMask = TMask;
    for i = 1 : N
        if sum(OffMask(i,:)) <= Tthetamid
            index0           = find(~TMask(i,:)&S1index);
            index0(index0>D) = [];
            minN             = floor(Tthetamid - sum(OffMask(i,:)));
            index            = index0(TStransMany(Fitness1(index0),Fitness2(index0),minN));
            OffMask(i,index) = 1;
        else
            index            = find(~TMask(i,:)&SMask(i,:));
            index(index>D)   = [];
            index            = index(TStrans(Fitness1(index),Fitness2(index)));
            OffMask(i,index) = SMask(i,index);
        end
    end

    for i = 1 : N
        index            = find(OffMask(i,:));
        index(index>D)   = [];
        index            = index(TS(-Fitness1(index)));
        OffMask(i,index) = 0;
    end

    if any(Problem.encoding~=4)
        OffDec = TDec;
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(size(OffMask));
    end
end

function index = TStransMany(Fitness1,Fitness2,minN)
% Binary tournament selection

    if isempty(Fitness1)
        index = [];
    else
        index = TournamentSelection(2,minN,Fitness1,Fitness2);
    end
end

function index = TS(Fitness)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2,1,Fitness);
    end
end

function index = TStrans(Fitness1,Fitness2)
% Binary tournament selection

    if isempty(Fitness1)
        index = [];
    else
        index = TournamentSelection(2,1,Fitness1,Fitness2);
    end
end
```

### `SourceTaskrand.m`
```matlab
function [SourceId,TF1] = SourceTaskrand(Problem,SubPopulation,Dec,Mask,FrontNo,CrowdDis,EachN,Fitness,FitnessDec,pDecIndex,Buquan)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    TaskNum  = size(Problem.SubM,2);
    SourceId = zeros(1,TaskNum);
    TF1 = [];
    for i = 1 : TaskNum
        TF1 = [TF1;Fitness{i}/5,zeros(1,Buquan(i))];
        TF1(TF1==0) = max(Fitness{i}/5)+1;
    end
    mse_matrix = zeros(TaskNum);  
    for i = 1 : TaskNum
        for j = 1:TaskNum
            mse_matrix(i, j) = mean((TF1(i, :) - TF1(j, :)).^2); 
        end
        arr          = find(mse_matrix(i,:) ~= 0);
        shuffled_arr = arr(randperm(length(arr)));
        SourceId(i)  = shuffled_arr(1);
    end        
end
```

### `SparseEA_ESnouni.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = SparseEA_ESnouni(Population,Dec,Mask,N)
% The environmental selection of SparseEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `SparseEA_EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = SparseEA_EnvironmentalSelection(Population,Dec,Mask,N)
% The environmental selection of SparseEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions
    [~,uni]    = unique(Population.objs,'rows');
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```
