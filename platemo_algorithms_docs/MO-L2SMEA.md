# MO-L2SMEA

**Tags**: <2025> <multi> <real> <expensive> <large/none>

## Description
Multi-objective linear subspace surrogate modeling assisted evolutionary algorithm

## Reference
L. Si, X. Zhang, Y. Tian, S. Yang, L. Zhang, and Y. Jin. Linear subspace surrogate modeling for large-scale expensive single/multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(3): 697-710.

## Source Code

### `ConstructLinearKRGOS.m`
```matlab
function subproblemList = ConstructLinearKRGOS(refPoints,Archive,nLinear,MaxObj,MinObj,W)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Get basic information
    % Get the number of dimension
    [DL,D]   = size(refPoints);
    BU       = ones(1,D);
    BD       = zeros(1,D);
    [arcN,~] = size(Archive);
    % Check the number of reference point pairs
    if DL/2 ~= nLinear
        error('Check the number of reference points');
    end
    
    %% Construct linear space and build surrogate model
    % Tchebycheff approach with normalization
    Fitness = max(abs(Archive(:, D+1:end)-repmat(MinObj, arcN, 1))./repmat(MaxObj-MinObj,arcN,1).*W, [], 2);
    % Store linear space
    subproblemList = cell(1,nLinear);
    % Process
    for k = 0 : nLinear-1
        %% Construct linear space
        % Define subproblem struct
        subproblem = struct();
        % Get start point
        startPoint = refPoints(k*2+1,1:D);
        endPoint   = refPoints(k*2+2,1:D);
        subproblem.start = startPoint;
        % Calculate direction
        subproblem.direct = (endPoint-startPoint)./sqrt(sum((endPoint-startPoint).^2));
        % Find upper boundary and lower boundary of linear space
        [lub,llb]     = findBoundary(subproblem.start,subproblem.direct,BU,BD);
        subproblem.ub = lub;
        subproblem.lb = llb;
        
        %% Associate points to current linear space
        % Calculate the disance between start point and candidate points
        dist = pdist2(Archive(:,1:D),startPoint);
        % Get direction vector
        vector1 = repmat(subproblem.direct,arcN,1);
        % Get vector <point,startPoint>
        vector2 = Archive(:,1:D) - repmat(startPoint,arcN,1);
        % Calculate cos<vector1,vector2>
        MVL     = sum(vector2.^2,2);
        MVL(MVL==0) = inf;
        cosV    = sum(vector1.*vector2,2)./sqrt(MVL);
        % Calculate the distance between points and current line
        sinV    = sqrt(1-cosV.^2);
        allDist = sinV.*dist;
        % Calculate the transformed decision variables
        newX    = cosV.*dist;
        % associate process
        [sortDis,~] = sort(allDist,'ascend');
        trainSize   = 10;
        alpha  = sortDis(trainSize);
        select = allDist <= alpha;

        %% Build surrogate model
        % Radial basis function
        srgtOPTKRG = srgtsKRGSetOptions(newX(select),Fitness(select));
        srgtsKRG   = srgtsKRGFit(srgtOPTKRG);
        subproblem.fobj = @(Dec)srgtsKRGPredictor(Dec,srgtsKRG);

        subproblem.trainDec = newX(select);
        subproblem.maxTheta = mean(Fitness(select));
        subproblemList{k+1} = subproblem;
    end
end

function [linearBU,linearBD] = findBoundary(start,direct,BU,BD)
    lenMax = sqrt(sum((BU-BD).^2)) + 1e-8;
    
    %% Find the upper boundary of line
    ls = 0;
    le = lenMax;
    while true
        middle = (ls + le)/2;
        xNew   = start + middle*direct;
        if any(xNew>BU | xNew<BD) 
            le = middle;
        else
            ls = middle;
        end
        if le-ls < 1e-12
            break;
        end
    end
    linearBU = ls;
    
    %% Find the lower boundary of line
    ls = 0;
    le = -lenMax;
    while true
        middle = (ls + le)/2;
        xNew   = start + middle*direct;
        if any(xNew>BU | xNew<BD)
            le = middle;
        else
            ls = middle;
        end
        if ls-le < 1e-12
            break;
        end
    end
    linearBD = ls;
end
```

### `MOEAOptimizeOS.m`
```matlab
function candidates = MOEAOptimizeOS(subproblemList,popSize,maxIter)
% Optimize each subproblem by multi-objective algorithm

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Get dimension and boundary
    D = numel(subproblemList);
    [BU,BD] = deal(zeros(1,D));
    for k = 1 : D
        BU(k) = subproblemList{k}.ub;
        BD(k) = subproblemList{k}.lb;
    end
    
    %% Generate the random population
    population.decs = rand(popSize,D).*repmat(BU-BD,popSize,1) + repmat(BD,popSize,1);
    population.objs = SaEvaluateOS(subproblemList,population.decs);
    
    %% Optimize by multi-objective algorithm
    candidates = NSGAIIIopt(subproblemList,BU,BD,population,popSize,maxIter);
    [~,stds]   = SaEvaluateOS(subproblemList,candidates(:,1:D));
    candidates = [candidates, stds];
end
```

### `MOL2SMEA.m`
```matlab
classdef MOL2SMEA < ALGORITHM
% <2025> <multi> <real> <expensive> <large/none>
% Multi-objective linear subspace surrogate modeling assisted evolutionary algorithm
% NLinear --- 8 --- Number of linear subspaces

%------------------------------- Reference --------------------------------
% L. Si, X. Zhang, Y. Tian, S. Yang, L. Zhang, and Y. Jin. Linear subspace
% surrogate modeling for large-scale expensive single/multi-objective
% optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(3):
% 697-710.
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
            %% Generate random population
            NLinear    = Algorithm.ParameterSet(8);
            NT         = 2*Problem.D;
            MaxFEs     = Problem.maxFE - NT;
            Dec        = lhsdesign(NT,Problem.D).*repmat(Problem.upper-Problem.lower,NT,1) + repmat(Problem.lower,NT,1);
            Population = Problem.Evaluation(Dec);
            Objs       = Population.objs;
            MaxObj     = max(Objs,[],1);
            MinObj     = min(Objs,[],1);
            TArchive   = [Dec,Population.objs];
            TArchive(:,1:Problem.D) = (TArchive(:,1:Problem.D)-repmat(Problem.lower,NT,1))./repmat(Problem.upper-Problem.lower,NT,1);
            
            %% Set parameters
            % Parameter of multi-objective algorithm
            PopSize = 40;
            MaxIter = 30;
            % Parameter of surrogate model
            CurrFEs = 0;
            
            %% Cluster number is 5
            CLTn     = 5;
            [W,CLTn] = UniformPoint(CLTn, Problem.M);
            % Parameter in CMA-ES
            XMeanCell    = cell(1, CLTn);
            SigmaCell    = cell(1, CLTn);
            PathConvCell = cell(1, CLTn);
            PathSigCell  = cell(1,CLTn);
            BCell        = cell(1,CLTn);
            DiagCell     = cell(1,CLTn);
            ConvCell     = cell(1,CLTn);
            InvSqrtCCell = cell(1,CLTn);
            NObjs        = (TArchive(:, end-Problem.M+1:end) - repmat(MinObj,size(TArchive,1),1))./repmat(MaxObj-MinObj,size(TArchive,1),1);
            [~,grp]      = min(pdist2(W, NObjs,'cosine'), [], 1);
            for k = 1 : CLTn
                if any(grp==k)
                    XMeanCell{k} = mean(TArchive(grp==k,1:Problem.D),1)';
                else
                    XMeanCell{k} = mean(TArchive(:,1:Problem.D),1)';
                end
                SigmaCell{k}    = 0.5;
                PathConvCell{k} = zeros(Problem.D,1);
                PathSigCell{k}  = zeros(Problem.D,1);
                BCell{k}        = eye(Problem.D,Problem.D);
                DiagCell{k}     = ones(Problem.D,1);
                ConvCell{k}     = BCell{k}*diag(DiagCell{k}.^2)*BCell{k}';
                InvSqrtCCell{k} = BCell{k}*diag(DiagCell{k}.^-1)*BCell{k}';
            end
            EigenEval = 0;
            ChiD      = Problem.D^0.5*(1-1/(4*Problem.D)+1/21*Problem.D^2);

            % Selection
            Lambda  = 4*NLinear;
            Mu      = Lambda/2;
            Weights = log(Mu+1/2) - log(1:Mu)';
            Mu      = floor(Mu);
            Weights = Weights/sum(Weights);
            MuEff   = sum(Weights)^2/sum(Weights.^2);

            % Adaptation
            CumConv = (4+MuEff/Problem.D)/(Problem.D+4+2*MuEff/Problem.D);
            CumSig  = (MuEff+2)/(Problem.D+MuEff+5);
            CRate   = 2/((Problem.D+1.3)^2+MuEff);
            CMu     = min(1-CRate,2*(MuEff-2+1/MuEff)/((Problem.D+2)^2+MuEff));
            Damps   = 1 + 2*max(0,sqrt((MuEff-1)/(Problem.D+1))-1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                NObjs   = (TArchive(:, end-Problem.M+1:end) - repmat(MinObj,size(TArchive,1),1))./repmat(MaxObj-MinObj,size(TArchive,1),1);
                [~,grp] = min(pdist2(W, NObjs,'cosine'), [], 1);
                for k = 1 : CLTn
                    %% Construct several linear space
                    % Select reference points
                    RefPoints = SelectRef(NLinear,XMeanCell{k},SigmaCell{k},BCell{k},DiagCell{k});
                    tArc = TArchive(grp==k,:);
                    if sum(grp==k) < 30
                        disth = pdist2(W(k,:),NObjs,'cosine');
                        [~,indx] = sort(disth);
                        tArc = TArchive(indx(1:2*Problem.D),:);
                    end
                    % Construct process
                    SubproblemList = ConstructLinearKRGOS(RefPoints,tArc,NLinear, MaxObj,MinObj, W(k,:));
                    
                    %% Optimize process
                    % Optimize by multi-objective algorithm
                    Candidates = MOEAOptimizeOS(SubproblemList,PopSize,MaxIter);
                    % Select candidate solutions
                    Offspring = SelectCandidateKRGOS(SubproblemList,NLinear,Problem.D,Candidates);
                    Remain    = MaxFEs - CurrFEs;
                    
                    %% Update Archive
                    [TArchive,CurrFEs,tArc,Population, MaxObj,MinObj] = UpdateArchiveCLT(TArchive,Problem,Offspring,Remain,CurrFEs,tArc,Population,MaxObj,MinObj);
                    [FrontNo,~] = NDSort(tArc(:, Problem.D+1:end),size(tArc,1));
                    
                    if size(tArc,1) >= Mu
                        in = find(FrontNo==1);
                        id = find(FrontNo~=1);
                        if length(in) >= ceil(Mu/2)
                            if length(id) >= ceil(Mu/2)
                                sni     = randperm(length(in),ceil(Mu/2));
                                sdi     = randperm(length(id),ceil(Mu/2));
                                MeanDec = [tArc(in(sni),1:Problem.D);tArc(id(sdi),1:Problem.D)];
                            else
                                sni     = randperm(length(in),Mu-length(id));
                                MeanDec = [tArc(id,1:Problem.D);tArc(in(sni),1:Problem.D)];
                            end                    
                        else
                            sdi     = randperm(length(id),Mu-length(in));
                            MeanDec = [tArc(in,1:Problem.D);tArc(id(sdi),1:Problem.D)];
                        end
                    else
                        dist    = pdist2(W(k,:), NObjs,'cosine');
                        [~,cdt] = sort(dist,'ascend');
                        MeanDec = TArchive(cdt(1:Mu),1:Problem.D);
                    end

                    XOld         = XMeanCell{k};
                    XMeanCell{k} = MeanDec'*Weights;
                    
                    PathSigCell{k}  = (1-CumSig)*PathSigCell{k} + sqrt(CumSig*(2-CumSig))*MuEff*InvSqrtCCell{k}*(XMeanCell{k}-XOld)/SigmaCell{k};
                    HSig            = norm(PathSigCell{k})/sqrt(1-(1-CumSig)^(2*CurrFEs/Lambda))/ChiD < 1.4+2/(Problem.D+1);
                    PathConvCell{k} = (1-CumConv)*PathConvCell{k} + HSig*sqrt(CumConv*(2-CumConv)*MuEff)*(XMeanCell{k}-XOld)/SigmaCell{k};
                    ArcTmp          = (1/SigmaCell{k})*(MeanDec'-repmat(XOld,1,Mu));
                    
                    ConvCell{k} = (1-CRate-CMu)*ConvCell{k} + CRate*(PathConvCell{k}*PathConvCell{k}'+(1-HSig)*CumConv*(2-CumConv)*ConvCell{k}) + CMu*ArcTmp*diag(Weights)*ArcTmp';
                    
                    SigmaCell{k} = SigmaCell{k}*exp((CumSig/Damps)*(norm(PathSigCell{k})/ChiD-1));
                    if CurrFEs - EigenEval > Lambda/(CRate+CMu)/Problem.D/10
                        EigenEval       = CurrFEs;
                        ConvCell{k}     = triu(ConvCell{k}) + triu(ConvCell{k},1)';
                        [BCell{k},DiagCell{k}] = eig(ConvCell{k});
                        DiagCell{k}     = sqrt(diag(DiagCell{k}));
                        InvSqrtCCell{k} = BCell{k}*diag(DiagCell{k}.^-1)*BCell{k}';
                        if ~isreal(DiagCell{k})
                            DiagCell{k} = real(DiagCell{k});
                        end
                    end
                end
            end
        end
    end
end
```

### `SaEvaluate.m`
```matlab
function objs = SaEvaluate(subproblemList,decs)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,nLinear] = size(decs);
    objs = zeros(N,nLinear);
    for k = 1 : nLinear
        objs(:,k) = subproblemList{k}.fobj(decs(:,k));
    end
end
```

### `SaEvaluateOS.m`
```matlab
function [objs, stds] = SaEvaluateOS(subproblemList,decs)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,nLinear] = size(decs);
    objs = zeros(N,nLinear);
    stds = zeros(N,nLinear);
    for k = 1 : nLinear
        [objs(:,k), stds(:,k)] = subproblemList{k}.fobj(decs(:,k));
    end
end
```

### `SelectCandidateKRGOS.m`
```matlab
function newOffspring = SelectCandidateKRGOS(subproblemList,nLinear,D,candidates)
% A surrogate-assisted offspring generation method for expensive multi-objective optimization problems

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Get surrogate fitness
    PopObj = candidates(:,end-nLinear+1:2*nLinear);
    
    %% Non-dominated sorting
    FNO = NDSort(PopObj,size(PopObj,1));
    NDOffspring = candidates(FNO==1,:);
    
    %% Candidates selection process of each surrogate model
    newOffspring = [];
    for k = 1 : nLinear
        % Delete candidates closing to existing solutions
        theta     = 0.07 *(subproblemList{k}.ub-subproblemList{k}.lb)*nLinear;
        dist      = pdist2(NDOffspring(:,k),subproblemList{k}.trainDec);
        minDist   = min(dist,[],2);
        remain    = minDist >= theta;
        remainDec = NDOffspring(remain,k);
        remainFit = NDOffspring(remain,nLinear+k);
        remainStd = -NDOffspring(remain,2*nLinear+k);
        % Delete solutions closing to new candidate solutions
        if sum(remain) > 2
            [frontNo,~] = NDSort([remainStd,remainFit],1);
            select      = 1:1:sum(remain);
            while any(frontNo>1) && length(select) > 2
                delIdx         = frontNo>1;
                select(delIdx) = [];
                [frontNo,~]    = NDSort([remainStd(select),remainFit(select)],1);
            end
            finalDec = remainDec(select);
            finalObj = remainFit(select);
            n        = length(select);
            popDec   = repmat(subproblemList{k}.start,n,1) + repmat(finalDec,1,D).*repmat(subproblemList{k}.direct,n,1);
            newOffspring = [newOffspring;popDec,finalObj];
        elseif sum(remain) >= 1
            finalDec = remainDec;
            finalObj = remainFit;
            n        = sum(remain);
            popDec   = repmat(subproblemList{k}.start,n,1) + repmat(finalDec,1,D).*repmat(subproblemList{k}.direct,n,1);
            newOffspring = [newOffspring;popDec,finalObj];
        end
    end
    if isempty(newOffspring)
        [MR,MI] = min(NDOffspring(:,nLinear+1:2*nLinear));
        [~,LI]  = min(MR);
        if size(NDOffspring,1) == 1
            finalDec = NDOffspring(1,MI);
            finalObj = NDOffspring(1,nLinear+MI);
            popDec = repmat(subproblemList{MI}.start,1,1) + repmat(finalDec,1,D).*repmat(subproblemList{MI}.direct,1,1);
        else
            finalDec = NDOffspring(MI(LI),LI);
            finalObj = NDOffspring(MI(LI),nLinear+LI);
            popDec   = repmat(subproblemList{LI}.start,1,1) + repmat(finalDec,1,D).*repmat(subproblemList{LI}.direct,1,1);
        end
        newOffspring = [popDec,finalObj];
    end
end
```

### `SelectRef.m`
```matlab
function refPoints = SelectRef(nLinear,xmean,sigma,B,Diag)
% This function is used to select several pair of reference points

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    D = length(xmean);
    % Select reference points
    refPoints = zeros(2*nLinear,D);
    for k = 1 : 2*nLinear
        refPoints(k,:) = (xmean + sigma * B * (Diag.*randn(D,1)))';
    end
    % Repair invalidate reference points, default BU and BD is 1,0
    refPoints(refPoints<0) = 1e-6;
    refPoints(refPoints>1) = 1-(1e-6);
end
```

### `UpdateArchive.m`
```matlab
function [Archive,currFEs,Population] = UpdateArchive(Archive,Problem,popNew,remain,currFEs,Population,MaxObj,MinObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    newN = size(popNew);
    D    = Problem.D;
    BU   = Problem.upper;
    BD   = Problem.lower;
    
    %% Update Archive
    if newN == 0
        return;
    elseif newN <= remain
        % Evaluate all candidate solutions with true function
        Dec        = popNew(:,1:D).*repmat(BU-BD,newN,1) + repmat(BD,newN,1);
        NEWOFF     = Problem.Evaluation(Dec);
        Population = [Population,NEWOFF];
        NNEW       = length(NEWOFF);
        MinObj     = min([NNEW.objs;MinObj],[],1);
        MaxObj     = max([NNeW.objs;MaxObj],[],1);
        Obj        = (NEWOFF.objs-repmat(MinObj,NNEW,1))./repmat(MaxObj-MinObj,NNEW,1);
        Obj        = sum(Obj,2);
        Archive    = [Archive;popNew(:,1:D),Obj];
        currFEs    = currFEs + newN;
    else
        % Evaluate top 'remain' candidate solutions with true function
        [~,idx]    = sort(popNew(:,D+1),'ascend');
        topRemain  = idx(1:remain);
        Dec        = popNew(topRemain,1:D).*repmat(BU-BD,remain,1) + repmat(BD,remain,1);
        NEWOFF     = Problem.Evaluation(Dec);
        NNEW       = length(NEWOFF);
        Obj        = (NEWOFF.objs-repmat(MinObj,NNEW,1))./repmat(MaxObj-MinObj,NNEW,1);
        Obj        = sum(Obj,2);
        Population = [Population,NEWOFF];
        Archive    = [Archive;popNew(topRemain,1:D),Obj];
        currFEs    = currFEs + remain;
    end
end
```

### `UpdateArchiveCLT.m`
```matlab
function [Archive,currFEs,tArc,Population,MaxObj,MinObj] = UpdateArchiveCLT(Archive,Problem,popNew,remain,currFEs,tArc,Population,MaxObj,MinObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    newN = size(popNew,1);
    D    = Problem.D;
    BU   = Problem.upper;
    BD   = Problem.lower;
    
    %% Update Archive
    if newN == 0 || remain <= 0
        return;
    elseif newN <= remain
        % Evaluate all candidate solutions with true function
        Dec        = popNew(:,1:D).*repmat(BU-BD,newN,1) + repmat(BD,newN,1);
        NEWOFF     = Problem.Evaluation(Dec);
        Population = [Population,NEWOFF];
        DECS       = (NEWOFF.decs-repmat(Problem.lower,length(NEWOFF),1))./repmat(Problem.upper-Problem.lower,length(NEWOFF),1);
        MinObj     = min([MinObj;NEWOFF.objs],[],1);
        MaxObj     = max([MaxObj;NEWOFF.objs],[],1);
        tArc       = [tArc;DECS, NEWOFF.objs];
        Archive    = [Archive;DECS,NEWOFF.objs];
        currFEs    = currFEs + newN;
    else
        % Evaluate top 'remain' candidate solutions with true function
        [~,idx]    = sort(popNew(:,D+1),'ascend');
        topRemain  = idx(1:remain);
        Dec        = popNew(topRemain,1:D).*repmat(BU-BD,remain,1) + repmat(BD,remain,1);
        NEWOFF     = Problem.Evaluation(Dec);
        Population = [Population,NEWOFF];
        DECS       = (NEWOFF.decs-repmat(Problem.lower,length(NEWOFF),1))./repmat(Problem.upper-Problem.lower,length(NEWOFF),1);
        MinObj     = min([MinObj;NEWOFF.objs],[],1);
        MaxObj     = max([MaxObj;NEWOFF.objs],[],1);
        tArc       = [tArc;DECS, NEWOFF.objs];
        Archive    = [Archive;DECS,NEWOFF.objs];
        currFEs    = currFEs + remain;
    end
end
```
