# EDN-ARMOEA

**Tags**: <2022> <multi/many> <real/integer> <expensive>

## Description
Efficient dropout neural network based AR-MOEA

## Reference
D. Guo, X. Wang, K. Gao, Y. Jin, J. Ding, and T. Chai. Evolutionary optimization of high-dimensional multiobjective and many-objective expensive problems assisted by a dropout neural network. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(4): 2084-2097.

## Source Code

### `CalDistance.m`
```matlab
function Distance = CalDistance(PopObj,RefPoint)
% Calculate the distance between each solution to each adjusted reference point

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    N  = size(PopObj,1);
    NR = size(RefPoint,1);

    %% Adjust the location of each reference point
    index=find(sum(PopObj.^2,2)==0);
    if ~isempty(index)
        PopObj(index,:)=PopObj(index,:)+1e-06*rand(size(PopObj(index,:)));
    end
    Cosine = 1 - pdist2(PopObj,RefPoint,'cosine');%The cosine of the individual and the reference point in the population, N*NR
    NormR  = sqrt(sum(RefPoint.^2,2)); %NR*1
    NormP  = sqrt(sum(PopObj.^2,2)); %N*1
    d1     = repmat(NormP,1,NR).*Cosine; %N*NR
    d2     = repmat(NormP,1,NR).*sqrt(1-Cosine.^2); %N*NR
    [~,nearest] = min(d2,[],1); %1*NR
    RefPoint    = RefPoint.*repmat(d1(N.*(0:NR-1)+nearest)'./NormR,1,size(RefPoint,2));
    
    %% Calculate the distance between each solution to each point
    Distance = pdist2(PopObj,RefPoint);
end
```

### `EDNARMOEA.m`
```matlab
classdef EDNARMOEA < ALGORITHM
% <2022> <multi/many> <real/integer> <expensive>
% Efficient dropout neural network based AR-MOEA
% delta --- 0.05 --- Threshold of judging the diversity
% wmax  ---   20 --- Number of generations before updating Kriging models
% Ke    ---    3 --- Number of the solutions to be revaluated in each iteration

%------------------------------- Reference --------------------------------
% D. Guo, X. Wang, K. Gao, Y. Jin, J. Ding, and T. Chai. Evolutionary
% optimization of high-dimensional multiobjective and many-objective
% expensive problems assisted by a dropout neural network. IEEE
% Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(4):
% 2084-2097.
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
            assert(~isempty(ver('nnet')),'The execution of EDN-ARMOEA requires the Deep Learning Toolbox.');
            
            %% Parameter setting
            [delta,wmax,Ke] = Algorithm.ParameterSet(0.05,20,3);            
            W     = UniformPoint(Problem.N,Problem.M);
            NI    = 11*Problem.D-1;
            P     = UniformPoint(NI,Problem.D,'Latin');
            A     = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            tr_x  = A.decs;
            tr_y  = A.objs;
            [tr_xx,ps] = mapminmax(tr_x');tr_xx=tr_xx';
            [tr_yy,qs] = mapminmax(tr_y');tr_yy=tr_yy';
            Params.ps  = ps;Params.qs=qs;           
            RatioOld   = [];
            
            %% Train the model
            [net, Params] = trainmodel(tr_xx, tr_yy, Params);

            while Algorithm.NotTerminated(A)
                %% Update the model 
                net=updatemodel(tr_xx, tr_yy, Params, net);

                %% Generate the sampling points and random population
                popsize=Problem.N;
                PopDec=repmat(Problem.upper-Problem.lower,popsize,1).*rand(popsize,Problem.D)+repmat(Problem.lower,popsize,1);
                [PopObj, PopMSE]=Estimate(PopDec, net, Params, Problem.M);
                [Archive,RefPoint,Range, Ratio] = UpdateRefPoint(PopObj,W,[]);
                if isempty(RatioOld)
                    RatioOld=Ratio;
                end
               
                %% Start the interations
                w=1;
                while w < wmax
                    MatingPool = MatingSelection(PopObj,RefPoint,Range);
                    OffspringDec  = OperatorGA(Problem,PopDec(MatingPool,:),{1,20,1,20});
                    [OffspringObj, OffspringMSE] = Estimate(OffspringDec, net, Params, Problem.M);
                    [Archive,RefPoint,Range, Ratio] = UpdateRefPoint([Archive;OffspringObj],W,Range);
                    MediatePopDec=[PopDec;OffspringDec];
                    MediatePopObj=[PopObj;OffspringObj];
                    MediatePopMSE=[PopMSE;OffspringMSE];
                    [Index,Range]       = EnvironmentalSelection(MediatePopObj,RefPoint,Range,popsize);
                    PopDec=MediatePopDec(Index,:);
                    PopObj=MediatePopObj(Index,:);
                    PopMSE=MediatePopMSE(Index,:);
                    w=w+1;
                end
                flag=RatioOld-Ratio<delta;
                PopNew=IndividualSelect(PopDec, PopObj, PopMSE, Ke, flag);
                RatioOld=Ratio;
                New = Problem.Evaluation(PopNew);
                A   = [A,New];
                [tr_x, tr_y]=SelectTrainData(A, 11*Problem.D-1, length(New));
                [tr_xx,ps]=mapminmax(tr_x');tr_xx=tr_xx';
                [tr_yy,qs]=mapminmax(tr_y');tr_yy=tr_yy';
                Params.ps=ps;Params.qs=qs;   
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Next,Range] = EnvironmentalSelection(Obj,RefPoint,Range,N)
% The environmental selection of AR-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Selection among feasible solutions
    % Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Obj,N);%population 2*N
    Next = FrontNo < MaxFNo;
    % Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Obj(Last,:),RefPoint,Range,N-sum(Next));
    Next(Last(Choose)) = true;
    % Update the range for normalization
    Range(2,:) = max(Obj,[],1);
    Range(2,Range(2,:)-Range(1,:)<1e-6) = 1;
end

function Remain = LastSelection(PopObj,RefPoint,Range,K)
% Select part of the solutions in the last front

    N  = size(PopObj,1);
    NR = size(RefPoint,1);

    %% Calculate the distance between each solution and point
    Distance    = CalDistance(PopObj-repmat(Range(1,:),N,1),RefPoint);
    Convergence = min(Distance,[],2);
    
    %% Delete the solution which has the smallest metric contribution one by one
    [dis,rank] = sort(Distance,1);
    Remain     = true(1,N);
    while sum(Remain) > K
        % Calculate the fitness of noncontributing solutions
        Noncontributing = Remain;
        Noncontributing(rank(1,:)) = false;
        METRIC = sum(dis(1,:)) + sum(Convergence(Noncontributing));
        Metric = inf(1,N);
        Metric(Noncontributing) = METRIC - Convergence(Noncontributing);
        % Calculate the fitness of contributing solutions
        for p = find(Remain & ~Noncontributing)
            temp = rank(1,:) == p;
            noncontributing = false(1,N);
            noncontributing(rank(2,temp)) = true;
            noncontributing = noncontributing & Noncontributing;
            Metric(p) = METRIC - sum(dis(1,temp)) + sum(dis(2,temp)) - sum(Convergence(noncontributing));
        end
        % Delete the worst solution and update the variables
        [~,del] = min(Metric);
        temp    = rank ~= del;
        dis     = reshape(dis(temp),sum(Remain)-1,NR);
        rank    = reshape(rank(temp),sum(Remain)-1,NR);
        Remain(del) = false;
    end
end
```

### `IndividualSelect.m`
```matlab
function xnew = IndividualSelect(PopDec, PopObj, PopMSE, Ke, flag)
% The selection of individuals

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [IDX, ~]=kmeans(PopObj,Ke);
    UIDX=unique(IDX);  
    
    if flag==0
        Uncertainty=mean(PopMSE,2);
        for i=1:length(UIDX)
            Pindex=find(IDX==UIDX(i));
            [~,ind]=max(Uncertainty(Pindex));
            %disp(sprintf('ind is %u', ind)); disp(Pindex');
            xnew(i,:)=PopDec(Pindex(ind),:);          
        end
    else
        Convergence=sqrt(sum(PopObj.^2,2));
        for i=1:length(UIDX)
            Pindex=find(IDX==UIDX(i));
            [~,ind]=min(Convergence(Pindex));
            %disp(sprintf('ind is %u', ind));disp(Pindex');
            xnew(i,:)=PopDec(Pindex(ind),:);      
        end        
    end
end
```

### `MatingSelection.m`
```matlab
function MatingPool = MatingSelection(Obj,RefPoint,Range)
% The mating selection of AR-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each feasible solution based on IGD-NS
    % Calculate the distance between each solution and point
    N = size(Obj,1);
    Distance    = CalDistance(Obj-repmat(Range(1,:),N,1),RefPoint);%N*NR
    Convergence = min(Distance,[],2);% The minimum Angle between each solution and the reference point
    [dis,rank]  = sort(Distance,1);% In columns, for reference points
    % Calculate the fitness of noncontributing solutions
    Noncontributing = true(1,N);
    Noncontributing(rank(1,:)) = false;
    METRIC   = sum(dis(1,:)) + sum(Convergence(Noncontributing));
    % The sum of the minimum angles of all reference points plus the sum of the minimum angles of all non-contributing solutions
    fitness  = inf(1,N);
    fitness(Noncontributing) = METRIC - Convergence(Noncontributing);
    % Calculate the fitness of contributing solutions
    for p = find(~Noncontributing)
        temp = rank(1,:) == p;
        noncontributing = false(1,N);
        noncontributing(rank(2,temp)) = true;% After removing this contribution point, the new contribution point Index
        noncontributing = noncontributing & Noncontributing;% Judge whether the new contribution point is the original non-contribution point or not
        fitness(p) = METRIC - sum(dis(1,temp)) + sum(dis(2,temp)) - sum(Convergence(noncontributing));
        % When this contribution point is removed, the METRIC adjusts
    end

    %% Combine the fitness of feasible solutions with the fitness of infeasible solutions
    Fitness = fitness;
    
    %% Binary tournament selection
    CV=zeros(N,1);
    MatingPool = TournamentSelection(2,ceil(N/2)*2,CV,-Fitness);
end
```

### `SelectTrainData.m`
```matlab
function [tr_xx, tr_yy] = SelectTrainData(P, N1, N2)
% Select data to train the model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    tr_y  = P.objs;
    NA    = size(tr_y,1);
    Range = [min(tr_y,[],1);max(tr_y,[],1)];

    tr_y   = tr_y - repmat(Range(1,:),NA,1);
    Cosine = 1 - pdist2(tr_y,tr_y,'cosine');%NA*NA
    Cosine(logical(eye(size(Cosine,1)))) = 0;
    Choose = [false(1,NA-N2),true(1,N2)];

    while sum(Choose)<N1
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
    end

    tr_yy = tr_y(Choose,:);
    tr_x  = P.decs;
    tr_xx = tr_x(Choose,:);
end
```

### `UpdateRefPoint.m`
```matlab
function [Archive,RefPoint,Range, Ratio] = UpdateRefPoint(Archive,W,Range)
% Reference point adaption

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

	%% Delete duplicated and dominated solutions
    Archive = unique(Archive(NDSort(Archive,1)==1,:),'rows');
    % unique(A,'rows') gets the matrix formed by the different vectors of matrix A
    NA      = size(Archive,1);
    NW      = size(W,1);
    
	%% Update the ideal point
    if ~isempty(Range)
        Range(1,:) = min([Range(1,:);Archive],[],1);
    elseif ~isempty(Archive)
        Range = [min(Archive,[],1);max(Archive,[],1)];%2*M
    end
    
    %% Update archive and reference points
    if size(Archive,1) <= 1
        RefPoint = W;
        Ratio    = 0;
    else
        %% Find contributing solutions and valid weight vectors
        tArchive = Archive - repmat(Range(1,:),NA,1);
        W        = W.*repmat(Range(2,:)-Range(1,:),NW,1);
        Distance      = CalDistance(tArchive,W); %The (1-cosine) value of the individual and the reference point W in tArchive, NA*NW
        [~,nearestP]  = min(Distance,[],1);
        ContributingS = unique(nearestP);
        [~,nearestW]  = min(Distance,[],2);
        ValidW        = unique(nearestW(ContributingS));%The reference point with the smallest Angle to the contributing solution

        %% Update archive
        Choose = ismember(1:NA,ContributingS);%1*NA Logical vector
        Cosine = 1 - pdist2(tArchive,tArchive,'cosine');%NA*NA
        Cosine(logical(eye(size(Cosine,1)))) = 0;% The Cosine matrix has a diagonal of 0
        while sum(Choose) < min(3*NW,size(tArchive,1))
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            % Calculate the minimum included Angle between all other solutions and each contributing solution, and add the remaining solutions with the maximum of the minimum included Angle into the contributing solution
            Choose(unSelected(x)) = true;
        end
        Archive  = Archive(Choose,:);
        tArchive = tArchive(Choose,:);

        %% Update reference points
        RefPoint = [W(ValidW,:);tArchive];
        Choose   = [true(1,length(ValidW)),false(1,size(tArchive,1))];
        Cosine   = 1 - pdist2(RefPoint,RefPoint,'cosine');
        Cosine(logical(eye(size(Cosine,1)))) = 0;
        while sum(Choose) < min(NW,size(RefPoint,1))
            Selected = find(~Choose);
            [~,x]    = min(max(Cosine(~Choose,Choose),[],2));
            Choose(Selected(x)) = true;
        end
        RefPoint = RefPoint(Choose,:);
        Ratio    = length(ValidW)/NW;
    end 
end
```
