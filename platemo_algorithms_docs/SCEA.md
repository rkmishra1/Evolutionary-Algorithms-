# SCEA

**Tags**: <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>

## Description
Sparsity clustering basec evolutionary algorithm

## Reference
Y. Zhang, C. Wu, Y. Tian, and X. Zhang. A co-evolutionary algorithm based on sparsity clustering for sparse large-scale multi-objective optimization. Engineering Applications of Artificial Intelligence, 2024, 133: 108194

## Source Code

### `InitRank.m`
```matlab
function [SubPopulation,SubDec,SubMask,Rank] = InitRank(SubPopulation,SubDec,SubMask)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Rank = {};
    SubPopulationCount = zeros(1,size(SubPopulation,2));
    for i = 1 : size(SubPopulation,2)
        SubPopulationCount(i) = size(SubPopulation{i},2);
    end
    for j = 1 : size(SubPopulation,2)
        [~,FrontNo{j},CrowdDis{j}] = EnSelection(SubPopulation{j},SubPopulationCount(j));
        [FrontNo{j},I1] = sort(FrontNo{j});
        CrowdDis{j}     = CrowdDis{j}(I1);

        SubDec{j} = SubDec{j}(I1,:);

        SubMask{j} = SubMask{j}(I1,:);

        SubPopulation{j} = SubPopulation{j}(I1);
        count = zeros(1,max(FrontNo{j}));
        for i = 1 : max(FrontNo{j})
            for m = 1 : SubPopulationCount(j)
                if FrontNo{j}(m)==i
                    count(i) = count(i)+1;
                end
            end
        end
        NFindex = 1;
        for i = 1 : max(FrontNo{j})
            NFcount = count(i);
            [CrowdDis{j}(NFindex:NFindex+NFcount-1),I2] = sort(-CrowdDis{j}(NFindex:NFindex+NFcount-1));
            tempPop = SubPopulation{j}(NFindex:NFindex+NFcount-1);

            tempDec  = SubDec{j}(NFindex:NFindex+NFcount-1,:);
            tempMask = SubMask{j}(NFindex:NFindex+NFcount-1,:);

            tempPop = tempPop(I2);

            tempDec  = tempDec(I2,:);
            tempMask = tempMask(I2,:);

            SubPopulation{j}(NFindex:NFindex+NFcount-1) = tempPop;

            SubDec{j}(NFindex:NFindex+NFcount-1,:)  = tempDec;
            SubMask{j}(NFindex:NFindex+NFcount-1,:) = tempMask;

            NFindex = NFindex+NFcount;
        end
    end
    for i = 1 : size(SubPopulation,2)
        Rank{i} = 1:SubPopulationCount(i);
    end
end

function [Population,FrontNo,CrowdDis] = EnSelection(Population,N)
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
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Population,Dec,Mask,Fitness,winIndex,loseIndex1,loseIndex2,Problem,thetamid,REAL)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % Divide population into three subpopulations
    SubPopulation    = {};
    SubPopulation{1} = Population(winIndex);
    SubPopulation{2} = Population(loseIndex1);
    SubPopulation{3} = Population(loseIndex2);

    SubDec    = {};
    SubDec{1} = Dec(winIndex,:);
    SubDec{2} = Dec(loseIndex1,:);
    SubDec{3} = Dec(loseIndex2,:);

    SubMask    = {};
    SubMask{1} = Mask(winIndex,:);
    SubMask{2} = Mask(loseIndex1,:);
    SubMask{3} = Mask(loseIndex2,:);

    % Sort the solutions in population
    [SubPopulation,SubDec,SubMask,Rank] = InitRank(SubPopulation,SubDec,SubMask); 

    % Generate Offspring along the direction increasing sparsity to cosparsity
    if size(SubPopulation{2},2)>size(SubPopulation{3},2)&&size(SubPopulation{2},2)>floor(Problem.N/3)
        [OffDec1,OffMask1] = OperatorWin(Problem,[SubPopulation{1},SubPopulation{3}],[SubDec{1};SubDec{3}],[SubMask{1};SubMask{3}],[Rank{1},Rank{3}+max(Rank{1})],Fitness,REAL);
        [OffDec2,OffMask2] = OperatorMin(Problem,SubPopulation{1},SubDec{1},SubMask{1},Rank{1},SubPopulation{2},SubDec{2},SubMask{2},Rank{2},Fitness,thetamid,REAL);
        OffDec3  = [];
        OffMask3 = [];
    % Generate Offspring along the direction decreasing sparsity to cosparsity
    elseif size(SubPopulation{2},2)<size(SubPopulation{3},2)&&size(SubPopulation{3},2)>floor(Problem.N/3)
        [OffDec1,OffMask1] = OperatorWin(Problem,[SubPopulation{1},SubPopulation{2}],[SubDec{1};SubDec{2}],[SubMask{1};SubMask{2}],[Rank{1},Rank{2}+max(Rank{1})],Fitness,REAL);
        [OffDec3,OffMask3] = OperatorMax(Problem,SubPopulation{1},SubDec{1},SubMask{1},Rank{1},SubPopulation{3},SubDec{3},SubMask{3},Rank{3},Fitness,thetamid,REAL);
        OffDec2  = [];
        OffMask2 = [];
    else
        [OffDec1,OffMask1] = OperatorWin(Problem,SubPopulation{1},SubDec{1},SubMask{1},Rank{1},Fitness,REAL);
        if size(SubPopulation{2},2)>0
            [OffDec2,OffMask2] = OperatorMin(Problem,SubPopulation{1},SubDec{1},SubMask{1},Rank{1},SubPopulation{2},SubDec{2},SubMask{2},Rank{2},Fitness,thetamid,REAL);
        else
            OffDec2  = [];
            OffMask2 = [];
        end
        if size(SubPopulation{3},2)>0
            [OffDec3,OffMask3] = OperatorMax(Problem,SubPopulation{1},SubDec{1},SubMask{1},Rank{1},SubPopulation{3},SubDec{3},SubMask{3},Rank{3},Fitness,thetamid,REAL);
        else
            OffDec3  = [];
            OffMask3 = [];
        end
    end
    OffDec  = [OffDec1;OffDec2;OffDec3];
    OffMask = [OffMask1;OffMask2;OffMask3];
end

function [OffDec,OffMask] = OperatorWin(Problem,SubPopulation,SubDec,SubMask,Rank,Fitness,REAL)
    MatingPool = TournamentSelection(2,2*size(SubPopulation,2),Rank);
    ParentDec  = SubDec(MatingPool,:);
    ParentMask = SubMask(MatingPool,:);
    % Parameter setting
    [N,D] = size(ParentDec);
    randN = randperm(N);
    Parent1Mask = ParentMask(randN(1:N/2),:);
    Parent2Mask = ParentMask(randN(N/2+1:end),:);
    
    % Crossover
    k = rand(N/2,D) < 0.5;
    k(repmat(rand(N/2,1)>1,1,D)) = false;
    OffMask    = Parent1Mask;
    OffMask(k) = Parent2Mask(k);
    % Mutation
    Site = rand(N/2,D) < 1/D;
    OffMask(Site) = ~OffMask(Site);

    % Crossover and mutation for dec
    if REAL
        OffDec = OperatorGAhalf(Problem,ParentDec);
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

function [OffDec,OffMask] = OperatorMin(Problem,winpop,windec,winmask,winrank,SubPopulation,SubDec,SubMask,Rank,Fitness,thetamid,REAL)
    winPool     = TournamentSelection(2,size(SubPopulation,2),winrank);
    Parent1Dec  = windec(winPool,:);
    Parent1Mask = winmask(winPool,:);

    MatingPool  = TournamentSelection(2,size(SubPopulation,2),Rank);
    Parent2Dec  = SubDec(MatingPool,:);
    Parent2Mask = SubMask(MatingPool,:);

    ParentDec = [Parent1Dec;Parent2Dec];
    [N,D]     = size(ParentDec);
    
    % Crossover
    k = rand(N/2,D) < 0.5;
    k(repmat(rand(N/2,1)>1,1,D)) = false;
    OffMask    = Parent1Mask;
    OffMask(k) = Parent2Mask(k);

    % Mutation
    for i = 1 : N/2
        if sum(OffMask(i,:))>=thetamid
            % Bit-flip mutation
            Site = rand(1,D) < 1/D;
            OffMask(i,Site) = ~OffMask(i,Site);
        else
            index1 = find(OffMask(i,:));
            index0 = find(~OffMask(i,:));
            minN   = floor(thetamid - size(index1,2));
            index  = index0(TSmin(Fitness(index0),minN));
            OffMask(i,index) = 1;
        end
    end

    % Crossover and mutation for dec
    if REAL
        OffDec = OperatorGAhalf(Problem,ParentDec);
    else
        OffDec = ones(N/2,D);
    end
end

function index = TSmin(Fitness,minN)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2,minN,Fitness);
    end
end

function [OffDec,OffMask] = OperatorMax(Problem,winpop,windec,winmask,winrank,SubPopulation,SubDec,SubMask,Rank,Fitness,thetamid,REAL)
    winPool     = TournamentSelection(2,size(SubPopulation,2),winrank);
    Parent1Dec  = windec(winPool,:);
    Parent1Mask = winmask(winPool,:);

    MatingPool  = TournamentSelection(2,size(SubPopulation,2),Rank);
    Parent2Dec  = SubDec(MatingPool,:);
    Parent2Mask = SubMask(MatingPool,:);

    ParentDec = [Parent1Dec;Parent2Dec];
    [N,D]     = size(ParentDec);
    
    % Crossover
    k = rand(N/2,D) < 0.5;
    k(repmat(rand(N/2,1)>1,1,D)) = false;
    OffMask    = Parent1Mask;
    OffMask(k) = Parent2Mask(k);

    % Mutation
    for i = 1 : N/2
        if sum(OffMask(i,:))<=thetamid
            % Bit-flip mutation
            Site = rand(1,D) < 1/D;
            OffMask(i,Site) = ~OffMask(i,Site);
        else
            index1 = find(OffMask(i,:));
            maxN   = floor(size(index1,2) - thetamid);
            index  = index1(TSmax(-Fitness(index1),maxN));
            OffMask(i,index) = 0;
        end
    end

    % Crossover and mutation for dec
    if REAL
        OffDec = OperatorGAhalf(Problem,ParentDec);
    else
        OffDec = ones(N/2,D);
    end
end

function index = TSmax(Fitness,minN)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2,minN,Fitness);
    end
end
```

### `SCEA.m`
```matlab
classdef SCEA < ALGORITHM
% <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Sparsity clustering basec evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Zhang, C. Wu, Y. Tian, and X. Zhang. A co-evolutionary algorithm based
% on sparsity clustering for sparse large-scale multi-objective
% optimization. Engineering Applications of Artificial Intelligence, 2024,
% 133: 108194
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
            % Measure the importance of decision variables
            TDec    = [];
            TMask   = [];
            TempPop = [];
            Fitness = zeros(1,Problem.D);
            REAL    = ~strcmp(Problem.encoding,'binary');
            for i = 1 : 1+3*REAL
                if REAL
                    Dec = unifrnd(repmat(Problem.lower,Problem.D,1),repmat(Problem.upper,Problem.D,1));
                else
                    Dec = ones(Problem.D,Problem.D);
                end
                Mask       = eye(Problem.D);
                Population = Problem.Evaluation(Dec.*Mask);
                TDec       = [TDec;Dec];
                TMask      = [TMask;Mask];
                TempPop    = [TempPop,Population];
                Fitness    = Fitness + NDSort([Population.objs,Population.cons],inf);
            end
            % Calculate the current optimal sparsity
            if REAL
                VDec = unifrnd(repmat(Problem.lower,Problem.D,1),repmat(Problem.upper,Problem.D,1));
            else
                VDec = ones(Problem.D,Problem.D);
            end
            VMask = zeros(Problem.D,Problem.D);
            for i = 1 : Problem.D
                [~,rank1] = sort(Fitness);
                VMask(i,rank1(1:i)) = 1;
            end
            VPopulation = Problem.Evaluation(VDec.*VMask);
            [VPopulation,VDec,VMask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection([VPopulation,TempPop],[VDec;TDec],[VMask;TMask],Problem.N);
            popMask     = VMask(find(FrontNo ==1),:);
            thetaAll    = sum(popMask,2)';
            thetaUnique = unique(thetaAll);
            if mod(size(thetaUnique,2),2)==0
                minOdd = max(1,size(thetaAll,2)-1);
            else
                minOdd = max(1,size(thetaAll,2)-2);
            end
            [~,theta] = kmeans(thetaAll',minOdd);
            thetamid  = median(theta);
            
            % Generate initial population
            if REAL
                Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            else
                Dec = ones(Problem.N,Problem.D);
            end
            Mask = zeros(Problem.N,Problem.D);
            for i = 1 : Problem.N
                Mask(i,TournamentSelection(2,round(thetamid),Fitness)) = 1;
            end
            Population = Problem.Evaluation(Dec.*Mask);
            [Population,Dec,Mask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection([VPopulation,Population],[VDec;Dec],[VMask;Mask],Problem.N);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Calculate the current optimal sparsity
                winIndex    = find(FrontNo ==1);
                winPopMask  = Mask(find(FrontNo ==1),:);
                winTheta    = sum(winPopMask,2)';
                thetaUnique = unique(winTheta);
                if mod(size(thetaUnique,2),2)==0
                    minOdd = max(1,size(winTheta,2)-1);
                else
                    minOdd = max(1,size(winTheta,2)-2);
                end
                [~,theta] = kmeans(winTheta',minOdd);
                thetamid  = median(theta);
                
                loseIndex   = find(FrontNo ~=1);
                losePopMask = Mask(find(FrontNo ~=1),:);
                loseTheta   = sum(losePopMask,2)';
                
                loseIndex1 = loseIndex(find(loseTheta<=thetamid));
                loseIndex2 = loseIndex(find(loseTheta>thetamid));
                
                % Generate offspring
                [OffDec,OffMask] = Operator(Population,Dec,Mask,Fitness,winIndex,loseIndex1,loseIndex2,Problem,thetamid,REAL);
                Offspring        = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
            end
        end
    end
end
```

### `SPEA2_EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection(Population,Dec,Mask,N)
% Environmental selection

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
    
    %% Calculate the fitness of each solution
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    PopObj = Population.objs;
    fmax   = max(PopObj(FrontNo==1,:),[],1);
    fmin   = min(PopObj(FrontNo==1,:),[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);

    %% Environmental selection
    Last = find(FrontNo==MaxFNo);
    del  = Truncation(PopObj(Last,:),length(Last)-N+sum(Next));
    Next(Last(~del)) = true;
    % Population for next generation
    Population = Population(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
    FrontNo    = FrontNo(Next);
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
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
