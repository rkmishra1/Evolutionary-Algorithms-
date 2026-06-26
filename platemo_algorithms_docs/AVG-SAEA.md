# AVG-SAEA

**Tags**: <2025> <multi> <real/integer> <expensive> <large>

## Description
Adaptive variable grouping based surrogate-assisted evolutionary algorithm

## Reference
Y. Li, X. Feng, and H. Yu. Solving high-dimensional expensive multiobjective optimization problems by adaptive decision variable grouping. IEEE Transactions on Evolutionary Computation, 2025, 29(4): 1041-1054.

## Source Code

### `AVGSAEA.m`
```matlab
classdef AVGSAEA< ALGORITHM
% <2025> <multi> <real/integer> <expensive> <large>
% Adaptive variable grouping based surrogate-assisted evolutionary algorithm
% Numtrain --- 300 --- Number of train samples
% wmax     ---  20 --- Number of generations before updating models
% NumEsp   ---   2 --- Number of subPopulations
% mu       ---   5 --- Number of re-evaluated solutions in each iteration
 
%------------------------------- Reference --------------------------------
% Y. Li, X. Feng, and H. Yu. Solving high-dimensional expensive
% multiobjective optimization problems by adaptive decision variable
% grouping. IEEE Transactions on Evolutionary Computation, 2025, 29(4):
% 1041-1054.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yingwei Li

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [Numtrain,wmax,NumEsp,mu] = Algorithm.ParameterSet(300,20,5,5);
            
            %% Initalize the population by Latin hypercube sampling
            PopDec     = UniformPoint(Numtrain,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Numtrain,1).*PopDec+repmat(Problem.lower,Numtrain,1));
            Arc        = Population;
            subDec     = cell(1,NumEsp);
            trainDecs  = Population.decs;
            trainObjs  =  Population.objs;
            nor_Index  = [];
            flag       = 1;
            psset      = cell(NumEsp,Problem.M);
            qsset      = cell(NumEsp,Problem.M);
            modelset   = cell(NumEsp,Problem.M);
            PopObj     = cell(1,NumEsp);
            PopCon     = cell(1,NumEsp);
            muDec      = cell(1,NumEsp);

            %% Optimization
            while Algorithm.NotTerminated(Arc)
                % Adaptive variable grouping
                [better_Cpop,bad_Cpop] = DimSelect(Arc,floor(length(Arc)/4));
                CbetterDec             = better_Cpop.decs; 
                CbadDec                = bad_Cpop.decs;
                mean_betterCDec        = mean(CbetterDec);
                mean_badCDec           = mean(CbadDec);
                difference             = abs(mean_betterCDec - mean_badCDec); 
                [~, diff_idx]          = sort(difference, 'descend'); 
                varsPerGroup           = floor(Problem.D/NumEsp);
                for i = 1 : NumEsp-1
                    nor_Index = [nor_Index,ones(1,varsPerGroup).*i];
                end
                nor_Index = [nor_Index, ones(1,Problem.D-size(nor_Index,2)).*NumEsp];
                for k = 1 : Problem.D
                    change_Index(diff_idx(k)) = nor_Index(k);
                end
                for i = 1 : NumEsp
                    trainsubDec{i} = trainDecs(:,change_Index==i);
                end
                
                %% Train models
                trainLabel = trainObjs;
                 for i = 1 : NumEsp
                     for m = 1 : Problem.M
                        [Input,ps]    = mapminmax(trainsubDec{i}',0,1);
                        Input         = Input';
                        [Output,qs]   = mapminmax(trainLabel(:,m)',0,1);
                        Output        = Output';
                        dmodel        = newrbe(Input',Output', 1 );
                        modelset{i,m} = dmodel;
                        psset{i,m}    = ps;
                        qsset{i,m}    = qs;
                    end
                 end
                [Pop,~] = EnvironmentalSelection(Arc,Problem.N);
                PopDec  = Pop.decs;
                for i = 1 : NumEsp
                     subDec{i} = PopDec(:,change_Index==i);
                     PopObj{i} = Pop.objs;
                end

                %% Evolution of subpopulations
                w = 1;
                if flag == 2 
                    % Diversity-oriented environmental selection
                    while w <= wmax
                        drawnow('limitrate');
                        for j = 1 : NumEsp
                            PopCon{j}  = calCon(PopObj{j});
                            MatingPool = TournamentSelection(2,ceil(Problem.N/2)*2,PopCon{j});
                            OffDec     = OperatorG(Problem,subDec{j}(MatingPool,:),change_Index,j);  
                            N          = size(OffDec,1);
                            OffObj     = zeros(N,Problem.M);
                                for m = 1 : Problem.M
                                    normDec       = mapminmax('apply',OffDec',psset{j,m});
                                    [normPre,~,~] = sim(modelset{j,m},normDec);
                                    OffObj(:,m)   = (mapminmax('reverse',normPre,qsset{j,m}))';
                                end
                            [PopObj{j},subDec{j}] = DVsetEnvironmentalSelection([PopObj{j};OffObj],[subDec{j};OffDec],N);
                        end
                        w = w + 1;
                    end

                else
                    % Convergence-oriented environmental selection
                    while w <= wmax
                        drawnow('limitrate');
                        for i = 1 : NumEsp
                            PopCon{i}  = calCon(PopObj{i});
                            MatingPool = TournamentSelection(2,ceil(Problem.N/2)*2,PopCon{i});
                            OffDec     = OperatorG(Problem,subDec{i}(MatingPool,:),change_Index,i);  
                            N          = size(OffDec,1);
                            OffObj     = zeros(N,Problem.M);
                            for m = 1 : Problem.M
                                normDec       = mapminmax('apply',OffDec',psset{i,m});
                                [normPre,~,~] = sim(modelset{i,m},normDec);
                                OffObj(:,m)   = (mapminmax('reverse',normPre,qsset{i,m}))';
                            end
                            allCon  = calCon([PopObj{i};OffObj]);
                            Con     = allCon(1:N);
                            newCon  = allCon(N+1:end); 
                            updated = Con > newCon;
                            subDec{i}(updated,:) = OffDec(updated,:);
                            PopObj{i}(updated,:) = OffObj(updated,:);
                        end
                        w = w + 1;
                    end
                end

                %% Merge variables in each group into complete solution
                if flag == 1
                    for i = 1 : NumEsp   
                           betterIndex = SubEnvironmentalSelection(PopObj{i},mu);
                           muDec{i}    = subDec{i}(betterIndex,:);
                           NewDec(:,change_Index==i) = muDec{i};
                    end
                elseif flag == 2
                    for j = 1 : NumEsp   
                           [~,muDec{j}] = DSelectNew(PopObj{j},subDec{j},mu);
                           NewDec(:,change_Index==j) = muDec{j};
                    end
                    flag = 1;
                else
                    finalMergeDec = zeros(Problem.N,Problem.D); 
                    for i = 1 : NumEsp
                        finalMergeDec(:,change_Index==i) = subDec{i};
                    end
                    N = size(PopObj{1},1);
                    stdSamples = zeros(N,Problem.M);
                    for m = 1 : Problem.M
                        Samples = [];
                        for i = 1 : NumEsp   
                            Samples = [Samples,PopObj{i}(:,m)];
                        end
                        stdSamples(:,m) = std(Samples,0,2);
                    end
                    meanstd     = mean(stdSamples,2);
                    [~,std_Idx] = sort(meanstd,"descend");
                    NewDec      = finalMergeDec(std_Idx(1:mu),:);
                    flag        = 1;
                end

                %% Select mu new samples
                New = Problem.Evaluation(NewDec);
                lastPopulation = Arc;
                Arc = [Arc,New]; 
                if flag == 0
                    flag = 1;
                else
                    flag = CalFlag(Arc,lastPopulation);
                end

                %% Select train data for the next iteration
                [trainDecs, trainObjs] = SelectTrainData(Arc, Numtrain, length(New));
            end
        end
    end
end

function Con = calCon(PopObj)
% Calculate the convergence of each solution

    FrontNo = NDSort(PopObj,inf);
    Con     = sum(PopObj,2);
    Con     = FrontNo'*(max(Con)-min(Con)) + Con;
end
```

### `CalFitness.m`
```matlab
function [Convergence, Diversity, Fitness] = CalFitness(PopObj)
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

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
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
    Convergence = R;
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    Diversity = (Distance(:,floor(sqrt(N)))+2)';
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFlag.m`
```matlab
function flag = CalFlag(Population,lastPopulation)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yingwei Li

    FrontNo      = NDSort(Population.objs,inf);
    lastFrontNo  = NDSort(lastPopulation.objs,inf);
    PopObj       = Population.objs;
    lastPopObj   = lastPopulation.objs;
    NDPopObj     = Population(FrontNo==1).objs;
    lastNDPopObj = lastPopulation(lastFrontNo==1).objs;
    lastfmax     = max(lastPopObj(lastFrontNo==1,:),[],1);
    lastfmin     = min(lastPopObj(lastFrontNo==1,:),[],1);
    NormlastND   = (lastNDPopObj-repmat(lastfmin,size(lastNDPopObj,1),1))./repmat(lastfmax-lastfmin,size(lastNDPopObj,1),1);
    lastCon      = sum(NormlastND,2);
    NormND       = (NDPopObj-repmat(lastfmin,size(NDPopObj,1),1))./repmat(lastfmax-lastfmin,size(NDPopObj,1),1);
    Con          = sum(NormND,2);
    HVsum        = CalHV(NDPopObj);
    lastHVsum    = CalHV(lastNDPopObj);
    if lastHVsum < HVsum 
        if min(Con) < min(lastCon)
            flag = 1;
        else
            flag = 2;
        end
    else
        flag = 0;
    end
end

function Score = CalHV(PopObj)
% Calculate the estimated HV value

    RefPoint  = max(PopObj,[],1)*1.1;
    PopObj(any(PopObj>repmat(RefPoint,size(PopObj,1),1),2),:) = [];
    SampleNum = 10000;
    MaxValue  = RefPoint;
    MinValue  = min(PopObj,[],1);
    Samples   = unifrnd(repmat(MinValue,SampleNum,1),repmat(MaxValue,SampleNum,1));
    Domi      = false(1,SampleNum);
    for i = 1 : size(PopObj,1)
        Domi(all(repmat(PopObj(i,:),SampleNum,1)<=Samples,2)) = true;
    end
    Score = prod(MaxValue-MinValue)*sum(Domi)/SampleNum;
end
```

### `DSelectNew.m`
```matlab
function [Nextobj,Nextdec] = DSelectNew(Popobj,Popdec,N)
% The environmental selection of distribution optimization in LMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Choose = Truncation(Popobj,N);

    % Population for next generation
    Nextobj = Popobj(Choose,:);
    Nextdec = Popdec(Choose,:);
end

function Choose = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Calculate the normalized angle between each two solutions
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);
    Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    Cosine(logical(eye(length(Cosine)))) = 0;
    
    %% Truncation
    % Choose the extreme solutions first
    Choose = false(1,size(PopObj,1)); 
    [~,extreme] = max(PopObj,[],1);
    Choose(extreme) = true;
    % Choose the rest by truncation
    if sum(Choose) > K
        selected = find(Choose);
        Choose   = selected(randperm(length(selected),K));
    else
        while sum(Choose) < K
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
    end
end
```

### `DVsetEnvironmentalSelection.m`
```matlab
function [Nextobj,Nextdec] = DVsetEnvironmentalSelection(Popobj,Popdec,N)
% The environmental selection of distribution optimization in LMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Popobj,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = Truncation(Popobj(Last,:),N-sum(Next));
    Next(Last(Choose)) = true;
    % Population for next generation
    Nextobj = Popobj(Next,:);
    Nextdec = Popdec(Next,:);
end

function Choose = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Calculate the normalized angle between each two solutions
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);
    Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    Cosine(logical(eye(length(Cosine)))) = 0;
    
    %% Truncation
    % Choose the extreme solutions first
    Choose = false(1,size(PopObj,1)); 
    [~,extreme] = max(PopObj,[],1);
    Choose(extreme) = true;
    % Choose the rest by truncation
    if sum(Choose) > K
        selected = find(Choose);
        Choose   = selected(randperm(length(selected),K));
    else
        while sum(Choose) < K
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
    end
end
```

### `DimSelect.m`
```matlab
function [better_Cpop,bad_Cpop] = DimSelect(Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yingwei Li

    PopObj          = Population.objs;
    [Convergence,~] = CalFitness(PopObj);
    [~,Cidx]        = sort(Convergence);
    better_Cpop     = Population(Convergence==0);
    temple          = length(Convergence)-N+1;
    bad_Cpop        = Population(Cidx(temple:end));
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II

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
end
```

### `MergeSubpop.m`
```matlab
function [MergeDec] = MergeSubpop(eachDec,Index,cycle,NumEsp,mu,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yingwei Li

    MergeDec = zeros(mu,Problem.D);
    for j = 1 : cycle
        for i = 1 : NumEsp  
            MergeDec(:,Index{j}==i) = eachDec{i}; 
        end
    end
end
```

### `OperatorG.m`
```matlab
function Offspring = OperatorG(Problem,Parent,Index,curgroup)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
   
    [proC,disC,proM,disM] = deal(1,20,1,20);
    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        Parent    = Parent.decs;
    else
        evaluated = false;
    end
    Parent1   = Parent(1:floor(end/2),:);
    Parent2   = Parent(floor(end/2)+1:floor(end/2)*2,:);
    Offspring = zeros(2*size(Parent1,1),size(Parent1,2));
    if any(Problem.encoding~=4)   % Real and integer variables
        Offspring = RealCrossover(Parent1,Parent2,disC,proC);
        Offspring = RealMutation(Offspring,Problem.lower,Problem.upper,proM,disM,Index,curgroup);
    end
   
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end

% Genetic operators for real and integer variables
function Offspring = RealCrossover(Parent1,Parent2,disC,proC)
    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
end
             
function Offspring = RealMutation(Offspring,lower,upper,proM,disM,Index,curgroup)   
    %% Polynomial mutation
    [N,D] = size(Offspring);
    subLower = lower(Index == curgroup);
    subUpper = upper(Index == curgroup);
    Lower = repmat(subLower,N,1);
    Upper = repmat(subUpper,N,1);
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

% This function is written by Yingwei Li

    tr_y  = P.objs;
    NA    = size(tr_y,1);
    Range = [min(tr_y,[],1);max(tr_y,[],1)];

    tr_y   = tr_y - repmat(Range(1,:),NA,1);
    Cosine = 1 - pdist2(tr_y,tr_y,'cosine');%NA*NA
    Cosine(logical(eye(size(Cosine,1)))) = 0;
    Choose = [false(1,NA-N2),true(1,N2)];

    while sum(Choose) < N1
        unSelected = find(~Choose);
        [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
        Choose(unSelected(x)) = true;
    end

    tr_yy = tr_y(Choose,:);
    tr_x  = P.decs;
    tr_xx = tr_x(Choose,:);
end
```

### `SubEnvironmentalSelection.m`
```matlab
function Next = SubEnvironmentalSelection(PopObj,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yingwei Li

    %% Non-dominated sorting
    Next = false(size(PopObj,1),1);
    Con  = calCon(PopObj);
    [~,Rank] = sort(Con);
    Next(Rank(1:N)) = true;    
end

function Con = calCon(PopObj)
% Calculate the convergence of each solution

    FrontNo = NDSort(PopObj,inf);
    Con     = sum(PopObj,2);
    Con     = FrontNo'*(max(Con)-min(Con)) + Con;
end
```
