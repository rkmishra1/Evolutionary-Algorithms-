# AGSEA

**Tags**: <2024> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>

## Description
Automated guiding vector selection-based evolutionary algorithm

## Reference
S. Shao, Y. Tian, and X. Zhang. Deep reinforcement learning assisted automated guiding vector selection for large-scale sparse multi-objective optimization. Swarm and Evolutionary Computation, 2024, 88: 101606.

## Source Code

### `AGSEA.m`
```matlab
classdef AGSEA < ALGORITHM
% <2024> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
% Automated guiding vector selection-based evolutionary algorithm

%------------------------------- Reference --------------------------------
% S. Shao, Y. Tian, and X. Zhang. Deep reinforcement learning assisted
% automated guiding vector selection for large-scale sparse multi-objective
% optimization. Swarm and Evolutionary Computation, 2024, 88: 101606.
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
            % Calculate the fitness of each decision variable        
            [Fitness1,TDec,TMask,TempPop]      = FitnessCal(Problem,5);
            [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection(TempPop,TDec,TMask,Problem.N); 
            num_feature = 14;
            max_act     = 3;
            inputn      = zeros(num_feature,1);
            outputn     = zeros(1,1);
            net         = newff(inputn,outputn,[10 10 10],{'tansig','purelin'},'trainlm');
            Memory      = [];
            action      = 1;
            Fitness3    = zeros(1,Problem.D);
            Memory      = UpdateMemory(Problem,Memory,action,Population,Mask,Population,Mask);
            clear TempPop TDec TMask;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)                
                MatingPool     = TournamentSelection(2,2*Problem.N,FitnessSpea2);
                LastPopulation = Population;
                LastMask = Mask;
                [action, Fitness,Fitness3] = UsingNet(Problem,Fitness1,net,Memory,num_feature,max_act,Mask,action,Fitness3);
                [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),Fitness,Mask,num_feature,Memory);
                Offspring        = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
                Memory = UpdateMemory(Problem,Memory,action,LastPopulation,LastMask,Population,Mask);
                net    = TrainNet(Problem,net,Memory,num_feature,max_act);                
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj)
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
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection(Population,Dec,Mask,N)
% The environmental selection of SPEA2

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
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Calculate the fitness of each solution
    Fitness = CalFitness(Population.objs);

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
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
    FitnessSpea2 = Fitness;
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

### `FitnessCal.m`
```matlab
function [Fitness,TDec,TMask,TempPop]  = FitnessCal(Problem,SampleNum)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    REAL    = any(Problem.encoding==1);
    TDec    = [];
    TMask   = [];
    TempPop = [];
    if REAL
        % Latin hypercube sampling
        Fitness = zeros(1,Problem.D);
        DecMat  = repmat((Problem.upper - Problem.lower),SampleNum,1).*lhsdesign(SampleNum,Problem.D) - repmat(Problem.lower,SampleNum,1);
        for i = 1 : SampleNum
            Dec  = repmat(DecMat(i,:),Problem.D,1);
            Mask = eye(Problem.D);
            Population = Problem.Evaluation(Dec.*Mask);
            TDec       = [TDec;Dec];
            TMask      = [TMask;Mask];
            TempPop    = [TempPop,Population];
            Fitness    = Fitness + sum(Population.objs,2)';
        end        
        % To reduce computation and support parallelism
        if Problem.D > 0
            AllSample = randperm(length(TempPop));
            FinalSample = AllSample(1:Problem.D);
            TempPop = TempPop(FinalSample);
            TDec = TDec(FinalSample,:);
            TMask = TMask(FinalSample,:);
        end
        Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
        Dec(:,Problem.encoding==4) = 1;
        Mask = false(Problem.N,Problem.D);
        for i = 1 : Problem.N
            Mask(i,TournamentSelection(2,ceil(rand*Problem.D),Fitness)) = 1;
        end
        Population = Problem.Evaluation(Dec.*Mask);
        TDec       = [TDec;Dec];
        TMask      = [TMask;Mask];
        TempPop    = [TempPop,Population];
    else
        Fitness = zeros(1,Problem.D);
        for i = 1
            Dec = ones(Problem.D,Problem.D);
            Mask       = eye(Problem.D);
            Population = Problem.Evaluation(Dec.*Mask);
            TDec       = [TDec;Dec];
            TMask      = [TMask;Mask];
            TempPop    = [TempPop,Population];
            Fitness = Fitness + sum(Population.objs,2)';
        end
        Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
        Dec(:,Problem.encoding==4) = 1;
        Mask = false(Problem.N,Problem.D);
        for i = 1 : Problem.N
            Mask(i,TournamentSelection(2,ceil(rand*Problem.D),Fitness)) = 1;
        end
        Population = Problem.Evaluation(Dec.*Mask);
        TDec       = [TDec;Dec];
        TMask      = [TMask;Mask];
        TempPop    = [TempPop,Population];
    end   
end
```

### `GLP_OperatorGAhalf.m`
```matlab
function [Offspring,outIndexList,chosengroups] = GLP_OperatorGAhalf(Problem,Parent1,Parent2,numberOfGroups)
% Parent1 and Parent2 are the matrix of decision variables, not solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [proC,disC,~,disM] = deal(1,20,1,20);
    [N,D] = size(Parent1);
    
    %% Genetic operators for real encoding
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    [outIndexList,~] = CreateGroups(numberOfGroups,Offspring,D); 
    chosengroups     = randi(numberOfGroups,size(outIndexList,1),1);
    Site = outIndexList == chosengroups;
    mu   = rand(N,1);
    mu   = repmat(mu,1,D);
    temp = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));    
end

function [outIndexArray,numberOfGroupsArray] = CreateGroups(numberOfGroups, xPrime, numberOfVariables)
% Creat groups by ordered grouping

    outIndexArray       = [];
    numberOfGroupsArray = [];
    noOfSolutions       = size(xPrime,1);
    for sol = 1 : noOfSolutions        
        varsPerGroup = floor(numberOfVariables/numberOfGroups);
        vars  = xPrime(sol,:);
        [~,I] = sort(vars);
        outIndexList = ones(1,numberOfVariables);
        for i = 1 : numberOfGroups-1
            outIndexList(I(((i-1)*varsPerGroup)+1:i*varsPerGroup)) = i;
        end
        outIndexList(I(((numberOfGroups-1)*varsPerGroup)+1:end)) = numberOfGroups;    
        outIndexArray = [outIndexArray;outIndexList];
        numberOfGroupsArray = [numberOfGroupsArray;numberOfGroups];    
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Fitness,Mask,num_feature,Memory)
% The operator of AGSEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [N,~]       = size(ParentDec);
    Parent1Dec  = ParentDec(1:floor(end/2),:);
    Parent2Dec  = ParentDec(floor(end/2)+1:floor(end/2)*2,:);
    Parent1Mask = ParentMask(1:floor(end/2),:);
    Parent2Mask = ParentMask(floor(end/2)+1:floor(end/2)*2,:);
    
    VaryGroup = kmeans(Fitness',2)';
    MaxGroup  = max(VaryGroup);  

    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        [OffDec,~,~] = GLP_OperatorGAhalf(Problem,Parent1Dec,Parent2Dec,4);	% 4 -- numberofgroups
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(size(Parent1Dec));
    end

    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        SelectedGroup = randi(MaxGroup,1);
        index = xor(Parent1Mask(i,:),Parent2Mask(i,:));     
        if rand < 0.5
            index = (SelectedGroup == VaryGroup) & index & rand(1,Problem.D) < 1;           
            OffMask(i,index) = 0;
        else
            index = (SelectedGroup == VaryGroup) & index & rand(1,Problem.D) < 1;           
            OffMask(i,index) = 1;
        end       
    end
    for i = 1 : N/2     
        if rand < 0.5
            index =  rand(1,Problem.D) < (100*mean(Mask,'all'))/(Problem.D*length(unique(Memory(max(end-10,1):end,num_feature)))^2);           
            OffMask(i,index) = 0;
        else
            index =  rand(1,Problem.D) < (100*mean(Mask,'all'))/(Problem.D*length(unique(Memory(max(end-10,1):end,num_feature)))^2);           
            OffMask(i,index) = 1;
        end       
    end
end
```

### `TrainNet.m`
```matlab
function net = TrainNet(Problem,net,Memory,num_feature,max_act)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    inputn  = Memory(:,1:num_feature)';
    outputn = Memory(:,num_feature+1)';
    net.trainParam.epochs = 100;    
    net.trainParam.lr     = 0.001;         
    net.trainParam.goal   = 0.001;       
    net.trainParam.showWindow = 0;
    output = zeros(1,max_act);
    alpha  = 0.1;
    
    if size(Memory,1) == ceil(0.25*Problem.maxFE/100)
        net = newff(inputn,outputn,[10 10 10],{'tansig','purelin'},'trainlm'); 
    end
    if size(Memory,1) > ceil(0.25*Problem.maxFE/100) && mod(size(Memory,1),10) == 0
        MTrain = Memory(end-9:end,:);
        for i = 1 : size(MTrain,1)
            for j = 1 : max_act
                input = [MTrain(i,(num_feature+2):end) j]';
                output(j) = sim(net,input);
            end
            MTrain(i,num_feature+1) = MTrain(i,num_feature+1) + alpha*max(output);
        end
        inputn  = MTrain(:,1:num_feature)';
        outputn = MTrain(:,num_feature+1)';
        net     = train(net,inputn,outputn);
    end
end
```

### `UpdateMemory.m`
```matlab
function Memory = UpdateMemory(Problem,Memory,action,LastPopulation,LastMask,Population,Mask)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    LastPopulationHV = HV(LastPopulation,max([LastPopulation.objs;Population.objs]));
    PopulationHV     = HV(Population,max([LastPopulation.objs;Population.objs]));
    reward           = (PopulationHV - LastPopulationHV)/LastPopulationHV;
    
    LastSparse = mean(LastMask(:));
    LastStd    = std(sum(LastMask,2));
    LastSparseDistuibution = histcounts(sum(LastMask,2)./size(LastMask,2),0:0.1:1)./size(LastMask,2);
    LastState  = [LastSparse,LastStd,LastSparseDistuibution,Problem.FE/Problem.maxFE];

    CurrentSparse = mean(Mask(:));
    CurrentStd    = std(sum(LastMask,2));
    CurrentSparseDistuibution = histcounts(sum(Mask,2)./size(Mask,2),0:0.1:1)./size(Mask,2);
    CurrentState  = [CurrentSparse,CurrentStd,CurrentSparseDistuibution,Problem.FE/Problem.maxFE];
    
    NewMemory = [LastState,action,reward,CurrentState];
    Memory    = [Memory;NewMemory];
end
```

### `UsingNet.m`
```matlab
function [action, Fitness, Fitness3] = UsingNet(Problem,Fitness1,net,Memory,num_feature,max_act,Mask,action,Fitness3)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    output = zeros(1,max_act);
    if size(Memory,1) == 1      
    elseif rand < 0.3^(Problem.FE/Problem.maxFE) || size(Memory,1) < ceil(0.25*Problem.maxFE/100)
        action = randi([1 max_act],1);       
    else
        CurrentState = Memory(end,num_feature+2:end);
        for i = 1 : max_act
            input     = [CurrentState i]';
            output(i) = sim(net,input);
        end
        [~,action] = max(output);
    end     
    switch action
        case 1
            Fitness = Fitness1;
        case 2
            Fitness = sum(Mask==0);
            if size(Mask,1) == 1
                Fitness = Mask;
            end
        case 3
            Fitness = rand(1,Problem.D);
    end
end
```
