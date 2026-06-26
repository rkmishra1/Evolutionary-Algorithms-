# NUCEA

**Tags**: <2023> <multi> <real/binary> <large/none> <constrained/none> <sparse>

## Description
Non-uniform clustering based evolutionary algorithm

## Reference
S. Shao, Y. Tian, and X. Zhang. A non-uniform clustering based evolutionary algorithm for solving large-scale sparse multi-objective optimization problems. Proceedings of the 18th International Conference on Bio-inspired Computing: Theories and Applications, 2023.

## Source Code

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
    [~,uni]    = unique(Population.objs,'rows');
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
    Population   = Population(Next);
    Fitness      = Fitness(Next);
    Dec          = Dec(Next,:);
    Mask         = Mask(Next,:);
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
    chosengroups = randi(numberOfGroups,size(outIndexList,1),1);
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
        outIndexArray       = [outIndexArray;outIndexList];
        numberOfGroupsArray = [numberOfGroupsArray;numberOfGroups];    
    end
end
```

### `NUCEA.m`
```matlab
classdef NUCEA < ALGORITHM
% <2023> <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Non-uniform clustering based evolutionary algorithm

%------------------------------- Reference --------------------------------
% S. Shao, Y. Tian, and X. Zhang. A non-uniform clustering based
% evolutionary algorithm for solving large-scale sparse multi-objective
% optimization problems. Proceedings of the 18th International Conference
% on Bio-inspired Computing: Theories and Applications, 2023.
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
            TDec    = [];
            TMask   = [];
            TempPop = [];
            Fitness = zeros(1,Problem.D);
            for i = 1 : 1+4*any(Problem.encoding~=4)
                Dec = unifrnd(repmat(Problem.lower,Problem.D,1),repmat(Problem.upper,Problem.D,1));
                Dec(:,Problem.encoding==4) = 1;
                Mask       = eye(Problem.D);
                Population = Problem.Evaluation(Dec.*Mask);
                TDec       = [TDec;Dec];
                TMask      = [TMask;Mask];
                TempPop    = [TempPop,Population];
                Fitness    = Fitness + NDSort([Population.objs,Population.cons],inf);
            end
            % Generate initial population
            Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            Dec(:,Problem.encoding==4) = 1;
            Mask = false(Problem.N,Problem.D);
            for i = 1 : Problem.N
                Mask(i,TournamentSelection(2,ceil(rand*Problem.D),Fitness)) = 1;
            end
            Population = Problem.Evaluation(Dec.*Mask);
            [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection([Population,TempPop],[Dec;TDec],[Mask;TMask],Problem.N);
     
            %% Optimization
            while Algorithm.NotTerminated(Population)                
                MatingPool       = TournamentSelection(2,2*Problem.N,FitnessSpea2);
                [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),Fitness,Mask);
                Offspring        = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
            end
        end
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Fitness,Mask)
% The operator of NUCEA

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
    
    %% Clustering
    [~, index] = sort(Fitness);
    GroupSize  = ceil(mean(Mask,'all')*Problem.D);
    VaryGroup  = ones(1,Problem.D);
    start      = 1;
    GroupI     = 1;
    while true
        startend       = min(start+GroupI*GroupSize-1,Problem.D);
        ObjectiveIndex = start:startend;
        VaryGroup((index(ObjectiveIndex))) = GroupI;
        GroupI = GroupI + 1;
        start  = startend + 1;
        if start>Problem.D
            break;
        end
    end
    MaxGroup = max(VaryGroup);  

    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        [OffDec,groupIndex,chosengroups] = GLP_OperatorGAhalf(Problem,Parent1Dec,Parent2Dec,4);	% 4 -- numberofgroups
        OffDec(:,Problem.encoding==4)    = 1;
    else
        OffDec = ones(size(Parent1Dec));
    end

    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        SelectedGroup = randi(MaxGroup,1);
        index = xor(Parent1Mask(i,:),Parent2Mask(i,:));     
        if rand < 0.5
            index = (SelectedGroup == VaryGroup) & index;           
            OffMask(i,index) = 0;
        else
            index = (SelectedGroup == VaryGroup) & index;           
            OffMask(i,index) = 1;
        end       
    end    
    if any(Problem.encoding~=4) && SelectedGroup < MaxGroup      
        chosenindex = groupIndex == chosengroups;
        for i = 1 : N/2            
            if rand < 0.5
                index = find(OffMask(i,:)&chosenindex(i,:));
                index = index(TS(-Fitness(index)));
                OffMask(i,index) = 0;
            else
                index = find(~OffMask(i,:)&chosenindex(i,:));
                index = index(TS(Fitness(index)));
                OffMask(i,index) = 1;
            end            
        end                    
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
```

### `UpdateMemory.m`
```matlab
function Memory = UpdateMemory(Memory,action,LastPopulation,LastMask,Population,Mask)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    LastPopulationHV = HV(LastPopulation,[3,3]);
    PopulationHV     = HV(Population,[3,3]);
    reward           = (PopulationHV - LastPopulationHV)/LastPopulationHV;
    LastSparse       = mean(LastMask(:));
    LastStd          = std(sum(LastMask,2));
    LastSparseDistuibution = histcounts(sum(LastMask,2)./size(LastMask,2),0:0.1:1)./size(LastMask,2);
    LastState        = [LastSparse,LastStd,LastSparseDistuibution];

    CurrentSparse = mean(Mask(:));
    CurrentStd    = std(sum(LastMask,2));
    CurrentSparseDistuibution = histcounts(sum(Mask,2)./size(Mask,2),0:0.1:1)./size(Mask,2);
    CurrentState  = [CurrentSparse,CurrentStd,CurrentSparseDistuibution];
    
    NewMemory = [LastState,action,reward,CurrentState];
    Memory    = [Memory;NewMemory];
end
```
