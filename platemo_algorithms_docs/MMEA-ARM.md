# MMEA-ARM

**Tags**: <2026> <multi> <real/integer/binary> <large/none> <multimodal> <sparse>

## Description
Adaptive resource management based MMEA

## Reference
S. Shao, Y. Tian, and Y. Zhang. An adaptive resource management based evolutionary algorithm for large-scale multi-modal multi-objective optimization. Applied Soft Computing, 2026, 193: 114853.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,N,dis)
% The environmental selection of MMEA-ARM

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    if nargin > 4
        PopObj = [Population.objs,dis];
    else
        PopObj = Population.objs;
    end

    %% Delete duplicated solutions
    [~,uni]    = unique(PopObj,'rows');
    PopObj     = PopObj(uni,:);
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(PopObj,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObj,FrontNo);
    
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
        vars         = xPrime(sol,:);
        [~,I]        = sort(vars);
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

### `MMEAARM.m`
```matlab
classdef MMEAARM < ALGORITHM
% <2026> <multi> <real/integer/binary> <large/none> <multimodal> <sparse>
% Adaptive resource management based MMEA

%------------------------------- Reference --------------------------------
% S. Shao, Y. Tian, and Y. Zhang. An adaptive resource management based
% evolutionary algorithm for large-scale multi-modal multi-objective
% optimization. Applied Soft Computing, 2026, 193: 114853.
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
            Dec  = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1)); 
            Mask = rand(Problem.N, Problem.D) < 0.5;
            Population  = Problem.Evaluation(Dec.*Mask);
            K           = 10;  
            Masks       = cell(1,K);
            Decs        = cell(1,K);
            Populations = cell(1,K);
            GV          = cell(1,K);
            FrontNo     = cell(1,K);
            CrowdDis    = cell(1,K);
            index       = randperm(floor(Problem.N/K)*K);
            temp        = reshape(index,K,floor(Problem.N/K));
            for i = 1 : K
                Populations{i} = Population(temp(i,:));
                Masks{i}       = Mask(temp(i,:),:);
                Decs{i}        = Dec(temp(i,:),:);
                [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},length(Populations{i}));
                GV{i}          = UpdateGV(zeros(1,Problem.D),Masks{i},FrontNo{i});
            end
            endingFlag = 0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                LastPopulation = [Populations{:}];              
                allSolutions   = [Populations{:}];
                ref            = max(allSolutions.objs);
                perHV          = zeros(1, K);
                for i = 1 : K
                    perHV(i) = HV(Populations{i}, ref);
                end
                minPopNum = max(2,floor(Problem.N/K*(Problem.FE/Problem.maxFE)));
                resources = allocateResourcesWithMinimum(perHV, Problem.N, minPopNum);
                for i = 1 : K                
                    GV{i}            = UpdateGV(GV{i},Masks{i},FrontNo{i});
                    Mating           = TournamentSelection(2,2*resources(i),FrontNo{i},-CrowdDis{i});
                    [OffDec,OffMask] = Operator(Problem,Decs{i}(Mating,:),Masks{i}(Mating,:),GV{i});
                    Offspring        = Problem.Evaluation(OffDec.*OffMask);
                    Populations{i}   = [Populations{i},Offspring];
                    Decs{i}          = [Decs{i};OffDec];
                    Masks{i}         = [Masks{i};OffMask];
                    if i > 1 && Problem.FE/Problem.maxFE > 1
                        for j = 1 : i-1
                            [~,fs(j)] = min(mean(Populations{j}.objs,2));
                        end
                        R = zeros(1,Problem.D);
                        for j = 1 : i-1
                            R = R + Masks{j}(fs(j),:);
                        end
                        R(R>0) = 1;
                        dis = sum(repmat(R,length(Populations{i}),1)&Masks{i},2);
                        [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},floor(Problem.N/K),dis);
                    else
                        [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},floor(Problem.N/K));
                    end
                end
                 
                CurrentPopulation = [Populations{:}];              
                globalSS          = simility(sum(LastPopulation.decs~=0)>mean(sum(LastPopulation.decs~=0)),sum(CurrentPopulation.decs~=0)>mean(sum(CurrentPopulation.decs~=0)));
                if globalSS >= 1
                    divisionFlag = 1;
                else
                    divisionFlag = 0;
                end
               
                if mod(ceil(Problem.FE/Problem.N),20) == 0
                    [ss,index] = SubPopSimility(Populations,Masks);
                    if ss > 0.5
                        K = K-1;
                        i = index(1);
                        j = index(2);
                        [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection([Populations{i},Populations{j}],[Decs{i};Decs{j}],[Masks{i};Masks{j}],floor(Problem.N/K));
                        Populations(j) = [];
                        Decs(j)        = [];
                        Masks(j)       = [];
                        GV(j)          = [];
                        endingFlag     = endingFlag + 1;
                    elseif divisionFlag == 1
                        for i = 1 : K
                            [Populations{i},Decs{i},Masks{i},FrontNo{i},CrowdDis{i}] = EnvironmentalSelection(Populations{i},Decs{i},Masks{i},floor(Problem.N/(K+1)));
                        end
                        K    = K + 1;
                        Dec  = unifrnd(repmat(Problem.lower,floor(Problem.N/K),1),repmat(Problem.upper,floor(Problem.N/K),1));
                        Mask = zeros(floor(Problem.N/K),Problem.D);
                        F    = zeros(1,Problem.D);
                        for i = 1: K-1
                            F = F + GV{i};
                        end
                        for i = 1 : floor(Problem.N/K)
                            Mask(i,TournamentSelection(2,floor(rand*Problem.D),F)) = 1;
                        end
                        Populations{K} = Problem.Evaluation(Dec.*Mask);
                        Masks{K}       = Mask;
                        Decs{K}        = Dec;
                        GV{K}          = zeros(1,Problem.D);
                        [Populations{K},Decs{K},Masks{K},FrontNo{K},CrowdDis{K}] = EnvironmentalSelection(Populations{K},Decs{K},Masks{K},length(Populations{K}));
                        GV{K}          = UpdateGV(zeros(1,Problem.D),Masks{K},FrontNo{K});
                    end
                end
                Population = [Populations{:}];
            end
        end
    end
end


function resources = allocateResourcesWithMinimum(perHV, N, minResources)    
    M = length(perHV); % Number of subpopulations
    reserved    = minResources * ones(1, M);
    remaining_N = N - sum(reserved);
    % Check if the total resources are sufficient
    if remaining_N < 0
        error('Total resources are insufficient to guarantee the minimum allocation.');
    end
    epsilon = 1e-6; % Small value to avoid division by zero
    normHV  = (perHV - min(perHV)) / (max(perHV) - min(perHV) + epsilon);   
    invHV   = 1 - normHV;   
    weights = invHV / sum(invHV);
    remainingResources = round(remaining_N * weights);
    diff = remaining_N - sum(remainingResources);
    if diff ~= 0
        [~, idx] = max(weights); % Assign the difference to the largest weight
        remainingResources(idx) = remainingResources(idx) + diff;
    end
    resources = reserved + remainingResources;
end

function s = simility(subPop1,subPop2)
	s = sum(subPop1&subPop2)/min(sum(subPop1),sum(subPop2));
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,Fitness)
% The operator of MMEA-ARM

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
    GroupSize  = ceil(mean(ParentMask,'all')*Problem.D);
    VaryGroup  = ones(1,Problem.D);
    start      = 1;
    GroupI     = 1;
    while true
        startend = min(start+GroupI*GroupSize-1,Problem.D);
        ObjectiveIndex = start:startend;
        VaryGroup((index(ObjectiveIndex))) = GroupI;
        start = startend + 1;
        if start>Problem.D
            break
        end
    end
    MaxGroup = max(VaryGroup);  

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
            index = (SelectedGroup == VaryGroup) & index;           
            OffMask(i,index) = 0;
        else
            index = (SelectedGroup == VaryGroup) & index;         
            OffMask(i,index) = 1;
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

### `SubPopSimility.m`
```matlab
function [ss,index] = SubPopSimility(Populations,Masks)
% Calculate the similarity between subpopulations

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    K     = length(Populations);
    ss    = 0;
    index = [];
    for i = 1 : K-1
        for j = i+1 : K
            if simility(sum(Masks{i})>mean(sum(Masks{i})),sum(Masks{j})>mean(sum(Masks{j}))) > ss
                ss    = simility(sum(Masks{i})>mean(sum(Masks{i})),sum(Masks{j})>mean(sum(Masks{j})));
                index = [i,j];
            end
        end
    end
end

function s = simility(subPop1,subPop2)
	s = sum(subPop1&subPop2)/min(sum(subPop1),sum(subPop2));
end
```

### `UpdateGV.m`
```matlab
function gv = UpdateGV(~,Mask,FrontNo)
% Update the guiding vectors

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Mask = Mask(FrontNo==1,:);
    gv   = sum(Mask, 1)./size(Mask,1);
end
```
