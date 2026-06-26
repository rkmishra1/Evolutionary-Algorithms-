# DGEA

**Tags**: <2022> <multi/many> <real/integer> <large/none>

## Description
Direction guided evolutionary algorithm

## Reference
C. He, R. Cheng, and D. Yazdani. Adaptive offspring generation for evolutionary large-scale multiobjective optimization. IEEE Transactions on System, Man, and Cybernetics: Systems, 2022, 52(2): 786-798.

## Source Code

### `DGEA.m`
```matlab
classdef DGEA < ALGORITHM
% <2022> <multi/many> <real/integer> <large/none>
% Direction guided evolutionary algorithm
% operation ---   1 --- Operation of the environmental selection
% RefNo     ---  10 --- Number of reference vectors for offspring generation

%------------------------------- Reference --------------------------------
% C. He, R. Cheng, and D. Yazdani. Adaptive offspring generation for
% evolutionary large-scale multiobjective optimization. IEEE Transactions
% on System, Man, and Cybernetics: Systems, 2022, 52(2): 786-798.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            [operation, RefNo]= Algorithm.ParameterSet(1, 10);
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();
            Offspring     = Problem.Initialization();
            Arc           = Population;

            %% Optimization
            while Algorithm.NotTerminated(Arc)
                [Population,FrontNo] = PreSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2,RefNo);
                Offspring = DirectionReproduction(Problem,Population,FrontNo,RefNo);
                switch operation
                    case 1 
                        Arc = subRVEA([Arc,Offspring],V,(Problem.FE/Problem.maxFE)^2);
                    case 2
                        Arc = subNSGAII([Arc,Offspring],Problem.N);
                    case 3 
                        Arc = subIBEA([Arc,Offspring],Problem.N,0.05);
                    case 4 
                        Arc = subSPEA2([Arc,Offspring],Problem.N);
                    otherwise
                        Arc = [Population,Offspring];	% Without selection
                end
            end
        end
    end
end
```

### `DirectionReproduction.m`
```matlab
function Offspring = DirectionReproduction(Problem,Population,FrontNo,RefNo)
% The direction based offspring generation in DGEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

	[N,D]   = size(Population.decs);
    indexD  = find(FrontNo>1);
    domiN   = numel(indexD);
    indexN  = FrontNo == 1;
	nondec  = Population(indexN).decs;
    domidec = Population(indexD).decs;  
	subN    = max([floor(Problem.N/RefNo),10]);
    Lower   = repmat(Problem.lower,subN,1);
    Upper   = repmat(Problem.upper,subN,1);
    PopDec  = [];
    
    %% Select direction solutions
    startP = nondec(randperm(N-domiN,1),:);
    if domiN < RefNo  
        %This part increases the convergence
        if N <= RefNo
            endP = [domidec;nondec(randperm(N-domiN,N-domiN),:)];
        else
            endP = [domidec;nondec(randperm(N-domiN,RefNo-domiN),:)];
        end
    else
        % This part increase the diversity
        endP = domidec(randperm(domiN,RefNo),:);
    end
    RefNo   = size(endP,1);
    vector  = (endP - repmat(startP,RefNo,1));  
    Direct  = vector./repmat(sum(vector.^2,2).^(1/2),1,D);
    for i = 1 : RefNo
        lambda  = (nondec - repmat(startP,N-domiN,1))*Direct(i,:)';
        sigma   = std(lambda)*(1+((domiN+1)/N)^1);
        OffDec  = repmat(normrnd(0,sigma,[subN,1]),1,D).*repmat(Direct(i,:),subN,1)+repmat(startP,subN,1);
        PopDec  = [PopDec;max(min(OffDec,Upper),Lower)];
    end

    %% Polynomial mutation
	OffDec = PopDec;
    N      = size(OffDec,1);
    Lower  = repmat(Problem.lower,N,1);
    Upper  = repmat(Problem.upper,N,1);
    disM   = 20;
    Site   = rand(N,D) < 1/Problem.D;
    mu     = rand(N,D);
    temp   = Site & mu<=0.5;
    OffDec = max(min(OffDec,Upper),Lower);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
	Offspring = Problem.Evaluation(OffDec);
end
```

### `PreSelection.m`
```matlab
function [Population, FrontNo] = PreSelection(Population,V,theta,RefNo)
% The preselection strategy in DGEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    %% Preseleting some well-converged solutions
    NV     = size(V,1);
    [Front,MaxFront] = NDSort(Population.objs,min([NV,length(Population)]));
    if  sum(Front==1) >= RefNo  
        Next0 = Front < MaxFront;
        Last = find(Front == MaxFront);
    else
        Next0 = Front == 1;
        Last = find(Front >1);
    end

    %% Selecting those diversity-related but well-converged solutions
    PopObj = Population(Last).objs;
    [N,M]  = size(PopObj);

    %% Translate the population
    PopObj = PopObj - repmat(min(Population.objs,[],1),N,1);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current = find(associate==i);
        % Calculate the APD value of each solution
        APD = (1+M*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
        % Select the one with the minimum APD value
        [~,best] = min(APD);
        Next(i) = current(best);
    end
    % Population for next generation
    Next0(Last(Next(Next~=0))) = true;
    Population = Population(Next0);
    FrontNo = Front(Next0);
end
```

### `subIBEA.m`
```matlab
function Population = subIBEA(Population,N,kappa)
% The environmental selection of IBEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    Next = 1 : length(Population);
    [Fitness,I,C] = CalFitness(Population.objs,kappa);
    while length(Next) > N
        [~,x]   = min(Fitness(Next));
        Fitness = Fitness + exp(-I(Next(x),:)/C(Next(x))/kappa);
        Next(x) = [];
    end
    Population = Population(Next);
end


function [Fitness,I,C] = CalFitness(PopObj,kappa)

    N = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end
```

### `subNSGAII.m`
```matlab
function Population = subNSGAII(Population,N)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    %% Non-dominated sorting
    CC = min([N,length(Population)]);
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,CC);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:CC-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
end
```

### `subNSGAIII.m`
```matlab
function Population = subNSGAIII(Population,N,Z,Zmin)
% The environmental selection of NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end
    CC = min([length(Population),N]);
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,CC);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,CC-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
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

### `subRVEA.m`
```matlab
function Population = subRVEA(Population,V,theta)
% The environmental selection of RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    NV     = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the degree of violation of each solution
    CV = sum(max(0,Population.cons),2);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current1 = find(associate==i & CV==0);
        current2 = find(associate==i & CV~=0);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD = (1+M*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
            % Select the one with the minimum APD value
            [~,best] = min(APD);
            Next(i)  = current1(best);
        elseif ~isempty(current2)
            % Select the one with the minimum CV value
            [~,best] = min(CV(current2));
            Next(i)  = current2(best);
        end
    end
    % Population for next generation
    Population = Population(Next(Next~=0));
end
```

### `subSPEA2.m`
```matlab
function Population = subSPEA2(Population,N)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    %% Calculate the fitness of each solution
    Fitness = CalFitness(Population.objs);
    CC = min([length(Population),N]);
    
    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < CC
        [~,Rank] = sort(Fitness);
        Next(Rank(1:CC)) = true;
    elseif sum(Next) > CC
        Del  = Truncation(Population(Next).objs,sum(Next)-CC);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
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

function Fitness = CalFitness(PopObj)
% Calculate the fitness of each solution
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
