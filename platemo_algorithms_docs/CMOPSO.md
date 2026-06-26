# CMOPSO

**Tags**: <2018> <multi> <real/integer>

## Description
Competitive mechanism based multi-objective particle swarm optimizer

## Reference
X. Zhang, X. Zheng, R. Cheng, J. Qiu, and Y. Jin. A competitive mechanism based multi-objective particle swarm optimizer with fast convergence. Information Sciences, 2018, 427: 63-76.

## Source Code

### `CMOPSO.m`
```matlab
classdef CMOPSO < ALGORITHM
% <2018> <multi> <real/integer>
% Competitive mechanism based multi-objective particle swarm optimizer

%------------------------------- Reference --------------------------------
% X. Zhang, X. Zheng, R. Cheng, J. Qiu, and Y. Jin. A competitive mechanism
% based multi-objective particle swarm optimizer with fast convergence.
% Information Sciences, 2018, 427: 63-76.
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
            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Offspring  = Operator(Problem,Population);
                Population = EnvironmentalSelection([Population,Offspring],Problem.N);
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N)
% The environmental selection of CMOPSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    PopObj = Population.objs;
    fmax   = max(PopObj(FrontNo==1,:),[],1);
    fmin   = min(PopObj(FrontNo==1,:),[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);

    %% Select the solutions in the last front
    Last = find(FrontNo==MaxFNo);
    del  = Truncation(PopObj(Last,:),length(Last)-N+sum(Next));
    Next(Last(~del)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    N = size(PopObj,1);
    
    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,N);
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Population,Parameter)
% The particle swarm optimization in CMOPSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    if nargin > 2
        [proM,disM] = deal(Parameter{:});
    else
        [proM,disM] = deal(1,20);
    end
    P_Dec = Population.decs;     
    [N,D] = size(P_Dec); 
    P_Obj = Population.objs;
    V     = Population.adds(zeros(N,D));
    Off_P = zeros(N,D);
    Off_V = zeros(N,D);  
    
    %% Get leaders  
    Front     = NDSort(P_Obj,inf);    
    [~,rank]  = sortrows([Front',-CrowdingDistance(P_Obj,Front)']);
    LeaderSet = rank(1:10);
    
    %% Learning
    for i = 1 : N
        % Competition according to the angle 
        winner = LeaderSet(randperm(length(LeaderSet),2));
        c1     = dot(P_Obj(i,:),P_Obj(winner(1),:))/(norm(P_Obj(i,:))*norm(P_Obj(winner(1),:)));
        angle1 = rad2deg(acos(c1));
        c2     = dot(P_Obj(i,:),P_Obj(winner(2),:))/(norm(P_Obj(i,:))*norm(P_Obj(winner(2),:)));
        angle2 = rad2deg(acos(c2));
        mask   = (angle1 > angle2);
        winner = ~mask.*winner(1) + mask.*winner(2);
        % Learning
        r1 = rand(1,D);
        r2 = rand(1,D);
        Off_V(i,:) = r1.*V(i,:) + r2.*(P_Dec(winner,:)-P_Dec(i,:));
        Off_P(i,:) = P_Dec(i,:) + Off_V(i,:);
    end

    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Off_P = min(max(Off_P,Lower),Upper);
    Off_P(temp) = Off_P(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                  (1-(Off_P(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Off_P(temp) = Off_P(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                  (1-(Upper(temp)-Off_P(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
	Offspring = Problem.Evaluation(Off_P,Off_V);
end
```
