# FDV

**Tags**: <2023> <multi/many> <real/integer> <large/none>

## Description
Fuzzy decision variable framework with various internal optimizers

## Reference
X. Yang, J. Zou, S. Yang, J. Zheng, and Y. Liu. A fuzzy decision variables framework for large-scale multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2023, 27(3): 445-459.

## Source Code

### `EnvironmentalSelection_CMOPSO.m`
```matlab
function Population = EnvironmentalSelection_CMOPSO(Population,N)
% The environmental selection of CMOPSO

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

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

### `EnvironmentalSelection_LMOCSO.m`
```matlab
function Population = EnvironmentalSelection_LMOCSO(Population,V,theta)
% The environmental selection of LMOCSO

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

    Population = Population(NDSort(Population.objs,1)==1);%ȡ֧һ
    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    NV     = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
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
    Population = Population(Next(Next~=0));
end
```

### `EnvironmentalSelection_NSGAII.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection_NSGAII(Population,N)
% The environmental selection of NSGA-II

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

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

### `EnvironmentalSelection_NSGAIII.m`
```matlab
function Population = EnvironmentalSelection_NSGAIII(Population,N,Z,Zmin)
% The environmental selection of NSGA-III

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,N-sum(Next),Z,Zmin);
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

### `FDV.m`
```matlab
classdef FDV < ALGORITHM
% <2023> <multi/many> <real/integer> <large/none> 
% Fuzzy decision variable framework with various internal optimizers
% Rate      --- 0.8 --- Fuzzy evolution rate. Default = 0.8
% Acc       --- 0.4 --- Step acceleration. Default = 0.4
% optimizer ---   5 --- Internal optimisation algorithm. 1 = NSGA-II, 2 = NSGA-III, 3 = MOEA/D, 4 = CMOPSO, 5 = LMOCSO
% type      ---   1 --- The type of aggregation function for MOEA/D

%------------------------------- Reference --------------------------------
% X. Yang, J. Zou, S. Yang, J. Zheng, and Y. Liu. A fuzzy decision
% variables framework for large-scale multiobjective optimization. IEEE
% Transactions on Evolutionary Computation, 2023, 27(3): 445-459.
%--------------------------------------------------------------------------

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

    methods
        function main(Algorithm,Problem)
            %% Set the default parameters
            [Rate,Acc,optimizer,type] = Algorithm.ParameterSet(0.8,0.4,5,1);

            %% NSGAII
            if optimizer==1
                % Generate random population
                Population = Problem.Initialization();
                [~,FrontNo,CrowdDis] = EnvironmentalSelection_NSGAII(Population,Problem.N);

                % Optimization
                while Algorithm.NotTerminated(Population)
                    MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                    OffDec     = OperatorGA(Problem,Population(MatingPool).decs);
                    %% FDV
                    if Problem.FE/Problem.maxFE <= Rate
                        Offspring = FDVOperator(Problem,Rate,Acc,OffDec);
                    else
                        Offspring = Problem.Evaluation(OffDec);
                    end
                    %% 
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection_NSGAII([Population,Offspring],Problem.N);
                end
            end

            %% NSGAIII
            if optimizer==2
                % Generate the reference points and random population
                [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
                Population    = Problem.Initialization();
                Zmin          = min(Population(all(Population.cons<=0,2)).objs,[],1);

                % Optimization
                while Algorithm.NotTerminated(Population)
                    MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));
                    OffDec     = OperatorGA(Problem,Population(MatingPool).decs);
                    %% FDV
                    if Problem.FE/Problem.maxFE <= Rate
                        Offspring = FDVOperator(Problem,Rate,Acc,OffDec);
                    else
                        Offspring = Problem.Evaluation(OffDec);
                    end
                    Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                    Population = EnvironmentalSelection_NSGAIII([Population,Offspring],Problem.N,Z,Zmin);
                end
            end

            %% MOEA/D
            if optimizer==3
                % Generate the weight vectors
                [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
                T = ceil(Problem.N/10);

                % Detect the neighbours of each solution
                B = pdist2(W,W);
                [~,B] = sort(B,2);
                B = B(:,1:T);

                % Generate random population
                Population = Problem.Initialization();
                Z = min(Population.objs,[],1);

                % Optimization
                while Algorithm.NotTerminated(Population)
                    % For each solution
                    for i = 1 : Problem.N      
                        % Choose the parents
                        P = B(i,randperm(size(B,2)));

                        % Generate an offspring
                        OffDec = OperatorGAhalf(Problem,Population(P(1:2)).decs);
                        %% FDV
                        if Problem.FE/Problem.maxFE <= Rate
                            Offspring = FDVOperator(Problem,Rate,Acc,OffDec);
                        else
                            Offspring = Problem.Evaluation(OffDec);
                        end

                        % Update the ideal point
                        Z = min(Z,Offspring.obj);

                        % Update the neighbours
                        switch type
                            case 1
                                % PBI approach
                                normW   = sqrt(sum(W(P,:).^2,2));
                                normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                                normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                                CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                                CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                                g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                                g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                            case 2
                                % Tchebycheff approach
                                g_old = max(abs(Population(P).objs-repmat(Z,T,1)).*W(P,:),[],2);
                                g_new = max(repmat(abs(Offspring.obj-Z),T,1).*W(P,:),[],2);
                            case 3
                                % Tchebycheff approach with normalization
                                Zmax  = max(Population.objs,[],1);
                                g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                                g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                            case 4
                                % Modified Tchebycheff approach
                                g_old = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2);
                                g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2);
                        end
                        Population(P(g_old>=g_new)) = Offspring;
                    end
                end
            end

            %% CMOPSO
            if optimizer == 4
                % Generate random population
                Population = Problem.Initialization();

                % Optimization
                while Algorithm.NotTerminated(Population)
                    [OffDec,OffVel] = Operator_CMOPSO(Problem,Population);
                    %% FDV
                    if Problem.FE/Problem.maxFE <= Rate
                        Offspring = FDVOperator(Rate,Acc,OffDec,OffVel);
                    else
                        Offspring = Problem.Evaluation(OffDec,OffVel);
                    end
                    Population = EnvironmentalSelection_CMOPSO([Population,Offspring],Problem.N);
                end
            end

            %% LMOCSO
            if optimizer == 5
                 % Generate random population
                [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
                Population    = Problem.Initialization();
                Population    = EnvironmentalSelection_LMOCSO(Population,V,(Problem.FE/Problem.maxFE)^2);

                % Optimization
                while Algorithm.NotTerminated(Population)
                    % Calculate the fitness by shift-based density   SDE (the shift-based density estimation strategy)
                    PopObj = Population.objs;
                    N      = size(PopObj,1);
                    fmax   = max(PopObj,[],1);
                    fmin   = min(PopObj,[],1);
                    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
                    Dis    = inf(N);
                    for i = 1 : N
                        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
                        for j = [1:i-1,i+1:N]
                            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:)); 
                        end
                    end
                    Fitness = min(Dis,[],2); 

                    if length(Population) >= 2
                        Rank = randperm(length(Population),floor(length(Population)/2)*2);
                    else
                        Rank = [1,1];
                    end
                    Loser  = Rank(1:end/2);
                    Winner = Rank(end/2+1:end);
                    Change = Fitness(Loser) >= Fitness(Winner);
                    Temp   = Winner(Change);
                    Winner(Change) = Loser(Change);
                    Loser(Change)  = Temp;

                    [OffDec,OffVel] = Operator_LMOCSO(Problem,Population(Loser),Population(Winner),Rate);
                    %% FDV
                    iter = Problem.FE/Problem.maxFE;
                    if iter <= Rate
                        Offspring = FDVOperator(Problem,Rate,Acc,OffDec,OffVel);
                    else
                        Offspring = Problem.Evaluation(OffDec,OffVel);
                    end
                    Population = EnvironmentalSelection_LMOCSO([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2);
                end
            end
        end
    end
end
```

### `FDVOperator.m`
```matlab
function Offspring = FDVOperator(Problem,Rate,Acc,OffDec,OffVel)

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>
    
    %% Fuzzy Evolution Sub-stages Division
    Total = 1;
    S = floor(sqrt(2*Rate*Total/Acc));
    Step = zeros(1,S+2);  % Step(1)=0Step(S+2) is the compensation step
    for i = 1 : S
        Step(i+1) = (S*i-i*i/2)*Acc;
    end
    Step(S+2) = Rate*Total;  % compensation step

    %% Fuzzy Operation
    R    = Problem.upper-Problem.lower;
    iter = Problem.FE/Problem.maxFE;  % step=[0,0.6,0.8,0.8]
    for i = 1 : S+1
        if iter>Step(i) && iter<=Step(i+1)
            gamma_a = R*10^-i.*floor(10^i*R.^-1.*(OffDec-Problem.lower)) + Problem.lower;
            gamma_b = R*10^-i.*ceil(10^i*R.^-1.*(OffDec-Problem.lower)) + Problem.lower;
            miu1    = 1./(OffDec-gamma_a);
            miu2    = 1./(gamma_b-OffDec);
            logical = miu1-miu2>0;
            OffDec  = gamma_b;
            OffDec(find(logical)) = gamma_a(find(logical));
        end
    end
    if nargin > 4
        Offspring = Problem.Evaluation(OffDec,OffVel);
    else
        Offspring = Problem.Evaluation(OffDec);
    end
end
```

### `Operator_CMOPSO.m`
```matlab
function [Off_P,Off_V] = Operator_CMOPSO(Problem,Population,Parameter)
% The particle swarm optimization in CMOPSO

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

    %% Parameter setting
    if nargin > 2
        [proM,disM] = deal(Parameter{:});
    else
        [proM,disM] = deal(1,20);
    end
    P_Dec  = Population.decs;     
    [N,D]  = size(P_Dec); 
    P_Obj  = Population.objs;
    V      = Population.adds(zeros(N,D));
    Off_P  = zeros(N,D);
    Off_V  = zeros(N,D);
    
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
end
```

### `Operator_LMOCSO.m`
```matlab
function [OffDec,OffVel] = Operator_LMOCSO(Problem,Loser,Winner,Rate)
% The competitive swarm optimizer of LMOCSO

%  Copyright (C) 2021 Xu Yang
%  Xu Yang <xuyang.busyxu@qq.com> or <xuyang369369@gmail.com>

    %% Parameter setting
    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(LoserDec);
	LoserVel  = Loser.adds(zeros(N,D));
    WinnerVel = Winner.adds(zeros(N,D));
    
    %% Competitive swarm optimizer
    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    
    OffVel = r1.*LoserVel + r2.*(WinnerDec-LoserDec);
    OffDec = LoserDec + OffVel;
    
    if Problem.FE/Problem.maxFE < Rate
        LoserVel1 = rand(N,D);
        OffVel1 = r1.*LoserVel1 + r2.*(WinnerDec-LoserDec);
        OffDec1 = LoserDec + OffVel1 + r1.*(OffVel1-LoserVel1);
        
        OffDec = [OffDec;OffDec1];
        OffVel = [OffVel;OffVel1];
    end
    
    %% Add the winners
    OffDec = [OffDec;WinnerDec];
    OffVel = [OffVel;WinnerVel];
 
    %% Polynomial mutation
    [N,D] = size(OffDec);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    disM  = 20;
    Site  = rand(N,D) < 1/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    OffDec       = max(min(OffDec,Upper),Lower);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));          
end
```
