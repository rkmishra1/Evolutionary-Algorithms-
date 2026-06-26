# POCEA

**Tags**: <2021> <multi> <real/integer> <large/none> <constrained>

## Description
Paired offspring generation based constrained evolutionary algorithm

## Reference
C. He, R. Cheng, Y. Tian, X. Zhang, K. C. Tan, and Y. Jin. Paired offspring generation for constrained large-scale multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(3): 448-462.

## Source Code

### `CHP.m`
```matlab
function [winner,loser] = CHP(P1,P2,epsilon) 
% Paired competition

% Copyright (c) 2020-2021 Cheng He

    winner = P1;
    loser  = P2;
    N      = length(P1);
    CV1    = sum(max(P1.cons,0),2);
    CV2    = sum(max(P2.cons,0),2);
    flag   = zeros(N,1);
    for i = 1 : N
        if max(CV1(i),CV2(i)) <= epsilon
            flag(i) = 2 - (norm(P1(i).objs)<norm(P2(i).objs));
        else
            flag(i) = 2 - (CV1(i)<CV2(i));
        end
        if flag(i) == 1
			[winner(i),loser(i)] = deal(P1(i),P2(i));
        else
            [winner(i),loser(i)] = deal(P2(i),P1(i));
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Loser,Winner)
% The competitive swarm optimizer

% Copyright (c) 2020-2021 Cheng He

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
    OffDec = LoserDec + OffVel + r1.*(OffVel-LoserVel);
    
	%% Add the winners
	OffDec = [OffDec;WinnerDec];
	OffVel = [OffVel;WinnerVel];
	N = size(OffDec,1);
    
    %% Polynomial mutation
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
	Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `POCEA.m`
```matlab
classdef POCEA < ALGORITHM
% <2021> <multi> <real/integer> <large/none> <constrained>
% Paired offspring generation based constrained evolutionary algorithm

%------------------------------- Reference --------------------------------
% C. He, R. Cheng, Y. Tian, X. Zhang, K. C. Tan, and Y. Jin. Paired
% offspring generation for constrained large-scale multiobjective
% optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(3):
% 448-462.
%--------------------------------------------------------------------------

% Copyright (c) 2020-2021 Cheng He

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            k = Algorithm.ParameterSet(5);
            Population     = Problem.Initialization();
            [V0,Problem.N] = UniformPoint(Problem.N,Problem.M);
            [Vs0,L]        = UniformPoint(floor(Problem.N/k),Problem.M);
            [V,Vs]         = deal(V0,Vs0);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                [INDEX,THETA,DIS] = Association(Population,Vs,k);
                CV = sum(max(Population.cons,0),2);
                rf = sum(CV<1e-6)/length(Population);
                Offspring = [];

                %% Paired offspring generation
                for i = 1 : L
                    % Subpopulation construction
                    SubPop = Population(INDEX(:,i));
                    theta  = THETA(INDEX(:,i));
                    if mean(theta) >= pi/L/2
                        [~,index] = sort(DIS);
                        selected  = index(1:min(k,length(index)));
                        SubPop    = [SubPop,Population(selected)];
                        epsilon   = max(CV([INDEX(:,i);selected]));
                    else
                        epsilon   = min(CV(INDEX(:,i)))*(1-rf)+mean(CV(INDEX(:,i)))*rf;
                    end
                    % Offspring generation
                    if length(SubPop) < 2
                        rank = [1, 1];
                    else
                        [~,rank]= sort(rand(k,length(SubPop)),2);
                    end
                    [winner,loser] = CHP(SubPop(rank(1)),SubPop(rank(2)),epsilon);
                    Offspring      = [Offspring,Operator(Problem,loser,winner)];
                end 
                Population = RVEASelection([Population,Offspring],V,Problem.N,(Problem.FE/Problem.maxFE)^2);
                if ~mod(ceil(Problem.FE/Problem.N),ceil(0.1*Problem.maxFE/Problem.N))
                    [V,Vs] = ReferenceVectorAdaptation(Population.objs,V0,Vs0);
                end
            end
        end
    end
end

function [INDEX,THETA,DIS] = Association(Population,V,k)
% Associate k candidate solutions to each reference vector

    PopObj    = Population.objs - repmat(min(Population.objs,[],1),length(Population),1);
	DIS	      = sum(PopObj.^2,2);
    THETA     = acos(1 - pdist2(PopObj,V,'cosine'));
    [~,index] = sort(THETA,1);
    INDEX     = index(1:min(k,end),:);
end
```

### `RVEASelection.m`
```matlab
function Population = RVEASelection(Population,V,popsize,theta)
% Uniformity optimization by RVEA

% Copyright (c) 2020-2021 Cheng He

    PopObj = Population.objs;
    [N,~]  = size(PopObj);
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
    pf   = sum(CV(randi(N,[N,1]))<1e-6)/N;
    for i = unique(associate)'
        cv   = CV(associate==i);
        Ns   = sum(associate==i);
        subN = ceil(popsize/numel(unique(associate)'));
		% Ensure the subpopulation size
        if Ns < subN
            epsilon = max(cv);
        else
            epsilon = min(cv)*(1-pf)+mean(cv)*pf;
        end
        current1 = find(associate==i & CV<=epsilon);
        current2 = find(associate==i & CV>epsilon);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD      = (1+theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
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

### `ReferenceVectorAdaptation.m`
```matlab
function [V1,V2] = ReferenceVectorAdaptation(PopObj,V1,V2)
% Reference vector adaption strategy of two reference vector sets

% Copyright (c) 2020-2021 Cheng He

    PopObj = max(PopObj,[],1)-min(PopObj,[],1);
    V1 	   = V1.*repmat(PopObj,size(V1,1),1); 
    V2     = V2.*repmat(PopObj,size(V2,1),1); 
end
```
