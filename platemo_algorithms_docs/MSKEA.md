# MSKEA

**Tags**: <2022> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>

## Description
Multi-stage knowledge-guided evolutionary algorithm

## Reference
Z. Ding, L. Chen, D. Sun, and X. Zhang. A multi-stage knowledge-guided evolutionary algorithm for sparse multi-objective optimization problems. Swarm and Evolutionary Computation, 2022, 73: 101119.

## Source Code

### `MSKEA.m`
```matlab
classdef MSKEA < ALGORITHM
% <2022> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
% Multi-stage knowledge-guided evolutionary algorithm
 
%------------------------------- Reference --------------------------------
% Z. Ding, L. Chen, D. Sun, and X. Zhang. A multi-stage knowledge-guided
% evolutionary algorithm for sparse multi-objective optimization problems.
% Swarm and Evolutionary Computation, 2022, 73: 101119.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lei Chen

    methods
         function main(Algorithm,Problem)
            %% pv and population initialization  
            TDec    = [];
            TMask   = [];
            TempPop = [];
            pv      = zeros(1,Problem.D);
            for i = 1 : 1+4*any(Problem.encoding~=4)
                Dec = unifrnd(repmat(Problem.lower,Problem.D,1),repmat(Problem.upper,Problem.D,1));
                Dec(:,Problem.encoding==4) = 1;
                Mask       = eye(Problem.D);
                Population = Problem.Evaluation(Dec.*Mask);
                TDec       = [TDec;Dec];
                TMask      = [TMask;Mask];
                TempPop    = [TempPop,Population];
                pv         = pv + NDSort([Population.objs,Population.cons],inf);
            end
            Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            Dec(:,Problem.encoding==4) = 1;
            Mask = false(Problem.N,Problem.D);
            
            %% Mask initialization by pv
            for i = 1 : Problem.N
                Mask(i,TournamentSelection(2,ceil(rand*Problem.D),pv)) = 1;
            end
            Population    = Problem.Evaluation(Dec.*Mask);
            [Population,Dec,Mask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection([Population,TempPop],[Dec;TDec],[Mask;TMask],Problem.N);
            sv            =zeros(1,Problem.D);
            Last_temp_num = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,2*Problem.N,FrontNo,-CrowdDis);
                %-------------update fv-----------%
                delta = Problem.FE/Problem.maxFE;
                if delta < 0.618
                    fv = std(Population(FrontNo==1).decs,0,1);
                    fv(:,Problem.encoding==4) = sum(Mask(FrontNo==1,Problem.encoding==4),1);
                end
                %-------------update sv-----------%
                First_Mask    = Mask(FrontNo==1,:);
                [temp_num,~]  = size(First_Mask);
                temp_vote     = sum(First_Mask,1);
                sv(1,:)       = (Last_temp_num/(Last_temp_num+temp_num))*sv(1,:)+(temp_num/(Last_temp_num+temp_num))*(temp_vote/temp_num);
                Last_temp_num = temp_num;
                %-------------update pv by sv-----------%
                if delta < 0.618
                    pv = pv.*(1-sv)*sqrt(delta)+pv;
                end
                %--------------------------------------%
                if (delta/0.618) < 0.618
                    [OffDec,OffMask] = Operator_pvfv(Problem,Dec(MatingPool,:),Mask(MatingPool,:),pv,fv,delta);
                elseif (delta/0.618)>=0.618 && delta< 0.618
                    if rand < 0.5
                        [OffDec,OffMask] = Operator_sv(Problem,Dec(MatingPool,:),Mask(MatingPool,:),sv);
                    else
                        [OffDec,OffMask] = Operator_pvfv(Problem,Dec(MatingPool,:),Mask(MatingPool,:),pv,fv,delta);
                    end
                else
                    [OffDec,OffMask] = Operator_sv(Problem,Dec(MatingPool,:),Mask(MatingPool,:),sv);
                end
                Offspring = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
            end
        end
    end
end
```

### `Operator_pvfv.m`
```matlab
function [OffDec,OffMask] = Operator_pvfv(Problem,ParentDec,ParentMask,pv,fv,delta)
% The operator of MSKEA guided by pv and sv

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lei Chen

    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        if rand < 0.5
            index = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
            index = index(TS(-pv(index)));
            OffMask(i,index) = 0;
        else
            index = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
            index = index(TS(pv(index)));
            OffMask(i,index) = Parent2Mask(i,index);
        end
    end
    
    %% Mutation for mask
    if rand < (1-delta)
        f_vector       = fv;
        f_vector(fv>0) = 1;
        for i = 1 : N/2
            index = find(OffMask(i,:)~=f_vector);
            if rand < 0.5
                index = index(TS(-fv(index)));
                OffMask(i,index) = 1;
            else
                index = index(TS(fv(index)));
                OffMask(i,index) = 0;
            end
        end
    else
        for i = 1 : N/2
            if rand < 0.5
                index = find(OffMask(i,:));
                index = index(TS(-pv(index)));
                OffMask(i,index) = 0;
            else
                index = find(~OffMask(i,:));
                index = index(TS(pv(index)));
                OffMask(i,index) = 1;
            end
        end
    end
    
    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

function index = TS(pv)
% Binary tournament selection

    if isempty(pv)
        index = [];
    else
        index = TournamentSelection(2,1,pv);
    end
end
```

### `Operator_sv.m`
```matlab
function [OffDec,OffMask] = Operator_sv(Problem,ParentDec,ParentMask,sv)
% The operator of MSKEA guided by pv and sv

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lei Chen
   
    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    rate0   = sv;           % The probability that  0 inverts to  1
    rate1   = 1-rate0;      % The probability that  1 inverts to  0

    for i = 1 : N/2
        diff       = find(Parent1Mask(i,:)~=Parent2Mask(i,:));
        temp_rate1 = rate1(diff);
        temp_rate0 = rate0(diff);
        rate = zeros(1,length(diff));
        rate(logical(OffMask(i,diff)))  = temp_rate1(logical(OffMask(i,diff)));
        rate(logical(~OffMask(i,diff))) = temp_rate0(logical(~OffMask(i,diff)));
        exchange = rand(1,length(diff)) < rate;
        OffMask(i,diff(exchange))=~OffMask(i,diff(exchange)); 
    end

    %% Mutation for mask
    Mutation_p  = 1/D;                      % Probability of mutation
    Mu_exchange = rand(N/2,D)<Mutation_p;   % The decision variables less than 1/D will be mutated
    for i = 1 : N/2
        if sum(Mu_exchange(i,:))
            subscript = find(Mu_exchange(i,:)==1);
            rate      = zeros(1,size(subscript,2));
            rate(logical(OffMask(i,subscript)))  = rate1(subscript(logical(OffMask(i,subscript))));
            rate(logical(~OffMask(i,subscript))) = rate0(subscript(logical(~OffMask(i,subscript))));
            exchange  = rand(1,size(subscript,2)) < rate;
            OffMask(i,subscript(exchange)) = ~OffMask(i,subscript(exchange));
        end
    end
   
    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end
```

### `SPEA2_EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = SPEA2_EnvironmentalSelection(Population,Dec,Mask,N)
% The environmental selection of MSKEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lei Chen

    %% Delete duplicated solutions
    [~,uni] = unique(Population.objs,'rows');
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
    CrowdDis   = CrowdingDistance(Population.objs,FrontNo);
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
