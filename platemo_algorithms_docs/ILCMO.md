# ILCMO

**Tags**: <2026> <multi/many> <real/integer> <large/none> <constrained/none>

## Description
Indicator-based evolutionary algorithm for large-scale constrained multi-objective optimization

## Reference
X. Zhong, X. Yao, K. Qiao, D. Gong, and Y. Jin. An indicator-based evolutionary algorithm for large-scale constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2026, 30(1): 271-285.

## Source Code

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon)
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
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
    end

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
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

### `CreateGroups.m`
```matlab
function [outIndexArray,numberOfGroupsArray] = CreateGroups(numberOfGroups, xPrime,numberOfVariables, method)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    outIndexArray       = [];
    numberOfGroupsArray = [];
    [noOfSolutions,D]   = size(xPrime);
    for sol = 1 : noOfSolutions
        switch method
            case 1 % linear grouping
                varsPerGroup = floor(numberOfVariables/numberOfGroups);
                outIndexList = [];
                for i = 1 : numberOfGroups-1
                   outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
                end
                outIndexList = [outIndexList, ones(1,numberOfVariables-size(outIndexList,2)).*numberOfGroups];
            case 2 % orderByValueGrouping
                varsPerGroup = floor(numberOfVariables/numberOfGroups);
                vars         = xPrime(sol,:);
                [~,I]        = sort(vars);
                outIndexList = ones(1,numberOfVariables);
                for i = 1 : numberOfGroups-1
                   outIndexList(I(((i-1)*varsPerGroup)+1:i*varsPerGroup)) = i;
                end
                outIndexList(I(((numberOfGroups-1)*varsPerGroup)+1:end)) = numberOfGroups;
            case 3 % random Grouping
                varsPerGroup = floor(numberOfVariables/numberOfGroups);
                outIndexList = [];
                for i = 1 : numberOfGroups-1
                   outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
                end
                outIndexList = [outIndexList, ones(1,numberOfVariables-size(outIndexList,2)).*numberOfGroups];
                outIndexList = outIndexList(randperm(length(outIndexList)));
        end
        outIndexArray       = [outIndexArray;outIndexList];
        numberOfGroupsArray = [numberOfGroupsArray;numberOfGroups];
    end
end
```

### `DI_Update.m`
```matlab
function [return_pop,return_Fitness,SelInd] = DI_Update(MaxPop,N,VAR1,VAR2,W)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj  = MaxPop.objs;
    Zmin    = min(PopObj,[],1);
    [Num,~] = size(PopObj); 
    Cons    = max(0,MaxPop.cons);
    CV      = sum(Cons,2);
    Gmin    = min(CV);
    Gmax    = max(CV);
    
    %% shift the objective space to R+
    PopObj = PopObj - repmat(Zmin,Num,1) + 1e-6;
    
    findex       = find(CV<=VAR1);
    ifindex      = find(CV>VAR1);
    fPopulation  = MaxPop(findex);
    ifPopulation = MaxPop(ifindex);
    Population   = [fPopulation,ifPopulation];  
    fnum         = length(fPopulation);
    inum         = length(ifPopulation);
    fPopObj      = PopObj(findex,:);
    ifPopObj     = PopObj(ifindex,:);
    PopObj       = [fPopObj;ifPopObj]; 
    fCV          = CV(findex);
    iCV          = CV(ifindex);
    CV           = [fCV;iCV];
    
    %% calculate the indicator matrix
    IMatrix = ones(Num,Num);
    for i = 1 : Num
        Ci = CV(i);
        if Ci <= VAR2
            Fi           = PopObj(i,:);
            Ir           = log(repmat(Fi,fnum,1)./fPopObj);
            MaxIr        = max(Ir,[],2);
            MinIr        = min(Ir,[],2);
            CVA          = MaxIr;
            DomInds      = find(MaxIr<=0);
            CVA(DomInds) = MinIr(DomInds);
            IndicatorV   = CVA;
        else
            IC         = repmat(Ci+1e-6,fnum,1)./(fCV+1e-6); 
            Fi         = PopObj(i,:);
            MaxF       = max(repmat(Fi,fnum,1),fPopObj);
            MinF       = min(repmat(Fi,fnum,1),fPopObj);
            CVF        = max(MaxF./MinF,[],2);
            IndicatorV = log(max([CVF,IC],[],2));
        end
        IMatrix(1:fnum,i) = IndicatorV;
        
        
        IC                    = repmat(Ci+1e-6,inum,1)./(iCV+1e-6); 
        IndicatorV            = log(IC)+repmat(log((Gmin+1e-6)/(Gmax+1e-6))+log(min(min(PopObj,[],1)./max(PopObj,[],1))),inum,1)-1e-6;
        IMatrix(1+fnum:Num,i) = IndicatorV;
    end
    
    IMatrix(logical(eye(Num))) = Inf;
    Fitness                    = min(IMatrix,[],2);

    % using indicator-based CHT to update the population
    SelInd         = Indicator_based_CHT(PopObj,IMatrix,W,N);
    return_pop     = Population(SelInd);
    return_Fitness = Fitness(SelInd);
end
```

### `ILCMO.m`
```matlab
classdef ILCMO < ALGORITHM
% <2026> <multi/many> <real/integer> <large/none> <constrained/none>
% Indicator-based evolutionary algorithm for large-scale constrained multi-objective optimization

%------------------------------- Reference --------------------------------
% X. Zhong, X. Yao, K. Qiao, D. Gong, and Y. Jin. An indicator-based
% evolutionary algorithm for large-scale constrained multiobjective
% optimization. IEEE Transactions on Evolutionary Computation, 2026, 30(1):
% 271-285.
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
            %% Parameter setting
            [typeOfGroups,numberOfGroups] = Algorithm.ParameterSet(2,4);
    
            %% Generate the reference points and random population
            W             = UniformPoint(Problem.N,Problem.M);
            Population{1} = Problem.Initialization();
            Population{2} = Problem.Initialization();
            Fmin          = min(Population{1}.objs,[],1);
            Fitness{1}    = CalFitness(Population{1}.objs,Population{1}.cons);
            Fitness{2}    = CalFitness(Population{2}.objs);
    
            %% Calculate the initial dynamic constraint boundary
            CPopulation = [Population{1},Population{2}];
            VAR0        = max(sum(max(CPopulation.cons,0),2));
            if VAR0 == 0
                VAR0 = 1;
            end
            X = 0;
    
            %% Optimization
            while Algorithm.NotTerminated([Population{1}])
                % Udate the epsilon value
                cp   = (-log(VAR0)-6)/log(1-0.5);
                VAR1 = VAR0*(1-X)^cp;
                rf   = sum(sum(max(Population{1}.cons,0),2)<1e-6)/length(Population{1});
                VAR2 = VAR1*(1-rf);
                X    = X+1/(Problem.maxFE/(2*Problem.N));
                % Offspring generation
                P = 1-(2/(1+exp(-5*Problem.FE/Problem.maxFE))-1);
                for i = 1 : 2
                    valOffspring{i} = VGDE_main(Problem, Population{i}, Fitness{i},numberOfGroups, typeOfGroups,P);
                end
                % Update ideal points
                Offspring = [valOffspring{1},valOffspring{2}];
                Fmin      = min([Fmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                % Environmental selection
                [Population{1},Fitness{1},~] = Main_task_Update([Offspring,Population{1}],Problem.N,Fmin);
                [Population{2},Fitness{2},~] = DI_Update([Offspring,Population{1},Population{2}],Problem.N,VAR1,VAR2,W);
            end
        end
    end
end
```

### `Indicator_based_CHT.m`
```matlab
function SelInd = Indicator_based_CHT(PopObj,IMatrix,W,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % calculate the size of individuals in the promising areas
    IrFitness   = min(IMatrix,[],2);
    Level1Index = find(IrFitness>=0);
    Len_Level1  = length(Level1Index);

    if Len_Level1 <= N
        [~,SortIndex] = sort(-IrFitness);
        SelInd        = SortIndex(1:N);
    else
        % only focus on the solutions in the promising areas
        SelInd        = Level1Index;
        PopObj        = PopObj(Level1Index,:);
        IMatrix       = IMatrix(Level1Index,Level1Index);
        [Num,M]       = size(PopObj);
        NormW         = W./repmat(sqrt(sum(W.^2,2)),1,M);
        NormPopObj    = PopObj./repmat(sqrt(sum(PopObj.^2,2)),1,M);
        [~,ZoneIndex] = max(NormPopObj * NormW',[],2);
        Num_W         = size(W,1);
        ZoneDensity   = zeros(1,Num_W);
        zone.index    = [];
        Zone          = repmat(zone,1,Num_W);
        for j = 1 : Num
            Zj              = ZoneIndex(j);
            Zone(Zj).index  = [Zone(Zj).index,j];
            ZoneDensity(Zj) = ZoneDensity(Zj) + 1;
        end

        [NDensity,SortIndex] = sort(-ZoneDensity);
        Density              = abs(NDensity);
        [Values,Neightboor]  = min(IMatrix,[],2);

        DelNum      = Num - N;
        Have_Delect = zeros(1,DelNum);

        for i = 1 : DelNum
            [MDen,MDInd]   = max(Density);
            CandidateIndex = Zone(SortIndex(MDInd)).index;

            [~,NowDel_Ind] = min(Values(CandidateIndex));
            Del_Ind        = CandidateIndex(NowDel_Ind);
            CandidateIndex(NowDel_Ind) = [];
            Have_Delect(i)     = Del_Ind;
            IMatrix(Del_Ind,:) = Inf;
            IMatrix(:,Del_Ind) = Inf;
            Need_Updata        = find(Neightboor==Del_Ind);
            L_Need             = length(Need_Updata);
            if L_Need > 0
                [Values(Need_Updata),Neightboor(Need_Updata)] = min(IMatrix(Need_Updata,:),[],2);
            end
            Values(Del_Ind)              = Inf;
            Zone(SortIndex(MDInd)).index = CandidateIndex;
            Density(MDInd)               = MDen - 1;
        end
        SelInd(Have_Delect) = [];
    end
end
```

### `Main_task_Update.m`
```matlab
function [return_pop,return_Fitness,SelInd] = Main_task_Update(MaxPop,N,Fmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj  = MaxPop.objs;
    Zmin    = min(PopObj,[],1);
    [Num,~] = size(PopObj); 
    Cons    = max(0,MaxPop.cons);
    CV      = sum(Cons,2);
    Gmin    = min(CV);
    Gmax    = max(CV);
    % shift the objective space to R+
    PopObj = PopObj - repmat(Zmin,Num,1) + 1e-6;
    
    findex       = find(CV<=0);
    ifindex      = find(CV>0);
    fPopulation  = MaxPop(findex);
    ifPopulation = MaxPop(ifindex);
    Population   = [fPopulation,ifPopulation];  
    fnum         = length(fPopulation);
    inum         = length(ifPopulation);
    fPopObj      = PopObj(findex,:);
    ifPopObj     = PopObj(ifindex,:);
    PopObj       = [fPopObj;ifPopObj]; 
    fCV          = CV(findex);
    iCV          = CV(ifindex);
    CV           = [fCV;iCV];

    %% calculate the indicator matrix
    IMatrix = ones(Num,Num);
    for i = 1 : Num
        Ci = CV(i);
        if Ci <= 0 %%%%% Xi is feasible
            Fi           = PopObj(i,:);
            Ir           = log(repmat(Fi,fnum,1)./fPopObj);
            MaxIr        = max(Ir,[],2);
            MinIr        = min(Ir,[],2);
            CVA          = MaxIr;
            DomInds      = find(MaxIr<=0);
            CVA(DomInds) = MinIr(DomInds);
            IndicatorV   = CVA;
        else  %%%%% Xi is an infeasible solution
            IC         = Inf(fnum,1); 
            IndicatorV = IC;
        end
        IMatrix(1:fnum,i)     = IndicatorV;
        IC                    = repmat(Ci+1e-6,inum,1)./(iCV+1e-6); 
        IndicatorV            = log(IC)+repmat(log((Gmin+1e-6)/(Gmax+1e-6))+log(min(min(PopObj,[],1)./max(PopObj,[],1))),inum,1)-1e-6;
        IMatrix(1+fnum:Num,i) = IndicatorV;
    end
    
    IMatrix(logical(eye(Num))) = Inf;
    IrFitness                  = min(IMatrix,[],2);
    Level1Index                = find(IrFitness>=0);
    Len_Level1                 = length(Level1Index);

    %% update the Archive
    if Len_Level1 <= N
        [~,SortInd]    = sort(-IrFitness);
        SelInd         = SortInd(1:N);
        return_pop     = Population(SelInd);
        return_Fitness = IrFitness(SelInd);
    else
        PopObj         = PopObj(Level1Index,:) + repmat(Zmin,Len_Level1,1) - repmat(Fmin,Len_Level1,1);
        SelInd         = Selection_Operator_of_PREA(PopObj,IMatrix(Level1Index,Level1Index),N);
        return_pop     = Population(Level1Index(SelInd));
        return_Fitness = IrFitness(Level1Index(SelInd));
    end
end
```

### `VGDE_main.m`
```matlab
function Offspring = VGDE_main(Problem, Population, Fitness,numberOfGroups, typeOfGroups,P)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N         = length(Population);
    Offspring = [];
    
    %% Mating selection
    MatingPool = TournamentSelection(2,N,Fitness);
    Population = Population(MatingPool);
    Fitness    = Fitness(MatingPool);
    
    %% Clustering
    [~, indBest] = sort(Fitness, 'descend');
    S1           = Population(indBest(1:N/2));%Winner
    location1    = rand(1,N/2)< P;
    S2           = Population(indBest(1+N/2:end));%Loser
    location2    = rand(1,N/2)<= P;
    
    %% Diversity evolution
    r0         = randperm(N/2);
    [r1,r2,r3] = gnR1R2R3(N/2, r0);
    Parent11   = S1(r0);
    Parent12   = S1(r1);
    Parent13   = S1(r2);
    
    if ~isempty(find(location1==1, 1))
        Off       = Group_DE_rand_1(Problem,Parent11(location1).decs,Parent12(location1).decs,Parent13(location1).decs,numberOfGroups, typeOfGroups);
        Offspring = [Offspring,Off];
    end
    if ~isempty(find(location1==0, 1))
        Off       = DE_rand_1(Problem,Parent11(~location1).decs,Parent12(~location1).decs,Parent13(~location1).decs);
        Offspring = [Offspring,Off];
    end

    %% Convergence evolution 
    Parent21 = S2(randperm(N/2));
    Parent22 = S1(randperm(N/2));
    if ~isempty(find(location2==1, 1))
        Off       = Group_DE_best_1(Problem,Parent21(location2).decs,Parent22(location2).decs,numberOfGroups, typeOfGroups);
        Offspring = [Offspring,Off];
    end
    if ~isempty(find(location2==0, 1))
        Off       = DE_best_1(Problem,Parent21(~location2).decs,Parent22(~location2).decs);
        Offspring = [Offspring,Off];
    end
end

function Offspring = Group_DE_rand_1(Problem,Parent1,Parent2,Parent3,numberOfGroups, typeOfGroups)
    [N,D] = size(Parent1);
    Fm    = [0.6,0.8,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));

    %% Differental evolution
    [outIndexList,~] = CreateGroups(numberOfGroups,Parent1,Problem.D,typeOfGroups);
    chosengroups     = randi(numberOfGroups,size(outIndexList,1),1);
    Site             = outIndexList == chosengroups;
    Offspring        = Parent1;
    Offspring(Site)  = Offspring(Site) + F(Site).*(Parent2(Site)-Parent3(Site));
    
    [proM,disM] = deal(1,20);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    mu    = rand(N,1);
    mu    = repmat(mu,1,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(Offspring);
end

function Offspring = DE_rand_1(Problem,Parent1,Parent2,Parent3)
    [N,D] = size(Parent1);
    Fm    = [0.6,0.8,1.0];
    CRm   = [0.1,0.2,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));
    index = randi([1,length(CRm)],N,1);
    CR    = CRm(index);
    CR    = CR';

    %% Differental evolution
    Site            = rand(N,D) < repmat(CR,1,D);
    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F(Site).*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(Offspring);
end

function Offspring = Group_DE_best_1(Problem,Parent1,Parent2,numberOfGroups,typeOfGroups)
    [N,D] = size(Parent1);
    Fm    = [0.6,0.8,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));

    %% Differental evolution
    [outIndexList,~] = CreateGroups(numberOfGroups,Parent1,D,typeOfGroups);
    chosengroups     = randi(numberOfGroups,size(outIndexList,1),1);
    Site             = outIndexList == chosengroups;
    Offspring        = Parent1;
    Offspring(Site)  = Offspring(Site) + F(Site).*(Parent2(Site)-Offspring(Site));

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    mu    = rand(N,1);
    mu = repmat(mu,1,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(Offspring);
end

function Offspring = DE_best_1(Problem,Parent1,Parent2)
    [N,D] = size(Parent1);
    Fm    = [0.6,0.8,1.0];
    CRm   = [0.1,0.2,1.0];
    index = randi([1,length(Fm)],N,1);
    F     = Fm(index);
    F     = F';
    F     = F(:, ones(1,D));
    index = randi([1,length(CRm)],N,1);
    CR    = CRm(index);
    CR    = CR';

    %% Differental evolution
    Site            = rand(N,D) < repmat(CR,1,D);
    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F(Site).*(Parent2(Site)-Offspring(Site));

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(Offspring);
end
```

### `gnR1R2R3.m`
```matlab
function [r1,r2,r3] = gnR1R2R3(NP1, r0)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    NP0 = length(r0);
    r1  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 1001
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else   % regenerate r1 if it is equal to r0
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000 % this has never happened so far
            error('Can not genrate r1 in 1000 iterations');
        end
    end

    r2  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 1001
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else   % regenerate r2 if it is equal to r0 or r1
            r2(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000 % this has never happened so far
            error('Can not genrate r2 in 1000 iterations');
        end
    end

    r3  = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : 1001
        pos = ((r3 == r1) | (r3 == r0) | (r3 == r2));
        if sum(pos) == 0
            break;
        else   % regenerate r2 if it is equal to r0 or r1
            r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000 % this has never happened so far
            error('Can not genrate r3 in 1000 iterations');
        end
    end
end
```
