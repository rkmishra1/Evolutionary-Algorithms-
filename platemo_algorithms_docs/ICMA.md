# ICMA

**Tags**: <2022> <multi> <real/integer> <constrained>

## Description
Indicator-based constrained multi-objective algorithm

## Reference
J. Yuan, H. Liu, Y. Ong, and Z. He. Indicator-based evolutionary algorithm for solving constrained multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2022, 26(2): 379-391.

## Source Code

### `ICMA.m`
```matlab
classdef ICMA < ALGORITHM
% <2022> <multi> <real/integer> <constrained>
% Indicator-based constrained multi-objective algorithm

%------------------------------- Reference --------------------------------
% J. Yuan, H. Liu, Y. Ong, and Z. He. Indicator-based evolutionary
% algorithm for solving constrained multi-objective optimization problems.
% IEEE Transactions on Evolutionary Computation, 2022, 26(2): 379-391.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    methods
        function main(Algorithm,Problem)
            %% Generate the random population
            Population = Problem.Initialization();
            Zmin       = min(Population.objs,[],1);
            Fmin       = min(Population(all(Population.cons<=0,2)).objs,[],1);
            Archive    = Population;
            W          = UniformPoint(Problem.N,Problem.M);
            Ra         = 1;
            
            %% Optimization
            while Algorithm.NotTerminated(Archive)
                Nt = floor(Ra*Problem.N);
                MatingPool = [Population(randsample(Problem.N,Nt)),Archive(randsample(Problem.N,Problem.N-Nt))];
                
                [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPool,Zmin);
                if rand > 0.5
                    Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3);
                else
                    Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3,{0.5,0.5,0.5,0.75});
                end
                
                Fmin = min([Fmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                Zmin = min([Zmin;Offspring.objs],[],1);
                [Population,Archive] = ICMA_Update([Population,Offspring,Archive],Problem.N,W,Zmin,Fmin);
                
                Ra = 1 - Problem.FE/Problem.maxFE;
            end
        end
    end
end
```

### `ICMA_Update.m`
```matlab
function [Population,Archive] = ICMA_Update(MaxPop,N,W,Zmin,Fmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    PopObj   = MaxPop.objs;
    [Num,M]  = size(PopObj);
    Cons     = max(0,MaxPop.cons);
    NormCons = Cons./repmat(max(1,max(Cons,[],1)),Num,1);
    CV       = sum(NormCons,2);

    % shift the objective space to R+
    PopObj = PopObj - repmat(Zmin,Num,1) + 1e-6;

    % calculate the indicator matrix
    IMatrix = ones(Num,Num);
    for i = 1 : 1:Num
        Ci = CV(i);    
        if Ci == 0 % Xi is feasible
            Fi           = PopObj(i,:);
            Ir           = log(repmat(Fi,Num,1)./PopObj);
            MaxIr        = max(Ir,[],2);
            MinIr        = min(Ir,[],2);
            CVA          = MaxIr;
            DomInds      = find(MaxIr<=0);
            CVA(DomInds) = MinIr(DomInds);
            IndicatorV   = CVA;
        else  % Xi is an infeasible solution
            IC         = repmat(Ci+1e-6,Num,1)./(CV+1e-6); 
            Fi         = PopObj(i,:);
            MaxF       = max(repmat(Fi,Num,1),PopObj);
            MinF       = min(repmat(Fi,Num,1),PopObj);
            CVF        = max(MaxF./MinF,[],2);
            IndicatorV = log(max([CVF,IC],[],2));
        end
        IMatrix(:,i) = IndicatorV;
        IMatrix(i,i) = Inf;
    end

    FeasibleInd = find(CV==0);
    Len_F       = length(FeasibleInd);

    if Len_F <= N
        [~,CV_SortInd] = sort(CV);
        Archive        = MaxPop(CV_SortInd(1:N));
    else
        FPopObj = PopObj(FeasibleInd,:) + repmat(Zmin,Len_F,1) - repmat(Fmin,Len_F,1);
        SelInd  = Selection_Operator_of_PREA(FPopObj,IMatrix(FeasibleInd,FeasibleInd),N);
        Archive = MaxPop(FeasibleInd(SelInd));
    end

    %  using indicator-based CHT to update the population
    SelInd     = Indicator_based_CHT(PopObj,IMatrix,W,N);
    Population = MaxPop(SelInd);
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

% This function is written by Jiawei Yuan

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

### `Neighbor_Pairing_Strategy.m`
```matlab
function [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPop,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    Objs    = MatingPop.objs;
    [Num,M] = size(Objs);
    Objs    = (Objs - repmat(Zmin,Num,1));
    Objs    = Objs./repmat(sqrt(sum(Objs.^2,2)),1,M);

    CosV = Objs * Objs';
    CosV = CosV - 3*eye(Num,Num);

    [~,SInd] = sort(-CosV,2);

    Nr       = 10;
    Neighbor = SInd(:,1:Nr);

    Mate1 = MatingPop;

    P = ones(Num,2);
    for i = 1 : Num
        P(i,1:2) = Neighbor(i,randsample(Nr,2));
        if rand>0.7
            P(i,2) = randsample(Num,1);
        end
    end

    Mate2 = MatingPop(P(:,1));
    Mate3 = MatingPop(P(:,2));
end
```

### `Selection_Operator_of_PREA.m`
```matlab
function NextIndex = Selection_Operator_of_PREA(PopObj,IMatrix,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jiawei Yuan

    [~,M] = size(PopObj);

    % calculate the size of individuals in the first nondominant level
    IrFitness   = min(IMatrix,[],2);
    Level1Index = find(IrFitness>=0);
    Len_Level1  = length(Level1Index);

    if Len_Level1<=N
        [~,SortIndex] = sort(-IrFitness);
        NextIndex     = SortIndex(1:N);
    else
        % only focus on the solutions in the first level
        AllIndex = Level1Index;
        PopObj   = PopObj(Level1Index,:);
        IMatrix  = IMatrix(Level1Index,Level1Index);

        %% select the valuable solutions in the current population
        MiddleIMatrix       = IMatrix;
        Ag1_n               = Len_Level1 - N;
        [Values,Neightboor] = min(MiddleIMatrix,[],2);
        BestInd             = 1 : Len_Level1;
        Have_Delect         = zeros(1,Ag1_n);

        % mark the N solutions with the best fitness value by excluding the
        % worst individuals one by one
        for i = 1 : Ag1_n
            [~,Del_Ind]    = min(Values);
            Have_Delect(i) = Del_Ind;
            MiddleIMatrix(Del_Ind,:) = Inf;
            MiddleIMatrix(:,Del_Ind) = Inf;
            Need_Updata = find(Neightboor==Del_Ind);
            L_Need      = length(Need_Updata);
            if L_Need > 0
                [Values(Need_Updata),Neightboor(Need_Updata)] = min(MiddleIMatrix(Need_Updata,:),[],2);
            end
            Values(Del_Ind) = Inf;
        end
        BestInd(Have_Delect) = [];

        % determine the boundary of promising region
        Zmax = max(PopObj(BestInd,:),[],1);

        % remove the individuals outside the promising region
        OutIndex            = find(min(repmat(Zmax,Len_Level1,1) - PopObj,[],2) < 0);
        AllIndex(OutIndex)  = [];
        PopObj(OutIndex,:)  = [];
        IMatrix(OutIndex,:) = [];
        IMatrix(:,OutIndex) = [];
        Num                 = length(AllIndex);


        %% diversity maintance mechanism based on parallel distance
        % normalize the promising region
        PopObj = PopObj./repmat(Zmax,Num,1);

        [Ir_Values,Ir_Neightboor] = min(IMatrix,[],2);

        DelectInd2 = [];

        DMatrix = zeros(Num,Num);

        % calculate the parallel distance matrix
        for i = 1 : Num
            Fi           = PopObj(i,:);
            Fdelta       = PopObj - repmat(Fi,Num,1);
            DMatrix(i,:) = sqrt(sum(Fdelta.^2,2) - (sum(Fdelta,2)).^2./M);
            DMatrix(i,i) = Inf;
        end

        [Dis_Values,Dis_Neightboor] = min(DMatrix,[],2);


        for l = 1 : (Num - N)
            [~,individual1] = min(Dis_Values);
            individual2     = Dis_Neightboor(individual1);

            if Ir_Values(individual1) < Ir_Values(individual2)
                k = individual1;
            else
                k = individual2;
            end

            DelectInd2 = [DelectInd2,k];

            DMatrix(k,:)    = Inf;
            DMatrix(:,k)    = Inf;
            Need_Updata_Dis = find(Dis_Neightboor==k);
            L_Need_Dis      = length(Need_Updata_Dis);
            if L_Need_Dis > 0
                [Dis_Values(Need_Updata_Dis),Dis_Neightboor(Need_Updata_Dis)] = min(DMatrix(Need_Updata_Dis,:),[],2);
            end
            Dis_Values(DelectInd2) = Inf;



            IMatrix(k,:)   = Inf;
            IMatrix(:,k)   = Inf;
            Need_Updata_Ir = find(Ir_Neightboor==k);
            L_Need_Ir      = length(Need_Updata_Ir);
            if L_Need_Ir > 0
                [Ir_Values(Need_Updata_Ir),Ir_Neightboor(Need_Updata_Ir)] = min(IMatrix(Need_Updata_Ir,:),[],2);
            end
        end
        AllIndex(DelectInd2) = [];
        NextIndex            = AllIndex;
    end
end
```
