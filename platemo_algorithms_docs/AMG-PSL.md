# AMG-PSL

**Tags**: <2025> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>

## Description
Adaptive multi-granular Pareto-optimal subspace learning

## Reference
C. Sun, Y. Tian, S. Shao, S. Yang, and X. Zhang. An adaptive multi- granular Pareto-optimal subspace learning algorithm for sparse large- scale multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2025.

## Source Code

### `AMGPSL.m`
```matlab
classdef AMGPSL < ALGORITHM
% <2025> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
% Adaptive multi-granular Pareto-optimal subspace learning

%------------------------------- Reference --------------------------------
% C. Sun, Y. Tian, S. Shao, S. Yang, and X. Zhang. An adaptive multi-
% granular Pareto-optimal subspace learning algorithm for sparse large-
% scale multi-objective optimization. Proceedings of the IEEE Congress on
% Evolutionary Computation, 2025.
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
            [~,Fitness,SparseRate,TDec,TMask,TempPop] = FitnessCal(Problem);
            [Population,Dec,Mask,FitnessSpea2]        = EnvironmentalSelection(TempPop,TDec,TMask,Problem.N);
            
            %% Initialize stage parameters
            NearStage               = ceil(Problem.FE/(Problem.maxFE/10));
            [FitnessLayer,LayerMax] = UpdateLayer(SparseRate,NearStage,Fitness,Problem,[]);
            rho = 0.5;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Model training phase
                Site = rho > rand(1,ceil(Problem.N/2));
                if any(Site)
                    [rbm,dae,allZero,allOne] = ModelTraining(Mask,Dec,any(Problem.encoding~=4),Problem);
                else
                    [rbm,dae,allZero,allOne] = deal([]);
                end
                
                % Mating selection & Update layers
                MatingPool = TournamentSelection(2,2*Problem.N,FitnessSpea2);
                [NearStage,Fitness,FitnessLayer,LayerMax] = ControlStage(SparseRate,NearStage,Mask,Dec,Fitness,FitnessLayer,LayerMax,Problem);
                
                % Generate offspring
                [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),rbm,dae,Site,allZero,allOne,FitnessLayer,LayerMax);
                Offspring = Problem.Evaluation(OffDec.*OffMask);
                
                % Environmental selection
                [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
                
                % Update rho adaptively
                success = false(1,length([Population,Offspring]));
                success(1:length(Population)) = true;
                s1  = sum(success(length(Population)+1:end))/2;
                s2  = sum(~success(length(Population)+1:end))/2;
                rho = (rho + min(max((s1+1e-6)/(s1+s2+1e-6),0.1),0.9))/2;
            end
        end
    end
end
```

### `ControlStage.m`
```matlab
function [NearStage,Fitness,FitnessLayer,LayerMax] = ControlStage(SparseRate,NearStage,Mask,Dec,Fitness,FitnessLayer,LayerMax,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Update stage
    Stage = ceil(Problem.FE/(Problem.maxFE/10));
    if (Stage ~= NearStage)
        NearStage = Stage;
        [FitnessLayer,LayerMax] = UpdateLayer(SparseRate,Stage,Fitness,Problem,Mask);
    end        
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection(Population,Dec,Mask,N)

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
    
    %% Calculate fitness for each solution using SPEA2 method
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
    
    %% Population for next generation
    Population   = Population(Next);
    FitnessSpea2 = Fitness(Next);
    Dec          = Dec(Next,:);
    Mask         = Mask(Next,:);
end

function Fitness = CalFitness(PopObj)
    N = size(PopObj,1);
    
    %% Calculate dominance relation
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
    
    %% Calculate final fitness
    Fitness = R + D';
end

function Del = Truncation(PopObj,K)
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
function [FitnessInit,FitnessOpt,SparseRate,TDec,TMask,TempPop] = FitnessCal(Problem)

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
        %% Multi-interval sampling for real variables
        Fitness  = zeros(5,Problem.D);
        Interval = (Problem.upper-Problem.lower)./5;
        for i = 1 : 5
            for j = 1 : 2
                Dec = unifrnd(repmat(Problem.lower + Interval*(i-1),Problem.D,1),...
                            repmat(Problem.lower + Interval*(i),Problem.D,1));
                Mask = eye(Problem.D);
                Population = Problem.Evaluation(Dec.*Mask);
                TDec = [TDec;Dec];
                TMask = [TMask;Mask];
                TempPop = [TempPop,Population];
                Fitness(i,:) = Fitness(i,:) + NDSort([Population.objs,Population.cons],inf);
            end        
        end
        
        %% Sample reduction for large-scale problems
        if Problem.D > 2000
            AllSample = randperm(length(TempPop));
            FinalSample = AllSample(1:Problem.D);
            TempPop = TempPop(FinalSample);
            TDec = TDec(FinalSample,:);
            TMask = TMask(FinalSample,:);
        end
        
        %% Calculate fitness
        FitnessInit = sum(Fitness);
        FitnessOpt = kmeans(sum(Fitness)',2);
        [SparseRate,FitnessOpt] = ClusterLabel(FitnessInit,FitnessOpt);
        FitnessOpt = FitnessOpt';
    else
        %% Binary variable handling
        Fitness = zeros(1,Problem.D);
        Dec = ones(Problem.D,Problem.D);
        Mask = eye(Problem.D);
        Population = Problem.Evaluation(Dec.*Mask);
        TDec = Dec;
        TMask = Mask;
        TempPop = Population;
        Fitness = NDSort([Population.objs,Population.cons],inf);
        
        FitnessInit = Fitness;
        FitnessOpt = kmeans(Fitness',2);
        [SparseRate,FitnessOpt] = ClusterLabel(FitnessInit,FitnessOpt);
        FitnessOpt = FitnessOpt';
    end   
end

function [SparseRate,Fitness] = ClusterLabel(FitnessInit,Fitness)
    %% Calculate cluster ratios and values
    Num1 = sum(Fitness == 1);
    Num2 = sum(Fitness == 2);
    Num1Value = sum(FitnessInit(Fitness == 1));
    Num2Value = sum(FitnessInit(Fitness == 2));
    
    %% Determine important variables
    if Num1Value < Num2Value
        SparseRate = Num1/(Num1 + Num2);
        Fitness(Fitness == 1) = 11;
        Fitness(Fitness == 2) = 12;
    else   
        SparseRate = Num2/(Num1 + Num2);
        Fitness(Fitness == 1) = 12;
        Fitness(Fitness == 2) = 11;
    end
end
```

### `ModelTraining.m`
```matlab
function [rbm,dae,allZero,allOne] = ModelTraining(Mask,Dec,REAL,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Basic variable setup
    allZero = all(~Mask,1);
    allOne  = all(Mask,1);
    other   = ~allZero & ~allOne;
    
    %% Non-dominated sorting for evaluation
    [FrontNo,MaxFront] = NDSort(Dec.*Mask,ones(size(Dec,1),1));
    
    %% Variable evaluation
    validMask = Mask(:,other);
    validDec  = Dec(:,other);
    
    if ~isempty(validMask)
        % Calculate variable contributions
        contribution = abs(validDec).*validMask;
        meanContrib  = mean(contribution,1);
        
        % Normalize meanContrib to [0, 1]
        meanContribNorm = (meanContrib - min(meanContrib)) / (max(meanContrib) - min(meanContrib) + eps);
        
        % Calculate variable ranking scores
        [~,rankIdx] = sort(meanContrib,'descend');
        rankScore   = zeros(1,length(meanContrib));
        rankScore(rankIdx) = linspace(1,0,length(rankIdx));
        
        % Analyze non-dominated solutions
        bestMask = validMask(FrontNo==1,:);
        if ~isempty(bestMask)
            successRate = mean(bestMask,1);
        else
            successRate = mean(validMask,1);
        end
        
        % Calculate comprehensive scores with normalized meanContrib
        varScore = 0.4*rankScore + 0.4*successRate + 0.2*meanContribNorm;
        
        % Adaptive threshold
        progress  = Problem.FE/Problem.maxFE;
        threshold = mean(varScore) * (1 + 0.2*(1-progress));
        Type      = varScore > threshold;
    else
        Type     = [];
        varScore = [];
    end
    
    %% Determine hidden layer size
    nValid = sum(other);
    if nValid == 0
        K = 1;
    else
        % Base K on valid variables and progress
        baseK = min(sum(Type), round(sqrt(nValid)));
        K     = min(max(round(baseK * (1-0.3*progress)), 1), size(Mask,1));
    end
    
    %% Train networks
    if nValid > 0
        rbm = RBM(nValid,K,10,1,0,0.5,0.1);
        rbm.train(Mask(:,other));
    else
        rbm = [];
    end
    
    if REAL
        dae = DAE(size(Dec,2),K,10,size(Dec,1),0.5,0.5,0.1);
        dae.train(Dec);
    else
        dae = [];
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,rbm,dae,Site,allZero,allOne,FitnessLayer,LayerMax)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    Parent1Dec  = ParentDec(1:N/2,:);
    Parent2Dec  = ParentDec(N/2+1:end,:);
    
    %% Generate mask using crossover (N/2 × D)
    OffMask     = Parent1Mask;
    SparseRate1 = sum(Parent1Mask,2);
    SparseRate2 = sum(Parent2Mask,2);
    Rate        = SparseRate1./(SparseRate1 + SparseRate2);
    
    for i = 1 : N/2
        index = rand(1,D) > Rate(i)/2;
        OffMask(i,index) = Parent2Mask(i,index);
    end
    
    %% Layer-based mutation (still N/2 × D)
    for i = 1 : N/2
        PointUp   = 1;
        PointDown = LayerMax;
        while PointUp < PointDown
            if rand < 0.5
                TargetUpLayer = find(FitnessLayer == PointUp);
                TargetUp      = TargetUpLayer(OffMask(i,TargetUpLayer) == 0);
                if ~isempty(TargetUp) && rand < 0.5
                    TargetUp = datasample(TargetUp,ceil(length(TargetUp)/2));
                    OffMask(i,TargetUp) = 1;
                end
                PointUp = PointUp + 1;
            else
                TargetDownLayer = find(FitnessLayer == PointDown);
                TargetDown      = TargetDownLayer(OffMask(i,TargetDownLayer) == 1);
                if ~isempty(TargetDown) && rand < 0.5
                    TargetDown = datasample(TargetDown,ceil(length(TargetDown)/2));
                    OffMask(i,TargetDown) = 0;
                end
                PointDown = PointDown - 1;
            end
            
            if rand < 0.5 || PointUp >= PointDown
                break;
            end
        end
    end
    
    %% Generate real variables (N/2 × D)
    if any(Problem.encoding~=4)
        if any(Site)
            % Model-based generation
            other = ~allZero & ~allOne;
            if ~isempty(rbm)
                tempMask = OffMask(:,other);
                for i = find(Site)
                    OffTemp = rbm.reduce(tempMask(i,:));
                    OffTemp = rbm.recover(OffTemp);
                    tempMask(i,:) = OffTemp;
                end
                OffMask(:,other)  = tempMask;
                OffMask(:,allOne) = true;
            end
            
            if ~isempty(dae)
                OffDec = zeros(N/2,D);
                for i = find(Site)
                    temp = dae.reduce(Parent1Dec(i,:));
                    temp = dae.recover(temp);
                    OffDec(i,:) = temp;
                end
                % Fill non-Site rows with regular crossover
                nonSite = find(~Site);
                if ~isempty(nonSite)
                    OffDec(nonSite,:) = RealCrossover(Parent1Dec(nonSite,:),Parent2Dec(nonSite,:));
                end
            else
                OffDec = RealCrossover(Parent1Dec,Parent2Dec);
            end
        else
            % Regular crossover
            OffDec = RealCrossover(Parent1Dec,Parent2Dec);
        end
        
        % Mutation and boundary handling
        OffDec = RealMutation(OffDec,Problem.lower,Problem.upper);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

function Offspring = RealCrossover(Parent1,Parent2)
    [proC,disC] = deal(1,20);
    [N,D]       = size(Parent1);
    
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta = beta.*rand(N,D);
    
    Offspring = (Parent1+Parent2)/2 + beta.*(Parent1-Parent2)/2;
end

function Offspring = RealMutation(Offspring,Lower,Upper)
    [proM,disM] = deal(1,20);
    [N,D] = size(Offspring);
    Lower = repmat(Lower,N,1);
    Upper = repmat(Upper,N,1);
    
    Site = rand(N,D) < proM/D;
    mu   = rand(N,D);
    temp = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring       = min(max(Offspring,Lower),Upper);
end
```

### `UpdateLayer.m`
```matlab
function [FitnessLayer,LayerMax] = UpdateLayer(SparseRate,Stage,Fitness,Problem,Mask)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate number of groups
    GroupNum = ceil(11 - Stage)/100*Problem.D;
    GroupNum = ceil(SparseRate*10*1*GroupNum);
    
    %% Sort fitness scores
    if sum(sum(Mask)) == 0
        [~,FitnessIndex] = sort(Fitness + rand(1,Problem.D));
    else
        [~,FitnessIndex] = sort(Fitness + sum(Mask == 0)./100000);
    end
    
    %% Assign layers
    FitnessIndexLayer          = ceil((1:Problem.D)./GroupNum);
    FitnessLayer               = zeros(1,Problem.D);
    FitnessLayer(FitnessIndex) = FitnessIndexLayer;
    LayerMax                   = max(FitnessLayer);
end
```
