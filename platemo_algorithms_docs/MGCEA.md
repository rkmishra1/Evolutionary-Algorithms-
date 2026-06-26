# MGCEA

**Tags**: <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>

## Description
Multi-granularity clustering based evolutionary algorithm

## Reference
Y. Tian, S. Shao, G. Xie, and Y. Jin. A multi-granularity clustering based evolutionary algorithm for large-scale sparse multi-objective optimization. Swarm and Evolutionary Computation, 2024, 84: 101453.

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

    Stage = ceil(Problem.FE/(Problem.maxFE/10));
    if Stage ~= NearStage
        NearStage = Stage;
        [FitnessLayer,LayerMax] = UpdateLayer(SparseRate,Stage,Fitness,Problem,Mask);
    end        
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
        Fitness  = zeros(5,Problem.D);
        Interval = (Problem.upper-Problem.lower)./5;
        for i = 1 : 5
            for j = 1 : 2
                Dec          = unifrnd(repmat(Problem.lower + Interval*(i-1),Problem.D,1),repmat(Problem.lower + Interval*(i),Problem.D,1));
                Mask         = eye(Problem.D);
                Population   = Problem.Evaluation(Dec.*Mask);
                TDec         = [TDec;Dec];
                TMask        = [TMask;Mask];
                TempPop      = [TempPop,Population];
                Fitness(i,:) = Fitness(i,:) + NDSort([Population.objs,Population.cons],inf);
            end        
        end
        % To reduce computation and support parallelism
        if Problem.D > 2000
            AllSample   = randperm(length(TempPop));
            FinalSample = AllSample(1:Problem.D);
            TempPop     = TempPop(FinalSample);
            TDec        = TDec(FinalSample,:);
            TMask       = TMask(FinalSample,:);
        end
        FitnessInit = sum(Fitness);
        FitnessOpt  = kmeans(sum(Fitness)',2); 
        [SparseRate,FitnessOpt] = ClusterLabel(FitnessInit,FitnessOpt);
        FitnessOpt = FitnessOpt';
    else
        Fitness = zeros(1,Problem.D);
        for i = 1
            Dec          = ones(Problem.D,Problem.D);
            Mask         = eye(Problem.D);
            Population   = Problem.Evaluation(Dec.*Mask);
        	TDec         = [TDec;Dec];
          	TMask        = [TMask;Mask];
           	TempPop      = [TempPop,Population];
           	Fitness(i,:) = Fitness(i,:) + NDSort([Population.objs,Population.cons],inf);
        end        
        FitnessInit = Fitness;
        FitnessOpt  = kmeans((Fitness)',2); 
        [SparseRate,FitnessOpt] = ClusterLabel(FitnessInit,FitnessOpt);
        FitnessOpt  = FitnessOpt';
        Mask        = zeros(Problem.N,Problem.D);
        Dec         = ones(Problem.N,Problem.D);
        for i = 1 : Problem.N
            Mask(i,TournamentSelection(2,ceil(rand*Problem.D),FitnessInit)) = 1;
        end           
        Population = Problem.Evaluation(Dec.*Mask);
        TDec       = [TDec;Dec];
        TMask      = [TMask;Mask];
        TempPop    = [TempPop,Population];
    end   
end

function [SparseRate,Fitness] = ClusterLabel(FitnessInit,Fitness)
    Num1 = sum(Fitness == 1);
    Num2 = sum(Fitness == 2);
    Num1Value = sum(FitnessInit(Fitness == 1));
    Num2Value = sum(FitnessInit(Fitness == 2));
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

### `MGCEA.m`
```matlab
classdef MGCEA < ALGORITHM
% <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Multi-granularity clustering based evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, S. Shao, G. Xie, and Y. Jin. A multi-granularity clustering
% based evolutionary algorithm for large-scale sparse multi-objective
% optimization. Swarm and Evolutionary Computation, 2024, 84: 101453.
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
            [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection(TempPop,TDec,TMask,Problem.N);            
            NearStage = ceil(Problem.FE/(Problem.maxFE/10));
            [FitnessLayer,LayerMax] = UpdateLayer(SparseRate,NearStage,Fitness,Problem,[]);

            %% Optimization           
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,2*Problem.N,FitnessSpea2);
                [NearStage,Fitness,FitnessLayer,LayerMax] = ControlStage(SparseRate,NearStage,Mask,Dec,Fitness,FitnessLayer,LayerMax,Problem);
                [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),FitnessLayer,LayerMax);
                Offspring = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FitnessSpea2] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
            end                       
        end
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,FitnessLayer,LayerMax)
% The operator of SparseEA

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

    OffMask        = Parent1Mask;
    SparseRate1    = sum(Parent1Mask,2);
    SparseRate2    = sum(Parent2Mask,2);
    Rate           = SparseRate1./(SparseRate1 + SparseRate2);
    index          = rand(N/2,D) > repmat(Rate/2,1,D);
    OffMask(index) = Parent2Mask(index);
    
    for i = 1 : N/2
        PointUp   = 1;
        PointDown = LayerMax;
        for j = 1 : LayerMax      
            TargetUpLayer   = find(FitnessLayer == PointUp);
            TargetUp        = TargetUpLayer(OffMask(i,TargetUpLayer) == 0);
            TargetDownLayer = find(FitnessLayer == PointDown);           
            TargetDown      = TargetDownLayer(OffMask(i,TargetDownLayer) == 1);
            if rand < 0.5
                if ~isempty(TargetUp)
                    if rand < 0.5
                        TargetUp = datasample(TargetUp,ceil(length(TargetUp)/2));
                        OffMask(i,TargetUp) = 1;
                    end
                end
                if rand < 0.5
                    PointUp = PointUp + 1;
                else
                    break;
                end
            else
                if ~isempty(TargetDown)
                    if rand < 0.5
                        TargetDown = datasample(TargetDown,ceil(length(TargetDown)/2));
                        OffMask(i,TargetDown) = 0;
                    end
                end
                if rand < 0.5
                    PointDown = PointDown - 1;
                else
                    break;
                end
            end
            if PointUp >= PointDown
                break;
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

    GroupNum = ceil(11 - Stage)/100*Problem.D;
    GroupNum = ceil(SparseRate*10*1*GroupNum);
    if sum(sum(Mask)) == 0
        [~,FitnessIndex] = sort(Fitness + rand(1,Problem.D));
    else
        [~,FitnessIndex] = sort(Fitness + sum(Mask == 0)./100000);
    end
    FitnessIndexLayer = ceil((1:Problem.D)./GroupNum);
    FitnessLayer      = zeros(1,Problem.D);
    FitnessLayer(FitnessIndex) = FitnessIndexLayer;
    LayerMax = max(FitnessLayer);
end
```
