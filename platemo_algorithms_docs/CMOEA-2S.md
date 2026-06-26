# CMOEA-2S

**Tags**: <2025> <multi> <real> <constrained>

## Description
Constrained MOEA with two types of evolution stages

## Reference
L. Si, X. Zhang, Y. Zhang, Y. Tian, and S. Yang. Reinforcement learning- assisted multi-stage evolutionary constrained multi-objective optimization. ACM Transactions on Evolutionary Learning, 2025.

## Source Code

### `AdaSearch.m`
```matlab
function Offspring = AdaSearch(Problem, Population, Fitness, IsTwo)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if IsTwo
        MatingPool  = TournamentSelection(2,Problem.N,Fitness);
        MatingPool0 = TournamentSelection(2,Problem.N/2,Fitness);
        Offspring   = OperatorDE(Problem,Population(MatingPool0),Population(MatingPool(1:end/2)),Population(MatingPool(end/2+1:end)));
    else
        MatingPool1 = TournamentSelection(2,Problem.N/2,Fitness);
        MatingPool2 = TournamentSelection(2,Problem.N/2,Fitness);
        MatingPool0 = TournamentSelection(2,Problem.N/2,Fitness);
        Offspring   = OperatorDE(Problem,Population(MatingPool0),Population(MatingPool1),Population(MatingPool2));
    end
end
```

### `AdaSelection.m`
```matlab
function [Population,Fitness, Next] = AdaSelection(Population,N,isOrigin)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    if isOrigin
        Fitness = AdaFitness(Population.objs, Population.cons);
    else
        Fitness = AdaFitness(Population.objs);
    end

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
    Population = Population(Next);
    Fitness    = Fitness(Next);
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

### `ApplyRL.m`
```matlab
function [Prob1,Prob2] = ApplyRL(State, Model, Paras)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    TestX1 = [State,1]; % 1-th action
    TestX2 = [State,2]; % 2-th action
    ps     = Paras.ps;
    qs     = Paras.qs;
    TestX1 = mapminmax('apply',TestX1',ps);
    TestX1 = TestX1';
    TestX2 = mapminmax('apply',TestX2',ps);
    TestX2 = TestX2';

    Prob1 = testNet(TestX1,Model,Paras);
    Prob1 = mapminmax('reverse',Prob1',qs);
    Prob1 = Prob1';

    Prob2 = testNet(TestX2,Model,Paras);
    Prob2 = mapminmax('reverse',Prob2',qs);
    Prob2 = Prob2';
end
```

### `BuildRL.m`
```matlab
function [model,Paras] = BuildRL(Records,StateNum)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % Build DQN model
    TrainX = Records(:,1:StateNum+1);
    TrainY = Records(:,StateNum+2:end);
    % Normalization
    [TrainX,ps]   = mapminmax(TrainX');
    TrainX        = TrainX';
    [TrainY,qs]   = mapminmax(TrainY');
    TrainY        = TrainY';
    Paras.ps      = ps;
    Paras.qs      = qs;
    [model,Paras] = trainmodel(TrainX,TrainY,Paras);
end
```

### `CMOEA2S.m`
```matlab
classdef CMOEA2S < ALGORITHM
% <2025> <multi> <real> <constrained>
% Constrained MOEA with two types of evolution stages

%------------------------------- Reference --------------------------------
% L. Si, X. Zhang, Y. Zhang, Y. Tian, and S. Yang. Reinforcement learning-
% assisted multi-stage evolutionary constrained multi-objective
% optimization. ACM Transactions on Evolutionary Learning, 2025.
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
            %% Initialization
            ArcSize = 2*Problem.N;
            ArcDec  = unifrnd(repmat(Problem.lower, ArcSize, 1), repmat(Problem.upper, ArcSize, 1)); 
            Archive = Problem.Evaluation(ArcDec);

            %% DQN model
            Learning   = 0.2;
            ModelExist = false;
            Counter    = 1;
            Greedy     = 0.1;
            Gama       = 0.9;
            Records    = [];
            ReSize     = 20;

            %% Action: 1-CDP, 2-MOP, 3-CDP+MOP
            ActionNum  = 2;
            StateNum   = 3;
            Iter       = 5;
            UpdateIter = 20;
            % Initial action
            Action = 1;

            %% Optimization
            while Algorithm.NotTerminated(Archive)
                %% Excute the action in iter iterations
                [ArchiveU, ratio] = ExcuteAction(Problem, Archive, Action, Iter);
             
                %% Recording
                % State of old Archive
                State = EstimateState(Archive,StateNum);

                % State of new Archive
                StateU = EstimateState(ArchiveU,StateNum);
                % Reward
                Reward = EstimateReward(Problem, Archive, ArchiveU, ratio,0,0,0);
                % Record
                Record = [State,Action,Reward,StateU];
                
                % Update Archive
                Archive = ArchiveU;

                % Update Records
                Records(mode(Counter-1,ReSize)+1,:) = Record;
                Counter = Counter + 1;

                %% Action selection
                if Problem.FE < Learning*Problem.maxFE
                    % Learning process
                    % Data collection
                    if Problem.FE/Problem.maxFE < Learning*0.2
                        % Random selection
                        Action = randi(ActionNum);
                    else
                        % DQN based selection
                        if ~ModelExist
                            % Build DQN model
                            [Model,Paras] = BuildRL(Records,StateNum);
                            ModelExist    = true;
                            Action        = randi(ActionNum);
                        else
                            if rand() < Greedy
                                % Random selection
                                Action = randi(ActionNum);
                            else
                                % DQN based selection
                                [Prob1,Prob2] = ApplyRL(State, Model, Paras);
                                Probs         = [Prob1;Prob2];
                                [~,Action]    = max(Probs(:,1));
                            end
                        end
                    end
                else
                    % Applying process
                    if rand() < Greedy
                        % Random selection
                        Action = randi(ActionNum);
                    else
                        % DQN based selection
                        [Prob1,Prob2] = ApplyRL(State, Model, Paras);
                        Probs         = [Prob1;Prob2];
                        [~,Action]    = max(Probs(:,1));
                    end
                end
                % Update model every UpdateIter iterations
                if ModelExist
                    if mode(Counter, UpdateIter) == 0
                        [Model, Paras] = UpdateRL(Records, Model, Paras,Gama, StateNum);
                    end
                end
            end
        end
    end
end
```

### `EstimateReward.m`
```matlab
function Reward = EstimateReward(Problem, ArchiveOld, ArchiveNew, ratio,Cons0,score_HV0,optimum0)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Phi     = Problem.FE/Problem.maxFE;
    PopObj1 = best(ArchiveOld);
    PopObj2 = best(ArchiveNew);

    if isempty(PopObj1) || isempty(PopObj2)
        Cons1  = sum(sum(max(ArchiveOld.cons,0),2));
        Cons2  = sum(sum(max(ArchiveNew.cons,0),2));
        Reward = (Cons1-Cons2)/Cons1;
    else
        PopObj1 = ArchiveOld(all(ArchiveOld.cons<=0,2));
        PopObj2 = ArchiveNew(all(ArchiveNew.cons<=0,2));

        [~,k] = size(PopObj1);
        for i = 1 : k
            obj1(i,:) = PopObj1(i).obj;
        end
        [~,k] = size(PopObj2);
        for i = 1 : k
            obj2(i,:) = PopObj2(i).obj;
        end
        [~,m] = size(obj1);
        for i = 1 : m
            optimum1(i) = max(obj1(:,i));
            optimum2(i) = max(obj2(:,i));
        end
        optimum1  = optimum1 * 1.1;
        score_HV1 = HV(ArchiveOld,optimum1);
        score_HV2 = HV(ArchiveNew,optimum1);
        score_HV  = (score_HV2 - score_HV1)/(score_HV1);

        score_PD1 = PD(ArchiveOld);
        score_PD2 = PD(ArchiveNew);
        score_PD  = (score_PD2 - score_PD1)/(score_PD1);

        Reward = Phi * score_PD + (1-Phi) * score_HV;
    end
end
```

### `EstimateState.m`
```matlab
function State = EstimateState(Archive,StateNum)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    State    = zeros(1, StateNum);
    Cons     = sum(max(Archive.cons,0),2);
    FIndex   = Cons == 0;
    State(1) = mean(Cons);
    if any(FIndex)
        [DNum,NNum] = Detect(Archive, FIndex);
        State(2)    = DNum/numel(Archive);
        State(3)    = NNum/numel(Archive);
    end    
end

function [DNum, NNum] = Detect(Archive, FIndex)
    DNum = 0;
    NNum = 0;
    FSet = Archive(FIndex);
    FNum = numel(FSet);
    ISet = Archive(~FIndex);
    INum = numel(ISet);
    for i = 1 : INum
        Del       = all(repmat(ISet(i).objs, FNum, 1) < FSet.objs, 2);
        FSet(Del) = [];
        DNum      = DNum + sum(Del);
        FNum      = numel(FSet);
        if FNum == 0
            return;
        end
    end
    NNum = sum(FIndex) - DNum;
end
```

### `ExcuteAction.m`
```matlab
function [Archive, ratio] = ExcuteAction(Problem, Archive, Action, Iter)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Form population
    switch Action
        case 1
            % Constrained optimization
            [Population1, ~, PopIndex] = FormPopulation(Archive, Action, Problem.N);
            Fitness1 = AdaFitness(Population1.objs, Population1.cons);
        case 2
            % Unconstrained optimization
            [~, Population2, PopIndex] = FormPopulation(Archive, Action, Problem.N);
            Fitness2 = AdaFitness(Population2.objs);
        case 3
            % Contrained and unconstrained optimization
            [Population1, Population2, PopIndex] = FormPopulation(Archive, Action, Problem.N);
            Fitness1 = AdaFitness(Population1.objs, Population1.cons);
            Fitness2 = AdaFitness(Population2.objs);
        otherwise
            error('Invalid action!');
    end
 
    %% optimization
    for i = 1 : Iter
        switch Action
            case 1
                % Constrained optimization
                Offspring1 = AdaSearch(Problem, Population1, Fitness1, false);
                [Population1,Fitness1] = AdaSelection([Population1,Offspring1],numel(Population1),true);
                if i == Iter
                    Population = Population1;
                end
            case 2
                % Unconstrained optimization
                Offspring2 = AdaSearch(Problem, Population2, Fitness2, false);
                [Population2,Fitness2] = AdaSelection([Population2,Offspring2],numel(Population2),false);
                if i == Iter
                    Population = Population2;
                end
            case 3
                % Contrained and unconstrained optimization
                Offspring1 = AdaSearch(Problem, Population1, Fitness1, true);
                Offspring2 = AdaSearch(Problem, Population2, Fitness2, true);
                [Population1,Fitness1] = AdaSelection([Population1,Offspring1, Offspring2],numel(Population1),true);
                [Population2,Fitness2] = AdaSelection([Population2,Offspring2, Offspring1],numel(Population2),false);
                if i == Iter
                    Population = [Population1, Population2];
                end            
            otherwise
                error('Invalid action!');
        end
    end
    if Action == 1 || Action == 2
        Archive(PopIndex) = Population;
    else
        [~, Index, ~] = unique(Population.decs, 'rows');
        Archive = Population(Index);
    end
    ratio = numel(Archive)/(1 + numel(Population));
end
```

### `FormPopulation.m`
```matlab
function [Population, PopulationD, PopIndex] = FormPopulation(Archive, Action, PopSize)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    ArcSize = numel(Archive);

    %% Form population(s)
    if ArcSize < PopSize
        PopIndex    = 1 : 1 : ArcSize;
        Population  = Archive;
        PopulationD = Archive;
    else
        if Action == 1
            [Population,~, PopIndex] = AdaSelection(Archive,PopSize,true);
            PopulationD = [];
        elseif Action == 2
            if rand()<0.1
                [~,~, PopIndex] = AdaSelection(Archive,PopSize,true);
                PopIndex        = ~PopIndex;
                PopulationD     = Archive(PopIndex);
            else
                [PopulationD,~, PopIndex] = AdaSelection(Archive,PopSize,false);
            end
            Population = [];
        end
    end
end
```

### `UpdateRL.m`
```matlab
function [Model, Paras] = UpdateRL(Records, Model, Paras, Gama, StateNum)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % update model here
    qs     = Paras.qs;
    TrainX = Records(:,1:StateNum+1);
    % Normalization
    [TrainX,ps] = mapminmax(TrainX');
    TrainX      = TrainX';

    Prob = testNet(TrainX,Model,Paras);
    Prob = mapminmax('reverse',Prob',qs);
    Prob = Prob';
    Prob = Prob(:,1);

    % update model here
    TrainY      = Records(:,StateNum+2)+Gama*max(Prob);
    [TrainY,qs] = mapminmax(TrainY');
    TrainY      = TrainY';
    Paras.ps    = ps;
    Paras.qs    = qs;
    Model       = updatemodel(TrainX,TrainY,Paras,Model);
end
```
