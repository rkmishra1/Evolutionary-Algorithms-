# PM-MOEA

**Tags**: <2022> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>

## Description
Pattern mining based multi-objective evolutionary algorithm

## Reference
Y. Tian, C. Lu, X. Zhang, F. Cheng, and Y. Jin. A pattern mining based evolutionary algorithm for large-scale sparse multi-objective optimization problems. IEEE Transactions on Cybernetics, 2022, 52(7): 6784-6797.

## Source Code

### `BinaryCrossover.m`
```matlab
function Offspring = BinaryCrossover(Parent1,Parent2)
% Unbalanced binary crossover

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Offspring = logical(Parent1);
    for i = 1 : size(Offspring,1)
        diff = find(Parent1(i,:)~=Parent2(i,:));
        MOne = mean(Offspring(i,diff));
        r    = min(min(0.5,2*MOne),2*(1-MOne));
        rate = zeros(1,length(diff));
        rate(Offspring(i,diff))     = r/2/MOne;
        rate(~Offspring(i,diff))    = r/2/(1-MOne);
        exchange                    = rand(1,length(diff)) < rate;
        Offspring(i,diff(exchange)) = ~Offspring(i,diff(exchange));
    end
end
```

### `BinaryMutation.m`
```matlab
function Offspring = BinaryMutation(Offspring)
% Unbalanced binary mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,D] = size(Offspring);
    MOne  = mean(Offspring,2);
    r     = min(min(1/D,2*MOne),2*(1-MOne));
    rate1 = repmat(r./2./MOne,1,D);
    rate0 = repmat(r./2./(1-MOne),1,D);
    rate  = zeros(N,D);
    rate(Offspring)     = rate1(Offspring);
    rate(~Offspring)    = rate0(~Offspring);
    exchange            = rand(N,D) < rate;
    Offspring(exchange) = ~Offspring(exchange);
end
```

### `EnvironmentalSelection.m`
```matlab
function varargout = EnvironmentalSelection(varargin)
% The environmental selection of PM-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions and non-dominated sorting
    if nargin == 3
        [PopDec,PopObj,N] = deal(varargin{:});
        [PopDec,uni] = unique(PopDec,'rows','stable');
        PopObj = PopObj(uni,:);
        [FrontNo,MaxFNo] = NDSort(PopObj,N);
    elseif nargin == 4
        [Population,Dec,Mask,N] = deal(varargin{:});
        PopObj = Population.objs;
        [FrontNo,MaxFNo] = NDSort(PopObj,Population.cons,N);
    end
    Next = FrontNo <= MaxFNo;
    
    %% Truncate the solutions in the last front
    Last = find(FrontNo==MaxFNo);
    if nargin == 3
        Del = Truncation(double(PopDec(Last,:)),sum(Next)-N);
    elseif nargin == 4
        Del = Truncation(PopObj(Last,:),sum(Next)-N);
    end
    Next(Last(Del)) = false;
    
    %% Population for next generation
    FrontNo = FrontNo(Next);
    if nargin == 3
        PopDec    = PopDec(Next,:);
        PopObj    = PopObj(Next,:);
        varargout = {PopDec,PopObj,FrontNo};
    elseif nargin == 4
        Population = Population(Next);
        Dec        = Dec(Next,:);
        Mask       = Mask(Next,:);
        varargout  = {Population,Dec,Mask,FrontNo};
    end
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

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

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,MaxP,MinP,Nonzero,PopDec)
% The operator of PM-MOEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,D]       = size(ParentMask);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Delete illegal subspaces
    exist = false(size(MaxP,1),size(MinP,1));
    for i = 1 : size(MinP,1)
        exist(:,i) = all(MaxP(:,MinP(i,:)),2);
    end
    MaxP(~any(exist,2),:)  = [];
    MinP(~any(exist,1),:)  = [];

    %% Binary variation
    OffMask = false(N/2,D);
    if ~isempty(MaxP) && ~isempty(MinP)
        for i = 1 : N/2
            maxp = MaxP(randi(end),:);
            minp = MinP(randi(end),:);
            maxp = maxp & ~minp;
            OffMask(i,Nonzero(maxp)) = BinaryCrossover(Parent1Mask(i,Nonzero(maxp)),Parent2Mask(i,Nonzero(maxp)));
            OffMask(i,Nonzero(minp)) = true;
        end
    else
        OffMask = BinaryCrossover(Parent1Mask,Parent2Mask);
    end
    OffMask = BinaryMutation(OffMask);
    
    %% Real variation
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(size(OffMask));
    end
    
    %% Remove duplicated solutions
    [~,uni] = unique(OffDec.*OffMask,'rows');
    OffDec  = OffDec(uni,:);
    OffMask = OffMask(uni,:);
    del            = ismember(OffDec.*OffMask,PopDec,'rows');
    OffDec(del,:)  = [];
    OffMask(del,:) = [];
end
```

### `PMMOEA.m`
```matlab
classdef PMMOEA < ALGORITHM
% <2022> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
% Pattern mining based multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, C. Lu, X. Zhang, F. Cheng, and Y. Jin. A pattern mining based
% evolutionary algorithm for large-scale sparse multi-objective
% optimization problems. IEEE Transactions on Cybernetics, 2022, 52(7):
% 6784-6797.
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
            Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            Dec(:,Problem.encoding==4) = 1;
            Mask = false(size(Dec));
            for i = 1 : Problem.N
                Mask(i,randperm(end,ceil(rand.^2*end))) = true;
            end
            Population = Problem.Evaluation(Dec.*Mask);
            [Population,Dec,Mask,FrontNo] = EnvironmentalSelection(Population,Dec,Mask,Problem.N);

            %% Optimization
            MaxP = false(20,Problem.D);
            MinP = MaxP;
            while Algorithm.NotTerminated(Population)
                [MaxP,MinP,Nonzero] = POSMining(logical(Population(FrontNo==1).decs),MaxP,MinP,20);
                MatingPool          = TournamentSelection(2,Problem.N,FrontNo);
                [OffDec,OffMask]    = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),MaxP(:,Nonzero),MinP(:,Nonzero),Nonzero,Population.decs);
                if ~isempty(OffDec)
                    [Population,Dec,Mask,FrontNo] = EnvironmentalSelection([Population,Problem.Evaluation(OffDec.*OffMask)],[Dec;OffDec],[Mask;OffMask],Problem.N);
                end
            end
        end
    end
end
```

### `POSMining.m`
```matlab
function [newMaxP,newMinP,Nonzero] = POSMining(Mask,MaxP,MinP,N)
% Mine the maximum and minimum optimal sparse subspaces

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Nonzero = any(Mask,1);
    maxp    = Mining(Mask(:,Nonzero),MaxP(:,Nonzero),N,true);
    newMaxP = false(size(maxp,1),size(Mask,2));
    newMaxP(:,Nonzero) = maxp;
    newMaxP(1:min(end,size(MaxP,1)),~Nonzero) = MaxP(1:min(end,size(newMaxP,1)),~Nonzero);
    minp    = Mining(Mask(:,Nonzero),MinP(:,Nonzero),N,false);
    newMinP = false(size(minp,1),size(Mask,2));
    newMinP(:,Nonzero) = minp;
    newMinP(1:min(end,size(MinP,1)),~Nonzero) = MinP(1:min(end,size(newMinP,1)),~Nonzero);
    Nonzero = find(Nonzero);
end

function [PopDec,PopObj] = Mining(Mask,PopDec,N,Maximize)
% Evolutionary multi-objective based pattern mining

    %% Population initialization
    Dec = false(N,size(PopDec,2));
    if Maximize
        for i = 1 : N
            Dec(i,:) = any(Mask(randperm(end,ceil(rand.^2*end)),:),1);
        end
    else
        for i = 1 : N
            Dec(i,:) = all(Mask(randperm(end,ceil(rand.^2*end)),:),1);
        end
    end
    PopDec = [PopDec;Dec;xor(Maximize,logical(eye(size(PopDec,2))));Mask];
    Lens   = sum(Mask,2);
    PopObj = CalObj(PopDec,Mask,Lens,Maximize);
	[PopDec,PopObj,FrontNo] = EnvironmentalSelection(PopDec,PopObj,N);
    
    %% Optimization
    for gen = 1 : 10
        MatingPool = TournamentSelection(2,2*N,FrontNo);
        OffDec     = BinaryCrossover(PopDec(MatingPool(1:2:end),:),PopDec(MatingPool(2:2:end),:));
        OffDec     = BinaryMutation(OffDec);
        OffObj     = CalObj(OffDec,Mask,Lens,Maximize);
        [PopDec,PopObj,FrontNo] = EnvironmentalSelection([PopDec;OffDec],[PopObj;OffObj],N);
    end
end

function Objs = CalObj(Decs,T,Lens,Maxmize)
% Calculate the objective values

    [N,D] = size(T);
    Objs  = zeros(size(Decs,1),2);
    Len   = sum(Decs,2);
    if Maxmize
        %% For mining maximum optimal sparse subspaces
        for i = 1 : size(Decs,1)
            Tx = false(1,N);
            for j = 1 : N
                m = 1;
                while m <= D && Decs(i,m) >= T(j,m)
                    m = m + 1;
                end
                Tx(j) = m > D;
            end
            if ~any(Tx)
                Objs(i,:) = 1;
            else
                Objs(i,1) = 1 - mean(Tx);
                Objs(i,2) = 1 - mean(Lens(Tx))./Len(i);
            end
        end
    else
        %% For mining minimum optimal sparse subspaces
        for i = 1 : size(Decs,1)
            Tx = all(T(:,Decs(i,:)),2);
            if ~any(Tx)
                Objs(i,:) = 1;
            else
                Objs(i,1) = 1 - mean(Tx);
                Objs(i,2) = 1 - mean(Len(i)./Lens(Tx));
            end
        end
    end
end
```
