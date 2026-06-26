# I-SIBEA

**Tags**: <2015> <multi> <real/integer/label/binary/permutation>

## Description
Interactive simple indicator-based evolutionary algorithm

## Reference
T. Chugh, K. Sindhya, J. Hakanen, and K. Miettinen. An interactive simple indicator-based evolutionary algorithm (I-SIBEA) for multiobjective optimization problems. Proceedings of the International Conference on Evolutionary Multi-Criterion Optimization, 2015, 277-291.

## Source Code

### `CalWHV.m`
```matlab
function Score = CalWHV(PopObj,RefPoint,Weight)
% Calculate the exact weighted hypervolume value of the population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    PopObj(any(PopObj>repmat(RefPoint,N,1),2),:) = [];
    if isempty(PopObj)
        Score = 0;
    else
        pl = sortrows(PopObj);
        S  = {1,pl};
        for k = 1 : M-1
            S_ = {};
            for i = 1 : size(S,1)
                Stemp = Slice(cell2mat(S(i,2)),k,RefPoint);
                for j = 1 : size(Stemp,1)
                    temp(1) = {cell2mat(Stemp(j,1))*cell2mat(S(i,1))};
                    temp(2) = Stemp(j,2);
                    S_      = Add(temp,S_);
                end
            end
            S = S_;
        end
        Score = 0;
        for i = 1 : size(S,1)
            p     = Head(cell2mat(S(i,2)));
            Score = Score + cell2mat(S(i,1))*abs(p(M)-RefPoint(M))*Weight(i);
        end
    end
end

function S = Slice(pl,k,RefPoint)
    p  = Head(pl);
    pl = Tail(pl);
    ql = [];
    S  = {};
    while ~isempty(pl)
        ql  = Insert(p,k+1,ql);
        p_  = Head(pl);
        cell_(1,1) = {abs(p(k)-p_(k))};
        cell_(1,2) = {ql};
        S   = Add(cell_,S);
        p   = p_;
        pl  = Tail(pl);
    end
    ql = Insert(p,k+1,ql);
    cell_(1,1) = {abs(p(k)-RefPoint(k))};
    cell_(1,2) = {ql};
    S  = Add(cell_,S);
end

function ql = Insert(p,k,pl)
    flag1 = 0;
    flag2 = 0;
    ql    = [];
    hp    = Head(pl);
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql;hp];
        pl = Tail(pl);
        hp = Head(pl);
    end
    ql = [ql;p];
    m  = length(p);
    while ~isempty(pl)
        q = Head(pl);
        for i = k : m
            if p(i) < q(i)
                flag1 = 1;
            else
                if p(i) > q(i)
                    flag2 = 1;
                end
            end
        end
        if ~(flag1 == 1 && flag2 == 0)
            ql = [ql;Head(pl)];
        end
        pl = Tail(pl);
    end  
end

function p = Head(pl)
    if isempty(pl)
        p = [];
    else
        p = pl(1,:);
    end
end

function ql = Tail(pl)
    if size(pl,1) < 2
        ql = [];
    else
        ql = pl(2:end,:);
    end
end

function S_ = Add(cell_,S)
    n = size(S,1);
    m = 0;
    for k = 1 : n
        if isequal(cell_(1,2),S(k,2))
            S(k,1) = {cell2mat(S(k,1))+cell2mat(cell_(1,1))};
            m = 1;
            break;
        end
    end
    if m == 0
        S(n+1,:) = cell_(1,:);
    end
    S_ = S;     
end
```

### `CalWHVLoss.m`
```matlab
function WHVLoss = CalWHVLoss(PopObj,FrontNo,wz,AA,RA)
% Calculate the weighted hypervolume (WHV) loss of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the weight of each solution for WHV calculation
    Weight = ones(1,size(PopObj,1));
    if nargin > 2 && ~isempty(wz)
        for i = 1 : size(PopObj,1)
            if any(all(repmat(PopObj(i,:),size(AA,1),1)<=AA,2))
                % The solution belongs to the preferred region (Pr)
                Weight(i) = wz(3);
            elseif any(all(repmat(PopObj(i,:),size(RA,1),1)>RA,2))
                % The solution belongs to the dominated region (Do)
                Weight(i) = wz(1);
            else
                % The solution belongs to the no preference information
                % region (In)
                Weight(i) = wz(2);
            end
        end
    end
    
    %% Calculate the WHV loss of each solution front by front
    WHVLoss  = zeros(1,size(PopObj,1));
    RefPoint = max(PopObj,[],1) + 0.1;
    for f = setdiff(unique(FrontNo),inf)
        current  = find(FrontNo==f);
        totalWHV = CalWHV(PopObj(current,:),RefPoint,Weight(current));
        for i = 1 : length(current)
            drawnow('limitrate');
            currenti           = current([1:i-1,i+1:end]);
            WHVLoss(current(i))= totalWHV - CalWHV(PopObj(currenti,:),RefPoint,Weight(currenti));
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,WHVLoss] = EnvironmentalSelection(Population,N,wz,AA,RA)
% The environmental selection of I-SIBEA

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
    Next = FrontNo < MaxFNo;
    
    %% Calculate the WHV loss of each solution front by front
    WHVLoss = CalWHVLoss(Population.objs,FrontNo,wz,AA,RA);
    
    %% Select the solutions in the last front based on their WHV loss
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(WHVLoss(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    WHVLoss    = WHVLoss(Next);
end
```

### `ISIBEA.m`
```matlab
classdef ISIBEA < ALGORITHM
% <2015> <multi> <real/integer/label/binary/permutation>
% Interactive simple indicator-based evolutionary algorithm
% Point --- --- Preferred point

%------------------------------- Reference --------------------------------
% T. Chugh, K. Sindhya, J. Hakanen, and K. Miettinen. An interactive simple
% indicator-based evolutionary algorithm (I-SIBEA) for multiobjective
% optimization problems. Proceedings of the International Conference on
% Evolutionary Multi-Criterion Optimization, 2015, 277-291.
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
            Point = Algorithm.ParameterSet(ones(1,Problem.M));

            %% Generate random population
            Population = Problem.Initialization();
            FrontNo    = NDSort(Evaluate(Population.objs,Point),inf);
            WHVLoss    = CalWHVLoss(Population.objs,FrontNo);
            wz = [];    % Weight distribution function value
            AA = [];    % Preferred set
            RA = [];    % Non-preferred set

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-WHVLoss);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,WHVLoss] = EnvironmentalSelection([Population,Offspring],Problem.N,wz,AA,RA);
                if ~mod(ceil(Problem.FE/Problem.N),ceil(ceil(Problem.maxFE/Problem.N)/4))
                    [wz,AA,RA] = Interaction(Population.objs,Point);
                end
            end
        end
     end
end

function PopObj = Evaluate(PopObj,Point)
% g-dominance based function evaluation

    Point = repmat(Point,size(PopObj,1),1);
    Flag  = all(PopObj<=Point,2) | all(PopObj>=Point,2);
    Flag  = repmat(Flag,1,size(PopObj,2));
    PopObj(~Flag) = PopObj(~Flag) + 1e10;
end
```

### `Interaction.m`
```matlab
function [wz,AA,RA] = Interaction(PopObj,Point)
% Identify the preferred point and all the others are treated as
% non-preferred points

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Identify the unique and non-dominated solutions
    PopObj = unique(PopObj,'rows');
    PopObj = PopObj(NDSort(PopObj,1)==1,:);
    
    %% Identify the preferred solution
    ideal = min(PopObj,[],1);
    % Calculate the Tchebycheff function value of each solution on Point
    Fitness = max((PopObj-repmat(ideal,size(PopObj,1),1))./repmat(Point,size(PopObj,1),1),[],2);
    % The one having the minimal function value is treated as the preferred
    % point
    [~,prefer] = min(Fitness);
    AA = PopObj(prefer,:);
    RA = PopObj([1:prefer-1,prefer+1:end],:);
    
    %% Calculate the weight distribution function value
    RefPoint = max(PopObj,[],1) + 0.1;
    wz = [0,1,1+CalWHV(AA,RefPoint,ones(1,size(AA,1)))/CalWHV(RA,RefPoint,ones(1,size(RA,1)))];
end
```
