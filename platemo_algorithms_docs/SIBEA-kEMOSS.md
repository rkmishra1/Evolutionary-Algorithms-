# SIBEA-kEMOSS

**Tags**: <2007> <many> <real/integer/label/binary/permutation>

## Description
SIBEA with minimum objective subset of size k with minimum error

## Reference
D. Brockhoff and E. Zitzler. Improving hypervolume-based multiobjective evolutionary algorithms by using objective reduction methods. Proceedings of the IEEE Congress on Evolutionary Computation, 2007, 2086-2093.

## Source Code

### `CalHV.m`
```matlab
function Score = CalHV(PopObj,RefPoint)
% Calculate the exact hypervolume value of the population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Liangli Zhen

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
            Score = Score + cell2mat(S(i,1))*abs(p(M)-RefPoint(M));
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

### `CalHVLoss.m`
```matlab
function HVLoss = CalHVLoss(PopObj,FrontNo)
% Calculate the weighted hypervolume (WHV) loss of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Liangli Zhen
  
    %% Calculate the WHV loss of each solution front by front
    HVLoss   = zeros(1,size(PopObj,1));
    RefPoint = max(PopObj,[],1) + 0.1;
    for f = setdiff(unique(FrontNo),inf)
        current  = find(FrontNo==f);
        totalWHV = CalHV(PopObj(current,:),RefPoint);
        for i = 1 : length(current)
            drawnow('limitrate');
            currenti           = current([1:i-1,i+1:end]);
            HVLoss(current(i)) = totalWHV - CalHV(PopObj(currenti,:),RefPoint);
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,objective_set)
% The environmental selection of SIBEA-kEMOSS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Liangli Zhen

    PopObj = Population.objs;
    
    %% Non-dominated sorting    
    [FrontNo,MaxFNo] = NDSort(PopObj(:,objective_set),N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front based on their HV loss
    Last = find(FrontNo==MaxFNo);
    % Calculate the WHV loss of each solution in the last front
    HVLoss   = CalHVLoss(PopObj(Last,objective_set),FrontNo(Last));
    [~,Rank] = sort(HVLoss,'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
end
```

### `SIBEAkEMOSS.m`
```matlab
classdef SIBEAkEMOSS < ALGORITHM
% <2007> <many> <real/integer/label/binary/permutation>
% SIBEA with minimum objective subset of size k with minimum error
% G --- 5 --- Reduction frequency generations
% k --- 2 --- Size of reduced objective set

%------------------------------- Reference --------------------------------
% D. Brockhoff and E. Zitzler. Improving hypervolume-based multiobjective
% evolutionary algorithms by using objective reduction methods. Proceedings
% of the IEEE Congress on Evolutionary Computation, 2007, 2086-2093.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Liangli Zhen

    methods
        function main(Algorithm,Problem)
           %% Parameter setting
           [G,k] = Algorithm.ParameterSet(5,2);

            %% Generate random population
            Population    = Problem.Initialization();
            iteration_num = 0;
            objective_set = 1 : k;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if mod(iteration_num,G) ==0
                    objective_set = kEMOSS(Population,k);
                end
                iteration_num = iteration_num + 1;
                MatingPool    = randi(length(Population),1,Problem.N);
                Offspring     = OperatorGA(Problem,Population(MatingPool));
                Population    = EnvironmentalSelection([Population,Offspring],Problem.N,objective_set);
            end
        end
    end
end
```

### `getDelataMinfor.m`
```matlab
function delta = getDelataMinfor(Population,ObjectiveSet1,ObjectiveSet2)
% Calculate the objective subset with KMOSS greedy method

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Liangli Zhen

   delta   = 0;
   PopSize = size(Population.decs,1);
   PopObj  = Population.objs;
   
   %% Calculate the weakly dominated solution pairs
   DominateRelation = zeros(PopSize,PopSize);
   for i = 1 : PopSize
       DominateRelation(:, i) = ~any(repmat(PopObj(i,ObjectiveSet1),PopSize,1)>PopObj(:,ObjectiveSet1),2);
   end
   [row, col] = find(DominateRelation==1);
   
   %% Compute the minimum value that makes the dominance relation unchanged
   delta = max(delta,max(max(PopObj(col,ObjectiveSet2)-PopObj(row,ObjectiveSet2))));
end
```

### `kEMOSS.m`
```matlab
function objective_subset = kEMOSS(Population,k)
% Calculate the objective subset with greedy k-EMOSS method

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Liangli Zhen

   objective_set   = 1:size(Population.objs,2);
   selected_subset = [];
   while length(selected_subset) < k
       unselected_subset = setdiff(objective_set,selected_subset);
       errors = zeros(length(unselected_subset),1);
       for i = 1 : length(unselected_subset)
           errors(i,1) = getDelataMinfor(Population, [selected_subset, unselected_subset(i)], objective_set);
       end
       [~, v] = min(errors);
       selected_subset = [selected_subset unselected_subset(v(1))];
   end
   objective_subset = selected_subset;
end
```
