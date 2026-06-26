# MaOEA-IGD

**Tags**: <2019> <many> <real/integer/label/binary/permutation>

## Description
IGD based many-objective evolutionary algorithm

## Reference
Y. Sun, G. G. Yen, and Z. Yi. IGD indicator-based evolutionary algorithm for many-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2019, 23(2): 173-187.

## Source Code

### `Assignmentoptimal.m`
```matlab
function [assignment, cost] = Assignmentoptimal(distMatrix)
%ASSIGNMENTOPTIMAL Compute optimal assignment by Munkres algorithm
% ASSIGNMENTOPTIMAL(DISTMATRIX) computes the optimal assignment (minimum
% overall costs) for the given rectangular distance or cost matrix, for
% example the assignment of tracks (in rows) to observations (in
% columns). The result is a column vector containing the assigned column
% number in each row (or 0 if no assignment could be done).
%
% [ASSIGNMENT, COST] = ASSIGNMENTOPTIMAL(DISTMATRIX) returns the
% assignment vector and the overall cost.
%
% The distance matrix may contain infinite values (forbidden
% assignments). Internally, the infinite values are set to a very large
% finite number, so that the Munkres algorithm itself works on
% finite-number matrices. Before returning the assignment, all
% assignments with infinite distance are deleted (i.e. set to zero).
%
%
%
% Markus Buehren
% Last modified 05.07.2011

% save original distMatrix for cost computation
originalDistMatrix = distMatrix;

% check for negative elements
if any(distMatrix(:) < 0)
error('All matrix elements have to be non-negative.');
end

% get matrix dimensions
[nOfRows, nOfColumns] = size(distMatrix);

% check for infinite values
finiteIndex = isfinite(distMatrix);
infiniteIndex = find(~finiteIndex);
if ~isempty(infiniteIndex)
% set infinite values to large finite value
maxFiniteValue = max(max(distMatrix(finiteIndex)));
if maxFiniteValue > 0
infValue = abs(10 * maxFiniteValue * nOfRows * nOfColumns);
else
infValue = 10;
end
if isempty(infValue)
% all elements are infinite
assignment = zeros(nOfRows, 1);
cost = 0;
return
end
distMatrix(infiniteIndex) = infValue;
end

% memory allocation
coveredColumns = zeros(1, nOfColumns);
coveredRows = zeros(nOfRows, 1);
starMatrix = zeros(nOfRows, nOfColumns);
primeMatrix = zeros(nOfRows, nOfColumns);

% preliminary steps
if nOfRows <= nOfColumns
minDim = nOfRows;

% find the smallest element of each row
minVector = min(distMatrix, [], 2);

% subtract the smallest element of each row from the row
distMatrix = distMatrix - repmat(minVector, 1, nOfColumns);

% Steps 1 and 2
for row = 1:nOfRows
for col = find(distMatrix(row,:)==0)
if ~coveredColumns(col)%~any(starMatrix(:,col))
starMatrix(row, col) = 1;
coveredColumns(col) = 1;
break
end
end
end

else % nOfRows > nOfColumns
minDim = nOfColumns;

% find the smallest element of each column
minVector = min(distMatrix);

% subtract the smallest element of each column from the column
distMatrix = distMatrix - repmat(minVector, nOfRows, 1);

% Steps 1 and 2
for col = 1:nOfColumns
for row = find(distMatrix(:,col)==0)'
if ~coveredRows(row)
starMatrix(row, col) = 1;
coveredColumns(col) = 1;
coveredRows(row) = 1;
break
end
end
end
coveredRows(:) = 0; % was used auxiliary above
end

if sum(coveredColumns) == minDim
% algorithm finished
assignment = buildassignmentvector__(starMatrix);
else
% move to step 3
[assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step3__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim); %#ok
end

% compute cost and remove invalid assignments
[assignment, cost] = computeassignmentcost__(assignment, originalDistMatrix, nOfRows);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function assignment = buildassignmentvector__(starMatrix)

[maxValue, assignment] = max(starMatrix, [], 2);
assignment(maxValue == 0) = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [assignment, cost] = computeassignmentcost__(assignment, distMatrix, nOfRows)

rowIndex = find(assignment);
costVector = distMatrix(rowIndex + nOfRows * (assignment(rowIndex)-1));
finiteIndex = isfinite(costVector);
cost = sum(costVector(finiteIndex));
assignment(rowIndex(~finiteIndex)) = 0;
end
% Step 2: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step2__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim)

% cover every column containing a starred zero
maxValue = max(starMatrix);
coveredColumns(maxValue == 1) = 1;

if sum(coveredColumns) == minDim
% algorithm finished
assignment = buildassignmentvector__(starMatrix);
else
% move to step 3
[assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step3__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim);
end
end
% Step 3: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step3__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim)

zerosFound = 1;
while zerosFound

zerosFound = 0;
for col = find(~coveredColumns)
for row = find(~coveredRows')
if distMatrix(row,col) == 0

primeMatrix(row, col) = 1;
starCol = find(starMatrix(row,:));
if isempty(starCol)
% move to step 4
[assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step4__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, row, col, minDim);
return
else
coveredRows(row) = 1;
coveredColumns(starCol) = 0;
zerosFound = 1;
break % go on in next column
end
end
end
end
end

% move to step 5
[assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step5__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim);
end
% Step 4: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step4__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, row, col, minDim)

newStarMatrix = starMatrix;
newStarMatrix(row,col) = 1;

starCol = col;
starRow = find(starMatrix(:, starCol));

while ~isempty(starRow)

% unstar the starred zero
newStarMatrix(starRow, starCol) = 0;

% find primed zero in row
primeRow = starRow;
primeCol = find(primeMatrix(primeRow, :));

% star the primed zero
newStarMatrix(primeRow, primeCol) = 1;

% find starred zero in column
starCol = primeCol;
starRow = find(starMatrix(:, starCol));

end
starMatrix = newStarMatrix;

primeMatrix(:) = 0;
coveredRows(:) = 0;

% move to step 2
[assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step2__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim);
end
% Step 5: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step5__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim)

% find smallest uncovered element
uncoveredRowsIndex = find(~coveredRows');
uncoveredColumnsIndex = find(~coveredColumns);
[s, index1] = min(distMatrix(uncoveredRowsIndex,uncoveredColumnsIndex));
[s, index2] = min(s); %#ok
h = distMatrix(uncoveredRowsIndex(index1(index2)), uncoveredColumnsIndex(index2));

% add h to each covered row
index = find(coveredRows);
distMatrix(index, :) = distMatrix(index, :) + h;

% subtract h from each uncovered column
distMatrix(:, uncoveredColumnsIndex) = distMatrix(:, uncoveredColumnsIndex) - h;

% move to step 3
[assignment, distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows] = step3__(distMatrix, starMatrix, primeMatrix, coveredColumns, coveredRows, minDim);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Rank,Dis] = EnvironmentalSelection(Population,W,N)
% The environmental selection of MaOEA/IGD

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Assign proximity distance
    [~,x]      = unique(round(Population.objs*1e4)/1e4,'rows');
    Population = Population(x);
    N          = min(N,length(Population));
    Rank = zeros(1,length(Population));
    Dis  = zeros(length(Population),size(W,1));
    for i = 1 : length(Population)
        temp = repmat(Population(i).obj,size(W,1),1);
        domi = any(temp<W,2) - any(temp>W,2);
        if any(domi==1)
            Rank(i)  = 1;
            Dis(i,:) = -sqrt(sum((temp-W).^2,2))';
        elseif any(domi==-1)
            Rank(i)  = 3;
            Dis(i,:) = sqrt(sum((temp-W).^2,2))';
        else
            Rank(i)  = 2;
            Dis(i,:) = sqrt(sum(max(temp-W,0).^2,2))';
        end
    end

    %% Select the solutions in the first fronts
    MaxFNo = find(cumsum(hist(Rank,1:3))>=N,1);
    Next   = Rank < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(Rank==MaxFNo);
    Choose = LastSelection(Dis(Last,:),N-sum(Next),W);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
    Rank       = Rank(Next);
    Dis        = Dis(Next,:);
end

function Choose = LastSelection(Dis,K,W)
% Select part of the solutions in one front

    %% Select K points from W
    Distance = pdist2(W,W);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(W,1));
    while sum(~Del) > K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
    Dis = Dis(:,~Del);
    
    if false
        %% Hungarian method based selection
        Choose = Assignmentoptimal(Dis'-min(Dis(:)));
    else
        %% Greedy algorithm based selection (more efficient)
        Choose = false(1,size(Dis,1));
        for i = 1 : size(Dis,2)
            remain   = find(~Choose);
            [~,best] = min(Dis(remain,i));
            Choose(remain(best)) = true;
        end
    end
end
```

### `MaOEAIGD.m`
```matlab
classdef MaOEAIGD < ALGORITHM
% <2019> <many> <real/integer/label/binary/permutation>
% IGD based many-objective evolutionary algorithm
% DNPE --- --- Number of evaluations for nadir point estimation

%------------------------------- Reference --------------------------------
% Y. Sun, G. G. Yen, and Z. Yi. IGD indicator-based evolutionary algorithm
% for many-objective optimization problems. IEEE Transactions on
% Evolutionary Computation, 2019, 23(2): 173-187.
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
            DNPE = Algorithm.ParameterSet(100*Problem.N);

            %% Nadir point estimation
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            % Optimize Problem.M single-objective optimization problems
            Population = Problem.Initialization();
            while Algorithm.NotTerminated(Population) && Problem.FE < DNPE
                Offspring  = OperatorGA(Problem,Population(randi(end,1,Problem.N)),{0.9,20,1,20});
                Population = [Population,Offspring];
                [~,rank]   = sort(Fitness(Population.objs),1);
                Population = Population(unique(rank(1:ceil(Problem.N/Problem.M),:)));
            end
            % Find the nadir point and ideal point
            [~,ext] = min(Fitness(Population.objs),[],1);
            zmax    = diag(Population(ext).objs)';
            zmin    = min(Population.objs,[],1);
            zmax(zmax<1e-6) = 1;
            % Transform the points into the Utopian Pareto front
            W = W.*repmat(zmax-zmin,Problem.N,1) + repmat(zmin,Problem.N,1);

            %% Generate random population
            Population = Problem.Initialization();
            [Population,Rank,Dis] = EnvironmentalSelection(Population,W,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,Rank,min(Dis,[],2));
                Offspring  = OperatorGAhalf(Problem,Population(MatingPool));
                [Population,Rank,Dis] = EnvironmentalSelection([Population,Offspring],W,Problem.N);
            end
        end
    end
end

function fit = Fitness(PopObj)
% Calculate the objective value of each solution on each single-objective
% optimization problem in nadir point estimation

    fit = zeros(size(PopObj));
    for i = 1 : size(PopObj,2)
        fit(:,i) = abs(PopObj(:,i)) + 100*sum(PopObj(:,[1:i-1,i+1:end]).^2,2);
    end
end
```
