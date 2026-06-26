# hpaEA

**Tags**: <2020> <multi/many> <real/integer/label/binary/permutation>

## Description
Hyperplane assisted evolutionary algorithm

## Reference
H. Chen, Y. Tian, W. Pedrycz, G. Wu, R. Wang, and L. Wang. Hyperplane assisted evolutionary algorithm for many-objective optimization problems. IEEE Transactions on Cybernetics, 2020, 50(7): 3367-3380.

## Source Code

### `hpaEA.m`
```matlab
classdef hpaEA < ALGORITHM
% <2020> <multi/many> <real/integer/label/binary/permutation>
% Hyperplane assisted evolutionary algorithm

%------------------------------- Reference --------------------------------
% H. Chen, Y. Tian, W. Pedrycz, G. Wu, R. Wang, and L. Wang. Hyperplane
% assisted evolutionary algorithm for many-objective optimization problems.
% IEEE Transactions on Cybernetics, 2020, 50(7): 3367-3380.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

    methods
        function main(Algorithm,Problem)
            % Randomly initiallize a population
            Population = Problem.Initialization();
            % Generate a set of reference vectors
            [V, ~] = UniformPoint(Problem.N, Problem.M);
            % Record the nadir point
            maxObj = Inf(1, Problem.M);
            % Initialize an array to record the indexes of prominent solutions
            PSI = [];

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Mating selection
                domNum = size(PSI, 2);
                MatingPool = randi(numel(Population), 1, Problem.N-domNum);
                MatingPool = [MatingPool, PSI];
                MatingPool(randperm(size(MatingPool, 2))) = MatingPool(1: end);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                CombinePop = [Population, Offspring];
                % Remove some solutions that cannot dominate the worst point
                fitness = max(CombinePop.objs - repmat(maxObj, numel(CombinePop), 1), [], 2);
                deleteI = fitness > 0;
                CombinePop(deleteI) = [];
                if numel(CombinePop) < 2
                    CombinePop = [Population, Offspring];
                end
                % Update the worst point
                tempMax  = max(CombinePop.objs);
                replaceI = maxObj > tempMax;
                maxObj(replaceI) = tempMax(replaceI);        
                % Environmental Selection
                [Population, PSI] = hpaEnvironmentalSelection(CombinePop, Problem, V);
            end
        end
    end
end
```

### `hpaEnvironmentalSelection.m`
```matlab
function [Population, PSI] = hpaEnvironmentalSelection(Population, Problem, V)
% The environmental selection of hpaEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen
    
    % Non-dominated sorting
    N = Problem.N;
    [FrontNo, MaxFNo] = NDSort(Population.objs, N);
    
    if numel(Population(FrontNo<=MaxFNo)) <= N
        SelI       = FrontNo<=MaxFNo;
        Population = Population(SelI);
        
        Chooose = FrontNo==1;
        CC      = Chooose + SelI;
        I       = find(CC == 2);
        PSI     = zeros(1, length(I));
        for j = 1 : length(I)
            index  = sum(SelI(1: I(j)));    % the index of the solutions in the selected population
            PSI(j) = index;
        end
    else % If the number of selected solutions is larger than the population size       
        if MaxFNo <= 1
            [Population, PSI] = hpaNDSolutionsSelectionStrategy(Population(FrontNo==1), V, Problem);           
        else % retain some dominated solutions to improve diversity
            % Record the selected solutions
            SelectedSolutions = Population(FrontNo < MaxFNo);
            SelNum = numel(SelectedSolutions);
            % The solutions in the last front
            CanSolutions = Population(FrontNo==MaxFNo);
            CanNum       = numel(CanSolutions);
            PopObj       = [SelectedSolutions.objs; CanSolutions.objs];
            % Normalization
            Zmin       = min(PopObj, [], 1);
            Zmax       = max(PopObj, [], 1);
            PopObj     = (PopObj-repmat(Zmin,size(PopObj,1),1))./repmat(Zmax-Zmin,size(PopObj,1),1);
            Population = [SelectedSolutions, CanSolutions];
            PSI        = 1 : SelNum;    % The index of the porminent solutions
            Next       = [true(1, SelNum), false(1, CanNum)];
            % Select remaining solutions based on angle
            angle = acos(1-pdist2(PopObj, PopObj, 'cosine'));
            while sum(Next) < N
                Select   = find(Next);
                Remain   = find(~Next);
                [~, rho] = max(min(angle(Remain, Select), [], 2));
                Next(Remain(rho)) = true;
            end
            Population = Population(Next);         
        end
    end
end
```

### `hpaNDSolutionsSelectionStrategy.m`
```matlab
function [Population, PSI] = hpaNDSolutionsSelectionStrategy(Population, V, Problem)
% The environmental selection of hpaEA
% Strategy for Enhancing Pressure: 
% 1) Select the solutions that locate between the Hyperplane and the ideal point
% 2) The Hyperplane is determined by the solutinons' nearest M neighborhoods
% 3) Neighborhood angle is used to select the remaining solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huangke Chen

    N    = Problem.N;
    Objs = Population.objs;
    Next = false(1, size(Objs, 1));

    %% Normalization
    Zmin = min(Objs, [], 1);
    Zmax = max(Objs, [], 1);
    Objs = (Objs - repmat(Zmin, size(Objs, 1), 1))./repmat(Zmax-Zmin, size(Objs, 1), 1);
        
    % search the prominent solutions
    [~, IM] = sort(Objs);
    
    for i = 1 : size(Objs, 1)
        tempObj = Objs(i, :); % the check objective vector
        % find its neighbourhood
        if all(mod(find(IM == i) - 1, size(Objs, 1)) ~= 0) % for the non-extreme solutions
            neighI = IM(find(IM == i) - 1); % the index of the M (objective count) neighbourhood
            if length(unique(neighI)) < length(neighI) || any(neighI < 3)
                continue; % skip the dominated solutions 
            end
            
            A  = ones(size(Objs, 2));
            A(:, 1: end-1) = Objs(neighI, 1: end-1);
            B  = Objs(neighI, end);
            CC = det(A);
            if CC < 1e-10
                continue;
            end
            coV  = (A\B)';  % the coefficient vector for the linear equation
            boun = sum(coV(1: end-1).*tempObj(1: end-1)) + coV(end);
            if tempObj(end) <= boun
                Next(i) = true;
            end            
        end        
    end
    OSI = Next; % record the prominent solutions

    % Associate solutions to different subspaces 
    Angle         = acos(1-pdist2(Objs, V, 'cosine'));
    [~,associate] = min(Angle, [], 2);
    for i = unique(associate)'
        current2 = find(associate == i);        
        if ~isempty(current2) % there exists no solutions in this subspace                        
            tempAn         = Angle(current2, i);
            [~, index]     = min(tempAn); 
            popIndex       = current2(index);
            Next(popIndex) = true;
        end        
    end
    
    % Select the extreme solutions
    if Problem.FE > 0.8*Problem.maxFE
        M        = size(Objs, 2);
        nEachObj = floor((2/3)*(N/M));
        for i = 1 : M
            tempObjS = Objs(:, i);
            maxObj   = max(tempObjS);
            minObj   = min(tempObjS);
            interval = (maxObj - minObj)/nEachObj;
            inIndex  = ceil((tempObjS - minObj + 10^-5)./interval);
            
            for f = unique(inIndex)'
                current1 = find(inIndex == f);
                if isempty(intersect(current1, find(Next==true))) || f == 1
                    Next(current1(1)) = true;
                end
            end
        end
    end
        
    % Select remaining solutions based on angle
    angle = acos(1-pdist2(Objs, Objs,'cosine'));
    while sum(Next) < N        
        Select   = find(Next);
        Remain   = find(~Next);
        [~, rho] = max(min(angle(Remain, Select), [], 2));
        Next(Remain(rho)) = true;
    end
    
    CC  = Next + OSI;
    I   = find(CC == 2);
    PSI = zeros(1, length(I));
    for j = 1 : length(I)
        index  = sum(Next(1: I(j)));    % the index of the solutions in the selected population
        PSI(j) = index;
    end
    
    Population = Population(Next);    
end
```
