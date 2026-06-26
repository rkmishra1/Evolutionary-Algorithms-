# MTS

**Tags**: <2009> <multi> <real/integer>

## Description
Multiple trajectory search

## Reference
L. Y. Tseng and C. Chen. Multiple trajectory search for unconstrained / constrained multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2009, 1951-1958.

## Source Code

### `AdjustAppSet.m`
```matlab
function AppSet = AdjustAppSet(AppSet,N)
% Reduce the size of approximation set

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    AppObj = AppSet.objs;
    
    %% Selete the solutions having the smallest values on at least one objective
    Choose = false(1,length(AppSet));
    for i = 1 : size(AppObj,2)
        Choose(AppObj(:,i)==min(AppObj(:,i))) = true;
    end
    
    %% Select other solutions one by one
    if sum(Choose) > N
        selected = find(Choose);
        Choose   = selected(randperm(length(selected),N));
    else
        Distance = pdist2(AppObj,AppObj);
        Distance(logical(eye(length(Distance)))) = inf;
        while sum(Choose) < N && ~all(Choose)
            unSelected = find(~Choose);
            [~,x]      = max(min(Distance(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
    end
    AppSet = AppSet(Choose);
end
```

### `Grading.m`
```matlab
function [grade,improve,AppSet] = Grading(X,oldX,grade,improve,AppSet,popsize)
% Check if the new solution can be added to the approximation set

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Decide whether to put the new solution into approximation set
    if isempty(AppSet)
        AppSet = X;
        flag   = false;
    else
        compare = AppSet.objs - repmat(X.obj,length(AppSet),1);
        if any(all(compare<=0,2))
            flag = false;
        else
            dominated = all(compare>=0,2);
            AppSet(dominated) = [];
            AppSet = [AppSet,X];
            flag   = true;
        end
    end
    % If the archive contains too many solutions, delete some. The original
    % algorithm does not limit the size of archive, so that the size of
    % archive will increase unrestrainedly
    if length(AppSet) > 10*popsize
        AppSet = AdjustAppSet(AppSet,5*popsize);
    end

    %% Update the information about the solution
    if flag
        grade = grade + 9;
    end
    if sum(X.obj<oldX.obj) > sum(X.obj>oldX.obj)
        grade   = grade + 2;
        improve = true;
    end
end
```

### `LocalSearch1.m`
```matlab
function [grade,X,SR,improve,AppSet] = LocalSearch1(Problem,X,SR,improve,AppSet)
% Local Search 1 of MTS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if ~improve
        SR = SR / 2;
        if all(SR<1e-8)
            SR = (Problem.upper-Problem.lower).*(rand(1,length(SR))/10+0.4);
        end
    end
    improve = false;
    grade   = 0;
    for i = randperm(length(SR))
        old_X  = X;
        dec    = X.dec;
        dec(i) = dec(i) + SR(i)*(rand*2-1);
        X      = Problem.Evaluation(dec);
        [grade,improve,AppSet] = Grading(X,old_X,grade,improve,AppSet,Problem.N);
        if all(old_X.obj<=X.obj)
            dec    = old_X.dec;
            dec(i) = dec(i) - 0.5*SR(i)*(rand*2-1);
            X      = Problem.Evaluation(dec);
            [grade,improve,AppSet] = Grading(X,old_X,grade,improve,AppSet,Problem.N);
            if all(old_X.obj<=X.obj)
                X = old_X;
            end
        end
    end
end
```

### `LocalSearch2.m`
```matlab
function [grade,X,SR,improve,AppSet] = LocalSearch2(Problem,X,SR,improve,AppSet)
% Local Search 2 of MTS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if ~improve
        SR = SR / 2;
        if all(SR<1e-8)
            SR = (Problem.upper-Problem.lower).*(rand(1,length(SR))/10+0.4);
        end
    end
    improve = false;
    grade   = 0;
    for l = 1 : length(SR)
        chosen = rand(1,length(SR)) < 1/4;
        old_X  = X;
        dec    = X.dec;
        dec(chosen) = dec(chosen) + SR(chosen).*(rand(1,sum(chosen))*2-1);
        X      = Problem.Evaluation(dec);
        [grade,improve,AppSet] = Grading(X,old_X,grade,improve,AppSet,Problem.N);
        if all(old_X.obj<=X.obj)
            dec = old_X.dec;
            dec(chosen) = dec(chosen) - 0.5*SR(chosen).*(rand(1,sum(chosen))*2-1);
            X   = Problem.Evaluation(dec);
            [grade,improve,AppSet] = Grading(X,old_X,grade,improve,AppSet,Problem.N);
            if all(old_X.obj<=X.obj)
                X = old_X;
            end
        end
    end
end
```

### `LocalSearch3.m`
```matlab
function [grade,X,SR,improve,AppSet] = LocalSearch3(Problem,X,SR,improve,AppSet)
% Local Search 3 of MTS

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    grade   = 0;
    SearchL = Problem.lower;
    SearchU = Problem.upper;
    Disp    = (SearchU-SearchL)/10;
    Best    = X;
    while any(Disp<1e-2)
        for i = randperm(length(SR))
            value = SearchL(i) : Disp(i) : SearchU(i);
            Decs  = repmat(Best.dec,length(value),1);
            Decs(:,i) = value;
            Y     = Problem.Evaluation(Decs);
            for j = 1 : length(Y)
                % 'grade', 'SR' and 'improve' will not be updated in local
                % search 3
                [~,~,AppSet] = Grading(Y(j),Best,grade,improve,AppSet,Problem.N);
                if all(Y(j).obj<=Best.obj)
                    Best = Y(j);
                end
            end
            SearchL(i) = max(Best.dec(i)-2*Disp(i),Problem.lower);
            SearchU(i) = min(Best.dec(i)+2*Disp(i),Problem.upper);
            Disp(i)    = (SearchU(i)-SearchL(i))/10;
        end
    end
    X = Best;
end
```

### `MTS.m`
```matlab
classdef MTS < ALGORITHM
% <2009> <multi> <real/integer>
% Multiple trajectory search
% popsize           --- 40 --- Size of the population
% ofLocalSearchTest ---  5 --- Number of iterations for determining the best local search
% ofLocalSearch     --- 45 --- Number of iterations for applying the best local search
% ofForeground      ---  5 --- Number of best solutions for local search

%------------------------------- Reference --------------------------------
% L. Y. Tseng and C. Chen. Multiple trajectory search for unconstrained /
% constrained multi-objective optimization. Proceedings of the IEEE
% Congress on Evolutionary Computation, 2009, 1951-1958.
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
            [popsize,ofLocalSearchTest,ofLocalSearch,ofForeground] = Algorithm.ParameterSet(40,5,45,5);

            %% Generate random population
            % Generate the SOA
            [~,SOA] = sort(rand(popsize,Problem.D));
            % Map the SOA to the decision space
            Decs = SOA./popsize.*repmat(Problem.upper-Problem.lower,popsize,1) + repmat(Problem.lower,popsize,1);
            % Generate the initial solutions
            Population  = Problem.Evaluation(Decs);
            AppSet      = Population;
            Enable      = true(1,popsize);
            Improve     = true(1,popsize);
            Grade       = zeros(1,popsize);
            SearchRange = repmat((Problem.upper-Problem.lower)/2,Problem.N,1);

            %% Optimization
            while Algorithm.NotTerminated(AppSet)
                for i = find(Enable)
                    Grade(i)     = 0;
                    LS_TestGrade = zeros(1,3);
                    for j = 1 : ofLocalSearchTest
                        [grade1,Population(i),SearchRange(i,:),Improve(i),AppSet] = LocalSearch1(Problem,Population(i),SearchRange(i,:),Improve(i),AppSet);
                        [grade2,Population(i),SearchRange(i,:),Improve(i),AppSet] = LocalSearch2(Problem,Population(i),SearchRange(i,:),Improve(i),AppSet);
                        [grade3,Population(i),SearchRange(i,:),Improve(i),AppSet] = LocalSearch3(Problem,Population(i),SearchRange(i,:),Improve(i),AppSet);
                        LS_TestGrade(1) = LS_TestGrade(1) + grade1;
                        LS_TestGrade(2) = LS_TestGrade(2) + grade2;
                        LS_TestGrade(3) = LS_TestGrade(3) + grade3;
                    end
                    [~,best]    = max(LS_TestGrade);
                    LocalSearch = {@LocalSearch1,@LocalSearch2,@LocalSearch3};
                    for j = 1 : ofLocalSearch
                        [grade,Population(i),SearchRange(i,:),Improve(i),AppSet] = LocalSearch{best}(Problem,Population(i),SearchRange(i,:),Improve(i),AppSet);
                        Grade(i) = Grade(i) + grade;
                    end
                end
                [~,rank]  = sort(Grade,'descend');
                Enable(:) = false;
                Enable(rank(1:ofForeground)) = true;
                if Problem.FE >= Problem.maxFE
                    AppSet = AdjustAppSet(AppSet,Problem.N);
                end
            end
        end
    end
end
```
