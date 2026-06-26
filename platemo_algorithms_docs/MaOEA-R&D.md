# MaOEA-R&D

**Tags**: <2016> <many> <real/integer/label/binary/permutation>

## Description
Many-objective evolutionary algorithm based on objective space reduction

## Reference
Z. He and G. G. Yen. Many-objective evolutionary algorithm: Objective space reduction and diversity improvement. IEEE Transactions on Evolutionary Computation, 2016, 20(1): 145-160.

## Source Code

### `Classification.m`
```matlab
function [Subpopulation,Z] = Classification(Population,W)
% Classify the population into each subpopulation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate ASF between each solution and classification vector
    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    Z      = min(PopObj,[],1);
    ASF    = zeros(N,M);
    for i = 1 : M
        ASF(:,i) = max((PopObj-repmat(Z,N,1))./repmat(W(i,:),N,1),[],2);
    end
    
    %% Classification
    class    = zeros(N,1);
    external = true(1,N);
    lacking  = true(1,M);
    while any(lacking)
        % Redistribute the external solutions
        lacks = find(lacking);
        [~,subclass]    = min(ASF(external,lacks),[],2);
        class(external) = lacks(subclass);
        external = false(1,N);
        % Clip the exceeded subpopulations
        for i = 1 : M
            sub = find(class==i);
            if length(sub) > N/M
                [~,rank] = sort(ASF(sub,i));
                external(sub(rank(N/M+1:end))) = true;
            end
            lacking(i) = length(sub) < N/M;
        end
    end
    Subpopulation = cell(1,M);
    for i = 1 : M
        Subpopulation{i} = Population(class==i);
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,TPObj)
% The environmental selection of MaOEA-R&D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = Population.objs;
    SIn    = find(all(PopObj<=repmat(max(TPObj,[],1),size(PopObj,1),1),2))';
    if length(SIn) > N
        SNd = SIn(NDSort(PopObj(SIn,:),1)==1);
        if length(SNd) > N
            % Select part of the non-dominated solutions
            Next = DiversityOperator(PopObj(SNd,:),N,max([PopObj;TPObj],[],1),min([PopObj;TPObj],[],1));
            Next = SNd(Next);
        else
            % Select several dominated solutions with larger closest
            % distance to non-dominated solutions
            Sd = setdiff(SIn,SNd);
            MinDis = zeros(1,length(Sd));
            for i = 1 : length(Sd)
                MinDis(i) = min(sqrt(sum((repmat(PopObj(Sd(i),:),length(SNd),1)-PopObj(SNd,:)).^2,2)));
            end
            [~,rank] = sort(MinDis,'descend');
            Next = [SNd,Sd(rank(1:N-length(SNd)))];
        end
    else
        % Select several outside solutions which are closest to any target
        % points or their middle points
        TPpair = nchoosek(1:size(TPObj,1),2);
        MiddlePoint = zeros(size(TPpair,1),size(TPObj,2));
        for i = 1 : size(MiddlePoint)
            MiddlePoint(i,:) = (TPObj(TPpair(i,1),:)+TPObj(TPpair(i,2),:))/2;
        end
        Points = [MiddlePoint;TPObj];
        SOut   = setdiff(1:size(PopObj,1),SIn);
        MinDis = zeros(1,length(SOut));
        for i = 1 : length(SOut)
            MinDis(i) = min(sqrt(sum((repmat(PopObj(SOut(i),:),size(Points,1),1)-Points).^2,2)));
        end
        [~,rank] = sort(MinDis);
        Next = [SIn,SOut(rank(1:N-length(SIn)))];
    end
    % Population for next generation
    Population = Population(Next);
end

function Next = DiversityOperator(PopObj,N,fmax,fmin)
% Diversity improvement strategy

    N1   = size(PopObj,1);
    Next = 1 : N1;

    %% Calculate the grid location of each solution
    fmax = repmat(fmax,N1,1);
    fmin = repmat(fmin,N1,1);
    GLoc = N*(PopObj-fmin)./(fmax-fmin);
    
    %% Reduce the number of solutions
    % The grid-based distances between each two solutions
    A = pdist2(GLoc,GLoc);
    A(logical(eye(length(A)))) = inf;
    while length(Next) > N
        [dis,si] = min(A(Next,Next),[],2);
        si = [dis,(1:length(si))',si];
        eliminated = false(1,length(Next));
        while ~isempty(si) && length(Next)-sum(eliminated) > N
            [~,i1] = min(si(:,1));
            i1 = si(i1,2);
            eliminated(i1) = true;
            si(si(:,2)==i1 | si(:,3)==i1,:) = [];
        end
        Next(eliminated) = [];
    end
end
```

### `MaOEARD.m`
```matlab
classdef MaOEARD < ALGORITHM
% <2016> <many> <real/integer/label/binary/permutation>
% Many-objective evolutionary algorithm based on objective space reduction
% and diversity improvement

%------------------------------- Reference --------------------------------
% Z. He and G. G. Yen. Many-objective evolutionary algorithm: Objective
% space reduction and diversity improvement. IEEE Transactions on
% Evolutionary Computation, 2016, 20(1): 145-160.
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
            %% Generate the weight vectors
            W = zeros(Problem.M) + 1e-6;
            W(logical(eye(Problem.M))) = 1;

            %% Generate random population
            Problem.N  = max(ceil(Problem.N/Problem.M)*Problem.M,2*Problem.M);
            Population = Problem.Initialization();

            %% Objective space reduction
            % Reducing the objective space
            while Algorithm.NotTerminated(Population) && Problem.FE < Problem.maxFE/2
                % Classification
                [Subpopulation,Z] = Classification(Population,W);
                % Evolve each subpopulation
                for i = 1 : Problem.M
                    MatingPool = randi(Problem.N/Problem.M,1,Problem.N/Problem.M);
                    Offspring  = OperatorGA(Problem,Subpopulation{i}(MatingPool));
                    Subpopulation{i} = [Subpopulation{i},Offspring];
                    ASF = max((Subpopulation{i}.objs-repmat(Z,length(Subpopulation{i}),1))./repmat(W(i,:),length(Subpopulation{i}),1),[],2);
                    [~,rank] = sort(ASF);
                    Subpopulation{i} = Subpopulation{i}(rank(1:Problem.N/Problem.M));
                end
                Population = [Subpopulation{:}];
            end
            % Identify the target points
            TP = UpdateTP(Population,W);

            %% Diversity improvement
            % Generate random population around TPs
            PopDec     = repmat(TP.decs,Problem.N/Problem.M-1,1);
            PopDec     = PopDec.*(1+randn(size(PopDec))/5);
            Population = [TP,Problem.Evaluation(PopDec)];
            % Evolve
            while Algorithm.NotTerminated(Population)
                MatingPool = randi(Problem.N,1,Problem.N);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,TP.objs);
                TP         = UpdateTP([Population,TP],W);
            end
        end
    end
end
```

### `UpdateTP.m`
```matlab
function TP = UpdateTP(Population,W)
% Update the target points

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    TP = [];
    [Subpopulation,Z] = Classification(Population,W);
    for i = 1 : length(Subpopulation)
        ASF = max((Subpopulation{i}.objs-repmat(Z,length(Subpopulation{i}),1))./repmat(W(i,:),length(Subpopulation{i}),1),[],2);
        [~,extreme] = min(ASF);
        TP = [TP,Subpopulation{i}(extreme)];
    end
end
```
