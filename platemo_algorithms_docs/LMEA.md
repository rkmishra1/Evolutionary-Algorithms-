# LMEA

**Tags**: <2018> <multi/many> <real/integer> <large>

## Description
Evolutionary algorithm for large-scale many-objective optimization

## Reference
X. Zhang, Y. Tian, R. Cheng, and Y. Jin. A decision variable clustering based evolutionary algorithm for large-scale many-objective optimization. IEEE Transactions on Evolutionary Computation, 2018, 22(1): 97-112.

## Source Code

### `CorrelationAnalysis.m`
```matlab
function DVSet = CorrelationAnalysis(Problem,Population,DV,nCor)
% Detect the group of each distance variable

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    DVSet = {};
    for v = DV
        RelatedSet = [];
        for d = 1 : length(DVSet)
            for u = DVSet{d}
                drawnow('limitrate');
                sign = false;
                for i = 1 : nCor
                    p    = Population(randi(length(Population)));
                    a2   = unifrnd(Problem.lower(v),Problem.upper(v));
                    b2   = unifrnd(Problem.lower(u),Problem.upper(u));
                    decs = repmat(p.dec,3,1);
                    decs(1,v)     = a2;
                    decs(2,u)     = b2;
                    decs(3,[v,u]) = [a2,b2];
                    F = Problem.Evaluation(decs);
                    delta1 = F(1).obj - p.obj;
                    delta2 = F(3).obj - F(2).obj;
                    if any(delta1.*delta2<0)
                        sign = true;
                        RelatedSet = [RelatedSet,d];
                        break;
                    end
                end
                if sign
                    break;
                end
            end
        end
        if isempty(RelatedSet)
            DVSet = [DVSet,v];
        else
            DVSet = [DVSet,[cell2mat(DVSet(RelatedSet)),v]];
            DVSet(RelatedSet) = [];
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N)
% The environmental selection of distribution optimization in LMEA

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

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = Truncation(Population(Last).objs,N-sum(Next));
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Choose = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Calculate the normalized angle between each two solutions
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);
    Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    Cosine(logical(eye(length(Cosine)))) = 0;
    
    %% Truncation
    % Choose the extreme solutions first
    Choose = false(1,size(PopObj,1)); 
    [~,extreme] = max(PopObj,[],1);
    Choose(extreme) = true;
    % Choose the rest by truncation
    if sum(Choose) > K
        selected = find(Choose);
        Choose   = selected(randperm(length(selected),K));
    else
        while sum(Choose) < K
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
    end
end
```

### `LMEA.m`
```matlab
classdef LMEA < ALGORITHM
% <2018> <multi/many> <real/integer> <large>
% Evolutionary algorithm for large-scale many-objective optimization
% nSel ---  5 --- Number of selected solutions for decision variable clustering
% nPer --- 50 --- Number of perturbations on each solution for decision variable clustering
% nCor ---  5 --- Number of selected solutions for decision variable interaction analysis
% type ---  1 --- Type of operator (1. GA 2. DE)

%------------------------------- Reference --------------------------------
% X. Zhang, Y. Tian, R. Cheng, and Y. Jin. A decision variable clustering
% based evolutionary algorithm for large-scale many-objective optimization.
% IEEE Transactions on Evolutionary Computation, 2018, 22(1): 97-112.
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
            [nSel,nPer,nCor,type] = Algorithm.ParameterSet(5,50,5,1);

            %% Generate random population
            Population = Problem.Initialization();

            %% Detect the group of each distance variable
            [PV,DV] = VariableClustering(Problem,Population,nSel,nPer);
            DVSet   = CorrelationAnalysis(Problem,Population,DV,nCor);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Convergence optimization
                for i = 1 : 10
                    drawnow('limitrate');
                    Population = ConvergenceOptimization(Problem,Population,DVSet,type);
                end
                % Distribution optimization
                for i = 1 : Problem.M
                    drawnow();
                    Population = DistributionOptimization(Problem,Population,PV);
                end
            end
        end
    end
end

function Population = ConvergenceOptimization(Problem,Population,DVSet,type)
    [N,D] = size(Population.decs);
    Con   = calCon(Population.objs);
    for i = 1 : length(DVSet)
        for j = 1 : length(DVSet{i})
            % Select parents
            MatingPool = TournamentSelection(2,2*N,Con);
            % Generate offsprings
            OffDec = Population.decs;
            if type == 1
                NewDec = OperatorGAhalf(Problem,Population(MatingPool).decs,{1,20,D/length(DVSet{i})/2,20});
            elseif type == 2
                NewDec = OperatorDE(Problem,Population.decs,Population(MatingPool(1:end/2)).decs,Population(MatingPool(end/2+1:end)).decs,{1,0.5,D/length(DVSet{i})/2,20});
            end
            OffDec(:,DVSet{i}) = NewDec(:,DVSet{i});
            Offspring          = Problem.Evaluation(OffDec);
            % Update each solution
            allCon  = calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
            Con(updated)        = newCon(updated);
        end
    end
end

function Population = DistributionOptimization(Problem,Population,PV)
% Distribution optimization

    N            = length(Population);
    OffDec       = Population(TournamentSelection(2,N,calCon(Population.objs))).decs;
    NewDec       = OperatorGA(Problem,Population(randi(N,1,N)).decs);
    OffDec(:,PV) = NewDec(:,PV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end

function Con = calCon(PopObj)
% Calculate the convergence of each solution

    FrontNo = NDSort(PopObj,inf);
    Con     = sum(PopObj,2);
    Con     = FrontNo'*(max(Con)-min(Con)) + Con;
end
```

### `VariableClustering.m`
```matlab
function [PV,DV] = VariableClustering(Problem,Population,nSel,nPer)
% Detect the kind of each decision variable

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,D] = size(Population.decs);
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end
    
    %% Calculate the proper values of each decision variable
    Angle  = zeros(D,nSel);
    RMSE   = zeros(D,nSel);
    Sample = randi(N,1,nSel);
    for i = 1 : D
        drawnow('limitrate');
        % Generate several random solutions by perturbing the i-th dimension
        Decs      = repmat(Population(Sample).decs,nPer,1);
        Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
        newPopu   = Problem.Evaluation(Decs);
        for j = 1 : nSel
            % Normalize the objective values of the current perturbed solutions
            Points = newPopu(j:nSel:end).objs;
            Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
            Points = Points - repmat(mean(Points,1),nPer,1);
            % Calculate the direction vector of the determining line
            [~,~,V] = svd(Points);
            Vector  = V(:,1)'./norm(V(:,1)');
            % Calculate the root mean square error
            error = zeros(1,nPer);
            for k = 1 : nPer
                error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
            end
            RMSE(i,j) = sqrt(sum(error.^2));
            % Calculate the angle between the line and the hyperplane
            normal     = ones(1,size(Vector,2));
            cosine     = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
            Angle(i,j) = real(acos(cosine)/pi*180);
        end
    end
    
    %% Detect the kind of each decision variable
    VariableKind = (mean(RMSE,2)<1e-2)';
    result       = kmeans(Angle,2)';
    if any(result(VariableKind)==1) && any(result(VariableKind)==2)
        if mean(mean(Angle(result==1&VariableKind,:))) < mean(mean(Angle(result==2&VariableKind,:)))
            VariableKind = VariableKind & result==1;
        else
            VariableKind = VariableKind & result==2;
        end
    end
    PV = find(~VariableKind);
    DV = find(VariableKind);
end
```
