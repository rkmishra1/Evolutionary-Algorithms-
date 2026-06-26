# CoMMEA

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <multimodal>

## Description
Coevolutionary multimodal multi-objective evolutionary algorithm

## Reference
W. Li, X. Yao, K. Li, R. Wang, T. Zhang, and  L. Wang. Coevolutionary framework for generalized multimodal multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2023, 10(7): 1544-1556.

## Source Code

### `CalFitness.m`
```matlab
function Fitness = CalFitness(Population,Operation)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = Population.objs;
    PopDec = Population.decs;
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

    if strcmp(Operation,'LocalC') % Dividing all solution into niches
        dist = pdist2(PopDec,PopDec);

        %% Method 1: distance based
        if size(PopDec,2)<=8
            R = mean(mean(dist))/4;
        else
            R = mean(mean(dist))/2;
        end
        Niching = dist<R;
        Dominate = Dominate & Niching;
    end

    %% Calculate S(i)
    S = sum(Dominate,2);

    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

    %% Calculate the fitnesses
    Fitness = R;
end
```

### `CoMMEA.m`
```matlab
classdef CoMMEA < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <multimodal>
% Coevolutionary multimodal multi-objective evolutionary algorithm
% eps --- 0.2 --- Parameter for quality of the local Pareto front (suggested to be 0 if the problem has no local Pareto front).

%------------------------------- Reference --------------------------------
% W. Li, X. Yao, K. Li, R. Wang, T. Zhang, and  L. Wang. Coevolutionary
% framework for generalized multimodal multi-objective optimization.
% IEEE/CAA Journal of Automatica Sinica, 2023, 10(7): 1544-1556.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li
% p.s. CoMMEA require more function evaluations to receive better
% performance for multimodal multiobjective optimization problems.
    
    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            eps = Algorithm.ParameterSet(0.2);
            
           %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Fitness1    = CalFitness(Population1,'Normal');
            Fitness2    = CalFitness(Population2,'LocalC');
            
           %% Optimization
            while Algorithm.NotTerminated(Population2)
                MatingPool1 = TournamentSelection(2,round(Problem.N/4),Fitness1);
                MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                Offspring2  = OperatorGA(Problem,Population2(MatingPool2));
                
               %% Enviornmental selection
                EvoState = Problem.FE / Problem.maxFE;
                CurEps = max(-log2(1.5*EvoState),eps); % Compute the current epsilon value
                [Population1,Fitness1] = EnvironmentalSelection1([Population1,Offspring1,Offspring2],Problem.N,'Normal',EvoState);
                [Population2,Fitness2] = EnvironmentalSelection2([Population2,Offspring1,Offspring2],Problem.N,'LocalC',CurEps);
            end
        end
    end
end
```

### `Crowding.m`
```matlab
function CrowdDis = Crowding(Pop)
% Harmonic average distance of each solution in the decision space
% return: the crowding distance of each individual

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N        = size(Pop,1);
    K        = N-1;
    Z        = min(Pop,[],1);
    Zmax     = max(Pop,[],1);
    pop      = (Pop-repmat(Z,N,1))./repmat(Zmax-Z,N,1);
    distance = pdist2(pop,pop);
    value    = sort(distance,2);
    CrowdDis = K./sum(1./value(:,2:N),2);
end
```

### `EnvironmentalSelection1.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection1(Population,N,Operation,EvoState)
% The environmental selection of CoMMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% SPEA2 selection
    Fitness = CalFitness(Population,Operation);
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sortrows([Fitness' -Crowding(Population.decs)]);
        Population = Population(Rank(1:N));
    else
        Population = Population(Next);
        while length(Population)>N
            dist = sort(pdist2(Population.decs,Population.decs));
            CrowdDis = sum(dist(1:3,:));
            [~,index] = min(CrowdDis);
            Population(index) = [];
        end
    end
    if EvoState<0.5
        Fitness=CalFitness(Population,Operation)-Crowding(Population.decs)';
    else
        Fitness=-Crowding(Population.decs)';
    end
end
```

### `EnvironmentalSelection2.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection2(Population,N,Operation,eps)
% The environmental selection of CoMMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Epsilon-dominance-based method
    [FrontNo,~] = NDSort(Population.objs,N);
    Next = find(FrontNo == 1);
    Remain = find(FrontNo > 1);
    isEpsSol = false(1,numel(Remain));

    for j=1:numel(Remain)
        TmpObj = [Population(Remain(j)).objs; (1+eps).*Population(Next).objs];
        [fno,~] = NDSort(TmpObj,inf);
        if fno(1)==1
            isEpsSol(j) = true;
        end
    end

    if length(Next) + sum(isEpsSol) < N
        Cdis = -Crowding(Population.decs);
        [~,idx] = sortrows([FrontNo' Cdis],'ascend');
        Population = Population(idx(1:N));
    else
        Population = [Population(Next) Population(Remain(isEpsSol))];
    end

    %% Calculate local convergence quality
    Fitness = CalFitness(Population,Operation);

    %% Crowding distance based second selection
    Next = Fitness < 1;
    K = 3;
    if sum(Next) < N
        dist = sort(pdist2(Population.decs,Population.decs));
        CrowdDis = -sum(dist(1:K,:));
        [~,idx] = sortrows([Fitness' CrowdDis']);
        Population = Population(idx(1:N));
    else
        Population = Population(Next);
        while length(Population)>N
            dobj = sort(pdist2(Population.objs,Population.objs)); dobj = sum(dobj(1:K,:));%dobj./max(max(dobj));
            ddec = sort(pdist2(Population.decs,Population.decs)); ddec = sum(ddec(1:K,:));%ddec./max(max(ddec));
            CrowdDis = dobj./max(dobj) + ddec./max(ddec);
            [~,index] = min(CrowdDis);
            Population(index) = [];
        end
    end

    %% Calculate fitness for parents selection
    Fitness = Kdis(Population,N/2);
end

function fDN = Kdis(Pop,K)
    PopObj=Pop.objs;
    PopDec=Pop.decs;
    Np = size(PopObj,1);
    d_dec = pdist2(PopDec,PopDec,'euclidean');
    d_dec(logical(eye(Np))) = inf;
    sdd = sort(d_dec);
    dn_dec = sum(sdd(1:K,:));
    avg_dn_dec = mean(dn_dec);
    if avg_dn_dec == 0
        avg_dn_dec = inf;
    end
    fDN = 1./(1+dn_dec./avg_dn_dec);
end
```
