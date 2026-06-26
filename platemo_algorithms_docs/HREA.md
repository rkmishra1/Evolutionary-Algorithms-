# HREA

**Tags**: <2023> <multi> <real/integer> <multimodal>

## Description
Hierarchy ranking based evolutionary algorithm

## Reference
W. Li, X. Yao, T. Zhang, R. Wang, and L. Wang. Hierarchy ranking method for multimodal multi-objective optimization with local Pareto fronts. IEEE Transactions on Evolutionary Computation, 2023, 27(1): 98-110.

## Source Code

### `ArchiveUpdate.m`
```matlab
function [Population,CrowdDis] = ArchiveUpdate(Population,N,eps,st)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    n = length(Population);

    if eps~=1 && st<0.5
       eps = 2*(1-eps)/(2*st+1)+2*eps-1;
    end

    %% Select global Pareto front
    [FrontNo,MaxFNo] = NDSort(Population.objs,n);
    next             = FrontNo==1;
    first_pf         = Population(next);
    new_pop          = first_pf;
    remain_pop       = Population(~next);

    V = 0.2*prod(max(Population.decs)-min(Population.decs)).^(1./size(Population.decs,2));

    while ~isempty(remain_pop)
        %% Delete close solutions
        dist  = min(pdist2(new_pop.decs,remain_pop.decs));
        index = dist<V;
        remain_pop(index) = [];
        if isempty(remain_pop)
            break;
        end

        %% Select remaining solutions
        [FrontNo,~] = NDSort(remain_pop.objs,length(remain_pop));
        pick_pop    = remain_pop(FrontNo==1);
        [nF,~]      = NDSort([pick_pop.objs .* (1-eps); first_pf.objs],length(pick_pop)+length(first_pf));
        nF          = nF(1:length(pick_pop));
        maxnF       = max(nF);

        if maxnF>1
            new_pop    = [new_pop pick_pop(nF==1)];
            remain_pop = remain_pop(FrontNo~=1);
            break;
        else
            new_pop    = [new_pop pick_pop];
            remain_pop = remain_pop(FrontNo~=1);
        end
    end
    Population = new_pop;

    %% Balance the number of solutions in each Pareto front
    if length(Population) > N
        awd_index        = [];
        [FrontNo,MaxFNo] = NDSort(Population.objs,length(Population));
        new_pop          = [];
        n_sub_pop        = ceil(N/MaxFNo);
        sel_pop          = [];
        tmp_pop          = [];
        for i = 1 : MaxFNo
            pop = Population(FrontNo==i);
            if length(pop) < n_sub_pop
                sel_pop   = [sel_pop pop];
                awd_index = [awd_index (n_sub_pop-length(pop)).*ones(1,length(pop))];
            else
                tmp_pop = [tmp_pop pop];
            end
        end
        while length(tmp_pop) > N - length(sel_pop)
            dist         = pdist2(tmp_pop.decs,tmp_pop.decs);
            dist         = sort(dist);
            dist         = sum(dist(1:3,:),1);
            [~,ind]      = min(dist);
            tmp_pop(ind) = [];
        end
        awd_index  = [awd_index zeros(1,length(tmp_pop))] + 1;
        Population = [sel_pop tmp_pop];
        CrowdDis   = Crowding(Population.decs);
        CrowdDis   = CrowdDis.* awd_index';
    else
        CrowdDis = Crowding(Population.decs);
    end
end
```

### `Crowding.m`
```matlab
function [CrowdDis]=Crowding(Pop)
%Harmonic average distance of each solution in the decision space
%Return: the crowding distance of each individual

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    [N,~]     = size(Pop);
    K         = N-1;
    Z         = min(Pop,[],1);
    Zmax      = max(Pop,[],1);
    pop       = (Pop-repmat(Z,N,1))./repmat(Zmax-Z,N,1);
    distance  = pdist2(pop,pop);
    [value,~] = sort(distance,2);
    CrowdDis  = K./sum(1./value(:,2:N),2);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,CrowdDis] = EnvironmentalSelection(Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    n = length(Population);

    %% Cal the local convergence
    dist = pdist2(Population.decs,Population.decs);
    V    = 0.2*prod(max(Population.decs)-min(Population.decs)).^(1./size(Population.decs,2));

    DominationX = zeros(n); % Pareto domination relationship between pair of solutions
    for i = 1 : n
        for j = i+1 : n
            if dist(i,j) > V
                continue
            end
            L1 = Population(i).objs < Population(j).objs;
            L2 = Population(i).objs > Population(j).objs;
            if all(L1|(~L2))
                DominationX(i,j) = 0;
                DominationX(j,i) = 1;
            elseif all(L2|(~L1))
                DominationX(i,j) = 1;
                DominationX(j,i) = 0;
            end
        end
    end

    LocalC = zeros(1,n);
    for i = 1 : n
        tmp       = dist(i,:);
        index     = tmp<V;
        LocalC(i) = (sum(DominationX(i,index)))./sum(index);
    end

    dist     = sort(pdist2(Population.decs,Population.decs));
    CrowdDis = sum(dist(1:3,:));

    [~,index]  = sortrows([LocalC' -CrowdDis']);
    Population = Population(index);

    if length(Population)>N
        Population = Population(1:N);
    end

    CrowdDis = Crowding(Population.decs);
end
```

### `HREA.m`
```matlab
classdef HREA < ALGORITHM
% <2023> <multi> <real/integer> <multimodal>
% Hierarchy ranking based evolutionary algorithm
% eps --- 0.3 --- Parameter for quality of the local Pareto front

%------------------------------- Reference --------------------------------
% W. Li, X. Yao, T. Zhang, R. Wang, and L. Wang. Hierarchy ranking method
% for multimodal multi-objective optimization with local Pareto fronts.
% IEEE Transactions on Evolutionary Computation, 2023, 27(1): 98-110.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    methods
        function main(Algorithm, Problem)
            eps = Algorithm.ParameterSet(0.3);
            p   = 0.5;
            %% Generate random population
            Population          = Problem.Initialization();
            [~,CrowdDis1]       = EnvironmentalSelection(Population,Problem.N);
            [Archive,CrowdDis2] = ArchiveUpdate(Population,Problem.N,eps,0);

            %% Optimization
            while Algorithm.NotTerminated(Archive)
                if Problem.FE >= Problem.maxFE * 0.5 && rand < p
                    MatingPool2 = TournamentSelection(2,round(Problem.N),-CrowdDis2);
                    Offspring   = OperatorGA(Problem,Archive(MatingPool2));
                else
                    MatingPool1 = TournamentSelection(2,round(Problem.N),-CrowdDis1);
                    Offspring   = OperatorGA(Problem,Population(MatingPool1));
                end
                [Population,CrowdDis1] = EnvironmentalSelection([Population,Offspring],Problem.N);
                [Archive,CrowdDis2]    = ArchiveUpdate([Archive,Offspring],Problem.N,eps,Problem.FE/Problem.maxFE);
            end
        end
    end
end
```
