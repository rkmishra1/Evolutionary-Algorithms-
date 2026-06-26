# MMEA-WI

**Tags**: <2021> <multi> <real/integer> <multimodal>

## Description
Weighted indicator-based evolutionary algorithm for multimodal multi-objective optimization

## Reference
W. Li, T. Zhang, R. Wang, and H. Ishibuchi. Weighted indicator-based evolutionary algorithm for multimodal multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6): 1064-1078.

## Source Code

### `CalFitness.m`
```matlab
function [Fitness,I,C] = CalFitness(Population,kappa)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    PopObj = Population.objs;
    N      = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C       = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection(Population,N,kappa,state)
% The environmental selection of IBEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    Next = 1 : length(Population);
    [Fitness,I,C] = CalFitness(Population,kappa);

    dist = pdist2(Population.decs,Population.decs);

    var = Population.decs;

    sigma = prod(max(var)-min(var))^(1/size(var,2))/size(var,1);
    sigma = sigma*(exp(-state));

    weight = 1/(sigma*(2*pi)^0.5);
    weight = weight.*exp(-dist.^2/(2*sigma^2));

    newfit = zeros(size(Fitness));
    for i = 1 : size(var,1)
        tmp = weight(i,:).*(Fitness);
        newfit(i) = sum(tmp);
    end

    weight = sum(weight);
    newfit = newfit.*weight;

    Fitness = newfit;

    while length(Next) > N
        [~,x]   = min(Fitness(Next));
        Fitness = Fitness + exp(-I(Next(x),:)/C(Next(x))/kappa);
        Next(x) = [];
    end
    Population = Population(Next);

    Fitness = Kdis(Population,N/2);    
end

function fDN = Kdis(Pop,K)
    PopObj = Pop.objs;
    PopDec = Pop.decs;
    Np     = size(PopObj,1);
    d_dec  = pdist2(PopDec,PopDec,'euclidean');
    d_dec(logical(eye(Np))) = inf;
    sdd    = sort(d_dec);
    dn_dec = sum(sdd(1:K,:));
    avg_dn_dec = mean(dn_dec);
    if avg_dn_dec == 0
        avg_dn_dec = inf;
    end
    fDN = 1./(1+dn_dec./avg_dn_dec);
end
```

### `MMEAWI.m`
```matlab
classdef MMEAWI < ALGORITHM
% <2021> <multi> <real/integer> <multimodal>
% Weighted indicator-based evolutionary algorithm for multimodal multi-objective optimization

%------------------------------- Reference --------------------------------
% W. Li, T. Zhang, R. Wang, and H. Ishibuchi. Weighted indicator-based
% evolutionary algorithm for multimodal multiobjective optimization. IEEE
% Transactions on Evolutionary Computation, 2021, 25(6): 1064-1078.
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
            %% Parameter setting
            kappa = 0.05;
            t_gen = ceil(Problem.maxFE*0.4);

            %% Generate random population
            Population        = Problem.Initialization();
            [Population,pfit] = EnvironmentalSelection(Population,Problem.N,kappa,1);
            [Arcd,afit]       = UpdateArc(Population,Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Arcd)
                if Problem.FE>=t_gen && rand>0.5 % Stage 2
                    [~,p1] = min(afit);
                    joint  = [Arcd Population];
                    dist   = pdist2(Arcd(p1).decs,joint.decs);
                    [~,so] = sort(dist,'ascend');
                    MatingPool = so(randperm(round(Problem.N/5),round(Problem.N/10))+1);
                    parents    = [Arcd(p1) joint(MatingPool)];
                    Offspring  = OperatorGA(Problem,parents);
                    [Population,pfit] = EnvironmentalSelection([Population,Offspring],Problem.N,kappa,Problem.FE/Problem.maxFE);
                    [Arcd,afit]       = UpdateArc(Arcd,Offspring,Problem.N);
                else % Stage 1
                    MatingPool  = TournamentSelection(2,round(Problem.N/10),pfit);
                    Offspring   = OperatorGA(Problem,Population(MatingPool));
                    [Population,pfit] = EnvironmentalSelection([Population,Offspring],Problem.N,kappa,Problem.FE/Problem.maxFE);
                    [Arcd,afit] = UpdateArc(Arcd,Offspring,Problem.N);
                end
            end
        end
    end
end
```

### `UpdateArc.m`
```matlab
function [Population,dk] = UpdateArc(Population,offspring,N)
% Update the archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Wenhua Li

    joint = [Population offspring];

    [FrontNo,MaxFNo] = NDSort(joint.objs,N);
    next             = FrontNo==1;
    Population       = joint(next);

    [Choose,fDKN] = DoubleNearestSelection(Population,N,5);
    Population    = Population(Choose);
    dk            = fDKN(Choose);
end

function [Choose,fDN] = DoubleNearestSelection(Pop,N,K)
    PopObj = Pop.objs;
    PopDec = Pop.decs;
    Np     = size(PopObj,1);
    Choose = true(1,Np);
    if Np <= K
        fDN = zeros(1,Np);
        return;
    end
    d_obj = pdist2(PopObj,PopObj,'euclidean');
    d_dec = pdist2(PopDec,PopDec,'euclidean');
    d_obj(logical(eye(Np))) = inf;
    d_dec(logical(eye(Np))) = inf;

    sdo    = sort(d_obj);
    sdd    = sort(d_dec);
    dn_obj = sum(sdo(1:K,:));
    dn_dec = sum(sdd(1:K,:));
    avg_dn_obj = mean(dn_obj);
    avg_dn_dec = mean(dn_dec);
    if avg_dn_obj == 0
        avg_dn_obj = inf;
    end
    if avg_dn_dec == 0
        avg_dn_dec = inf;
    end
    fDN = 1./(1+dn_obj./avg_dn_obj+dn_dec./avg_dn_dec);

    while sum(Choose) > N
        [~,Del] = max(fDN);

        Choose(Del)  = false;
        d_obj(Del,:) = inf;
        d_obj(:,Del) = inf;
        d_obj(Del,:) = inf;
        d_obj(:,Del) = inf;

        sdo    = sort(d_obj);
        sdd    = sort(d_dec);
        dn_obj = sum(sdo(1:K,:));
        dn_dec = sum(sdd(1:K,:));
        avg_dn_obj = mean(dn_obj);
        avg_dn_dec = mean(dn_dec);
        if avg_dn_obj == 0
            avg_dn_obj = inf;
        end
        if avg_dn_dec == 0
            avg_dn_dec = inf;
        end
        fDN = 1./(1+dn_obj./avg_dn_obj+dn_dec./avg_dn_dec);
        fDN(~Choose) = -inf;
    end
end
```
