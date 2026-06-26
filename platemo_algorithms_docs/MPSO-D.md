# MPSO-D

**Tags**: <2015> <multi/many> <real/integer>

## Description
Multi-objective particle swarm optimization algorithm based on

## Reference
C. Dai, Y. Wang, and M. Ye. A new multi-objective particle swarm optimization algorithm based on decomposition. Information Sciences, 2015, 325: 541-557.

## Source Code

### `Classification.m`
```matlab
function Next = Classification(Problem,Population,W,Z)
% Classify solutions into sub-regions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the cosine value of each solution to each vector
    % Note that here the value of each solution on each vector is not the
    % cosine value which is proposed in the paper
    PopObj = Population.objs - repmat(Z,length(Population),1);
    Value  = PopObj*W';
    [~,P]  = max(Value,[],2);
    
    %% Select one solution for each sub-region
    Next = Population(1:size(W,1));
    for i = 1 : size(W,1)
        Current = find(P==i);
        if isempty(Current)
            Next(i) = Problem.Initialization(1);
        else
            ND       = find(NDSort(Population(Current).objs,1)==1);
            [~,best] = max(PopObj(Current(ND),:)*W(i,:)'./sum(PopObj(Current(ND),:).^2,2).^0.6);
            Next(i)  = Population(Current(ND(best)));
        end
    end
end
```

### `MPSOD.m`
```matlab
classdef MPSOD < ALGORITHM
% <2015> <multi/many> <real/integer>
% Multi-objective particle swarm optimization algorithm based on
% decomposition

%------------------------------- Reference --------------------------------
% C. Dai, Y. Wang, and M. Ye. A new multi-objective particle swarm
% optimization algorithm based on decomposition. Information Sciences,
% 2015, 325: 541-557.
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
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            W = W./repmat(sqrt(sum(W.^2,2)),1,size(W,2));
            T = ceil(Problem.N/10);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization(2*Problem.N);
            Z          = min(Population.objs,[],1);
            Population = Classification(Problem,Population,W,Z);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                [Parent,Pbest,Gbest] = MatingSelection(Population.objs,B,W,Z);
                Offspring  = Operator(Problem,Population(Parent),Population(Pbest),Population(Gbest));
                Z          = min([Z;Offspring.objs],[],1);
                Population = Classification(Problem,[Population,Offspring],W,Z);
            end
        end
    end
end
```

### `MatingSelection.m`
```matlab
function [Parent,Pbest,Gbest] = MatingSelection(PopObj,B,W,Z)
% Mating selection of MPSO/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);

    %% Select N parents according to their crowding distances
    PopObj = PopObj - repmat(Z,size(PopObj,1),1);
    Parent = TournamentSelection(2,N,-CrowdingDistance(PopObj));
    
    %% Select the pbest and gbest of each parent
    Pbest = zeros(size(Parent));
    Gbest = zeros(size(Parent));
    for i = 1 : N
        if rand < 0.9
            P = B(i,:);
        else
            P = 1 : N;
        end
        Pbest(i) = P(randi(length(P)));
        [~,best] = max(PopObj(P,:)*mean(W(P,:),1)'./sum(PopObj(P,:).^2,2).^0.6);
        Gbest(i) = P(best);
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Particle,Pbest,Gbest,Parameter)
% Particle swarm optimization in MPSO/D
% c1   ---   2 --- Parameter in updating particle's velocity
% c2   ---   2 --- Parameter in updating particle's velocity
% CR   --- 0.5 --- Parameter CR in differental evolution
% F    --- 0.5 --- Parameter F in differental evolution
% proM ---   1 --- The expectation of number of bits doing mutation 
% disM ---  20 --- The distribution index of polynomial mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    if nargin > 4
        [c1,c2,CR,F,proM,disM] = deal(Parameter{:});
    else
        [c1,c2,CR,F,proM,disM] = deal(2,2,0.5,0.5,1,20);
    end
    ParticleDec = Particle.decs;
    PbestDec    = Pbest.decs;
    GbestDec    = Gbest.decs;
    [N,D]       = size(ParticleDec);
    ParticleVel = Particle.adds(zeros(N,D));
    
    %% Particle swarm optimization
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    DoPSO = repmat(rand(N,1)<0.5,1,D);
    W     = 0.9 - Problem.FE./Problem.maxFE*0.8;
    r1    = repmat(rand(N,1),1,D);
    r2    = repmat(rand(N,1),1,D);
    OffVel        = ParticleVel;
    OffDec        = ParticleDec;
    OffVel(DoPSO) = W.*ParticleVel(DoPSO) + c1.*r1(DoPSO).*(PbestDec(DoPSO)-ParticleDec(DoPSO)) + c2.*r2(DoPSO).*(GbestDec(DoPSO)-ParticleDec(DoPSO));
    OffDec(DoPSO) = ParticleDec(DoPSO) + OffVel(DoPSO);
    % Set the infeasible decision variables to the value of their parents
    Invalid         = OffDec < Lower | OffDec > Upper;
    OffDec(Invalid) = ParticleDec(Invalid);
    
    %% DE
    Site = ~DoPSO & rand(N,D)<CR;
    OffDec(Site) = ParticleDec(Site) + F.*(GbestDec(Site)-PbestDec(Site));
    % Set the infeasible decision variables to boundary values
    OffDec = max(min(OffDec,Upper),Lower);

    %% Polynomial mutation
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```
