# MMOPSO

**Tags**: <2015> <multi> <real/integer>

## Description
MOPSO with multiple search strategies

## Reference
Q. Lin, J. Li, Z. Du, J. Chen, and Z. Ming. A novel multi-objective particle swarm optimization with multiple search strategies. European Journal of Operational Research, 2015, 247(3): 732-744.

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

### `GetBest.m`
```matlab
function [Pbest,Gbest] = GetBest(Archive,W,Z)
% Update the pbest and gbest of each particle

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select the pbest
    AObj  = Archive.objs - repmat(Z,length(Archive),1);
    NormW = sqrt(sum(W.^2,2))';
    d1    = AObj*W'./repmat(NormW,length(Archive),1);
    d2    = sqrt(repmat(sum(AObj.^2,2),1,size(W,1))-d1.^2);
    PBI   = d1 + 5*d2;
    [~,p] = min(PBI,[],1);
    Pbest = Archive(p);
    
    %% Select the gbest
    Gbest = Archive(randi(size(Archive,1),1,size(W,1)));
end
```

### `MMOPSO.m`
```matlab
classdef MMOPSO < ALGORITHM
% <2015> <multi> <real/integer>
% MOPSO with multiple search strategies

%------------------------------- Reference --------------------------------
% Q. Lin, J. Li, Z. Du, J. Chen, and Z. Ming. A novel multi-objective
% particle swarm optimization with multiple search strategies. European
% Journal of Operational Research, 2015, 247(3): 732-744.
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

            %% Generate random population
            Population = Problem.Initialization();
            Z          = min(Population.objs,[],1);
            Population = Classification(Problem,Population,W,Z);
            Archive    = UpdateArchive(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Archive)
                [Pbest,Gbest] = GetBest(Archive,W,Z);
                Population    = Operator(Problem,Population,Pbest,Gbest);
                Z             = min([Z;Population.objs],[],1);
                Archive       = UpdateArchive([Archive,Population],Problem.N); 
                S             = OperatorGAhalf(Problem,Archive([1:length(Archive),randi(ceil(length(Archive)/2),1,length(Archive))]));
                Z             = min([Z;S.objs],[],1);
                Archive       = UpdateArchive([Archive,S],Problem.N);
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Particle,Pbest,Gbest)
% Particle swarm optimization in MMOPSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    ParticleDec = Particle.decs;
    PbestDec    = Pbest.decs;
    GbestDec    = Gbest.decs;
    [N,D]       = size(ParticleDec);
    ParticleVel = Particle.adds(zeros(N,D));

    %% Particle swarm optimization
    W  = repmat(unifrnd(0.1,0.5,N,1),1,D);
    r1 = repmat(rand(N,1),1,D);
    r2 = repmat(rand(N,1),1,D);
    C1 = repmat(unifrnd(1.5,2,N,1),1,D);
    C2 = repmat(unifrnd(1.5,2,N,1),1,D);
    OffVel        = W.*ParticleVel;
    temp          = repmat(rand(N,1)<0.7,1,D);
    OffVel(temp)  = OffVel(temp) + C1(temp).*r1(temp).*(PbestDec(temp)-ParticleDec(temp));
    OffVel(~temp) = OffVel(~temp) + C2(~temp).*r2(~temp).*(GbestDec(~temp)-ParticleDec(~temp));
    OffDec        = ParticleDec + OffVel;
    Offspring     = Problem.Evaluation(OffDec,OffVel);
end
```

### `UpdateArchive.m`
```matlab
function A = UpdateArchive(A,N)
% Update the external archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select only the non-dominated solutions in the archive
    A = A(NDSort(A.objs,1)==1);
    
    %% Sort the solutions in A according to their crowding distances
    [~,rank] = sort(CrowdingDistance(A.objs),'descend');
    A        = A(rank(1:min(N,end)));
end
```
