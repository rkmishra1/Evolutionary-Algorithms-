# MOPSO-CD

**Tags**: <2005> <multi> <real/integer>

## Description
MOPSO with crowding distance

## Reference
C. R. Raquel and P. C. Naval Jr. An effective use of crowding distance in multiobjective particle swarm optimization. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2005, 257-264.

## Source Code

### `MOPSOCD.m`
```matlab
classdef MOPSOCD < ALGORITHM
% <2005> <multi> <real/integer>
% MOPSO with crowding distance

%------------------------------- Reference --------------------------------
% C. R. Raquel and P. C. Naval Jr. An effective use of crowding distance in
% multiobjective particle swarm optimization. Proceedings of the Annual
% Conference on Genetic and Evolutionary Computation, 2005, 257-264.
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
            %% Generate random population
            Population = Problem.Initialization();
            Archive    = UpdateArchive(Population,Problem.N);
            Pbest      = Population;

            %% Optimization
            while Algorithm.NotTerminated(Archive)
                Gbest      = Archive(randi(ceil(length(Archive)*0.1),1,Problem.N));
                Population = Operator(Problem,Population,Pbest,Gbest);
                Archive    = UpdateArchive([Archive,Population],Problem.N);
                Pbest      = UpdatePbest(Pbest,Population);
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Particle,Pbest,Gbest)
% Particle swarm optimization in MOPSO-CD

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
    C1 = repmat(unifrnd(1.5,2.5,N,1),1,D);
    C2 = repmat(unifrnd(1.5,2.5,N,1),1,D);
    OffVel = W.*ParticleVel + C1.*r1.*(PbestDec-ParticleDec) + C2.*r2.*(GbestDec-ParticleDec);
    OffDec = ParticleDec + OffVel;
    
    %% Deterministic back
    Lower  = repmat(Problem.lower,N,1);
    Upper  = repmat(Problem.upper,N,1);
    repair = OffDec < Lower | OffDec > Upper;
    OffVel(repair) = -OffVel(repair);
    OffDec = max(min(OffDec,Upper),Lower);
    
    %% Polynomial mutation
    if Problem.FE <= Problem.maxFE*0.5
        disM = 20;
        Site = rand(N,D) < 1/D;
        mu   = rand(N,D);
        temp = Site & mu<=0.5;
        OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                       (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        temp  = Site & mu>0.5; 
        OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                       (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    end
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `UpdateArchive.m`
```matlab
function Archive = UpdateArchive(Archive,N)
% Update the archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Find the non-dominated solutions
    Archive = Archive(NDSort(Archive.objs,1)==1);
    
    %% Truncate the archive according to the crowding distances
    while length(Archive) > N
        [~,rank] = sort(CrowdingDistance(Archive.objs));
        Archive(rank(randi(ceil(length(rank)*0.1)))) = [];
    end
    [~,rank] = sort(CrowdingDistance(Archive.objs),'descend');
    Archive  = Archive(rank);
end
```

### `UpdatePbest.m`
```matlab
function Pbest = UpdatePbest(Pbest,Population)
% Update the local best position of each particle

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    temp     = Pbest.objs - Population.objs;
    Dominate = any(temp<0,2) - any(temp>0,2);
    Pbest(Dominate==-1) = Population(Dominate==-1);
end
```
