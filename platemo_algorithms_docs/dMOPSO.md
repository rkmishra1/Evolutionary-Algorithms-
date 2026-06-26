# dMOPSO

**Tags**: <2011> <multi> <real/integer>

## Description
MOPSO based on decomposition

## Reference
S. Z. Martinez and C. A. Coello Coello. A multi-objective particle swarm optimizer based on decomposition. Proceedings of the Annual Conference on Genetic and Evolutionary Computation, 2011, 69-76.

## Source Code

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Particle,Pbest,Gbest,W)
% The particle swarm optimization in dMOPSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    if nargin < 5
        W = 0.4;
    end
    ParticleDec = Particle.decs;
    PbestDec    = Pbest.decs;
    GbestDec    = Gbest.decs;
    [N,D]       = size(ParticleDec);
    ParticleVel = Particle.adds(zeros(N,D));

    %% Particle swarm optimization
    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    OffVel = W.*ParticleVel + r1.*(PbestDec-ParticleDec) + r2.*(GbestDec-ParticleDec);
    OffDec = ParticleDec + OffVel;
    
    %% Deterministic back
    repair = OffDec < repmat(Problem.lower,N,1) | OffDec > repmat(Problem.upper,N,1);
    OffVel(repair) = -OffVel(repair);
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `UpdateGbest.m`
```matlab
function Gbest = UpdateGbest(W,Population,Z)
% Update the global best set

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% PBI value of each solution on each weight vector
    PopObj = Population.objs - repmat(Z,length(Population),1);
    Cosine = 1 - pdist2(PopObj,W,'cosine');
    NormP  = sqrt(sum(PopObj.^2,2));
    PBI    = repmat(NormP,1,size(W,1)).*(Cosine+5*sqrt(1-Cosine.^2));

    %% Find the global best particle of each weight vector
    Associate = zeros(1,size(W,1));
    Remain    = 1 : length(Population);
    for i = 1 : size(W,1)
        [~,best]     = min(PBI(Remain,i));
        Associate(i) = Remain(best);
        Remain(best) = [];
    end
    Gbest = Population(Associate);
end
```

### `dMOPSO.m`
```matlab
classdef dMOPSO < ALGORITHM
% <2011> <multi> <real/integer>
% MOPSO based on decomposition
% Ta --- 2 --- Age threshold

%------------------------------- Reference --------------------------------
% S. Z. Martinez and C. A. Coello Coello. A multi-objective particle swarm
% optimizer based on decomposition. Proceedings of the Annual Conference on
% Genetic and Evolutionary Computation, 2011, 69-76.
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
            Ta = Algorithm.ParameterSet(2);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);

            %% Generate random population
            Population = Problem.Initialization();
            Age        = zeros(1,Problem.N);         	% Age of each particle
            Z          = min(Population.objs,[],1);     % Ideal point
            Pbest      = Population;                    % Personal best
            Gbest      = UpdateGbest(W,Population,Z);   % Global best

            %% Optimization
            while Algorithm.NotTerminated(Gbest)
                % Generate offsprings
                Population(Age<Ta) = Operator(Problem,Population(Age<Ta),Pbest(Age<Ta),Gbest(Age<Ta));
                for i = find(Age>=Ta)
                    temp          = randn(1,Problem.D);
                    Population(i) = Problem.Evaluation((Gbest(i).dec-Pbest(i).dec)/2+temp.*abs(Gbest(i).dec-Pbest(i).dec));
                    Age(i)        = 0;
                end

                % Update Z
                Z = min(Z,min(Population.objs,[],1));

                % Update personal best
                PbeObj    = Pbest.objs - repmat(Z,Problem.N,1);
                PopObj    = Population.objs - repmat(Z,Problem.N,1);
                normW     = sqrt(sum(W.^2,2));
                normPbe   = sqrt(sum(PbeObj.^2,2));
                normPop   = sqrt(sum(PopObj.^2,2));
                CosinePbe = sum(PbeObj.*W,2)./normW./normPbe;
                CosinePop = sum(PopObj.*W,2)./normW./normPop;
                g_old     = normPbe.*CosinePbe + 5*normPbe.*sqrt(1-CosinePbe.^2);
                g_new     = normPop.*CosinePop + 5*normPop.*sqrt(1-CosinePop.^2);
                Pbest(g_new<=g_old) = Population(g_new<=g_old);
                Age(g_new<=g_old)   = 0;
                Age(g_new>g_old)    = Age(g_new>g_old) + 1;

                % Update global best
                Gbest = UpdateGbest(W,[Gbest,Population],Z);
            end
        end
    end
end
```
