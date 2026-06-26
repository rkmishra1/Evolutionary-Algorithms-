# BCE-IBEA

**Tags**: <2016> <multi/many> <real/integer/label/binary/permutation>

## Description
Bi-criterion evolution based IBEA

## Reference
M. Li, S. Yang, and X. Liu. Pareto or non-Pareto: Bi-criterion evolution in multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 645-665.

## Source Code

### `BCEIBEA.m`
```matlab
classdef BCEIBEA < ALGORITHM
% <2016> <multi/many> <real/integer/label/binary/permutation>
% Bi-criterion evolution based IBEA
% kappa --- 0.05 --- Fitness scaling factor

%------------------------------- Reference --------------------------------
% M. Li, S. Yang, and X. Liu. Pareto or non-Pareto: Bi-criterion evolution
% in multiobjective optimization. IEEE Transactions on Evolutionary
% Computation, 2016, 20(5): 645-665.
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
            kappa = Algorithm.ParameterSet(0.05);

            %% Generate random population
            NPC      = Problem.Initialization();
            [PC,nND] = PCSelection(NPC,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(PC)
                % PC evolving
                NewPC = Exploration(Problem,PC,NPC,nND,Problem.N);
                % NPC selection
                NPC = EnvironmentalSelection([NPC,NewPC],Problem.N,kappa);
                % NPC evolving
                MatingPool = TournamentSelection(2,Problem.N,-CalFitness(NPC.objs,kappa));
                NewNPC     = OperatorGA(Problem,NPC(MatingPool));
                NPC        = EnvironmentalSelection([NPC,NewNPC],Problem.N,kappa);
                % PC selection
                [PC,nND] = PCSelection([PC,NewNPC,NewPC],Problem.N);
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function [Fitness,I,C] = CalFitness(PopObj,kappa)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,kappa)
% The environmental selection of IBEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Next = 1 : length(Population);
    [Fitness,I,C] = CalFitness(Population.objs,kappa);
    while length(Next) > N
        [~,x]   = min(Fitness(Next));
        Fitness = Fitness + exp(-I(Next(x),:)/C(Next(x))/kappa);
        Next(x) = [];
    end
    Population = Population(Next);
end
```

### `Exploration.m`
```matlab
function Offspring = Exploration(Problem,PC,NPC,nND,N)
% Individual exploration in BCE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PCObj  = PC.objs;
    NPCObj = NPC.objs;
    
    %% Normalization
    fmax   = max(PCObj,[],1);
    fmin   = min(PCObj,[],1);
    PCObj  = (PCObj-repmat(fmin,size(PCObj,1),1))./repmat(fmax-fmin,size(PCObj,1),1);
    NPCObj = (NPCObj-repmat(fmin,size(NPCObj,1),1))./repmat(fmax-fmin,size(NPCObj,1),1);

    %% Determine the size of the niche
    d  = pdist2(PCObj,PCObj);
    d(logical(eye(length(d)))) = inf;
    d  = sort(d,2);
    r0 = mean(d(:,min(3,size(d,2))));
    r  = nND/N*r0;
    
    %% Detect the solutions in PC to be explored
    d = pdist2(PCObj,NPCObj);
    S = find(sum(d<=r,2)<=1);
    
    %% Generate new solutions
    if ~isempty(S)
        MatingPool = randi(length(PC),1,length(S));
        Offspring  = OperatorGAhalf(Problem,PC([S',MatingPool]));
    else
        Offspring = [];
    end
end
```

### `PCSelection.m`
```matlab
function [PC,nND] = PCSelection(PC,N)
% PC selection and population maintenance in BCE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% PC selection
    PC    = PC(NDSort(PC.objs,1)==1);
    PC    = PC(randperm(length(PC)));
    PCObj = PC.objs;
    nND   = length(PC);
    
    %% Population maintenance
    if length(PC) > N
        % Normalization
        fmax  = max(PCObj,[],1);
        fmin  = min(PCObj,[],1);
        PCObj = (PCObj-repmat(fmin,nND,1))./repmat(fmax-fmin,nND,1);
        % Determine the radius of the niche
        d  = pdist2(PCObj,PCObj);
        d(logical(eye(length(d)))) = inf;
        sd = sort(d,2);
        r  = mean(sd(:,min(3,size(sd,2))));
        R  = min(d./r,1);
        % Delete solution one by one
        while length(PC) > N
            [~,worst]  = max(1-prod(R,2));
            PC(worst)  = [];
            R(worst,:) = [];
            R(:,worst) = [];
        end
    end
end
```
