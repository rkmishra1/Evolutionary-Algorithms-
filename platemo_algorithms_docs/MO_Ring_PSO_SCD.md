# MO_Ring_PSO_SCD

**Tags**: <2018> <multi> <real/integer> <multimodal>

## Description
Multiobjective PSO using ring topology and special crowding distance

## Reference
C. Yue, B. Qu, and J. Liang. A multiobjective particle swarm optimizer using ring topology for solving multimodal multiobjective problems. IEEE Transactions on Evolutionary Computation, 2018, 22(5): 805-817.

## Source Code

### `MO_Ring_PSO_SCD.m`
```matlab
classdef MO_Ring_PSO_SCD < ALGORITHM
% <2018> <multi> <real/integer> <multimodal>
% Multiobjective PSO using ring topology and special crowding distance

%------------------------------- Reference --------------------------------
% C. Yue, B. Qu, and J. Liang. A multiobjective particle swarm optimizer 
% using ring topology for solving multimodal multiobjective problems. IEEE 
% Transactions on Evolutionary Computation, 2018, 22(5): 805-817.
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
            %% Initialize parameters
            n_PBA = 5;          % Maximum size of PBA
            n_NBA = 3*n_PBA;	% Maximum size of NBA

            %% Generate random population
            mv   = 0.5*(Problem.upper-Problem.lower);
            Vmin = -mv;
            Vmax = mv;
            ParticleDec = Problem.lower+(Problem.upper-Problem.lower).*rand(Problem.N,Problem.D);
            ParticleVel = Vmin+2.*Vmax.*rand(Problem.N,Problem.D);
            Population  = Problem.Evaluation(ParticleDec,ParticleVel);

            %% Initialize personal best archive PBA and Neighborhood best archive NBA
            PBA = cell(1,Problem.N);
            NBA = cell(1,Problem.N);
            for i = 1:Problem.N
                PBA{i} = Population(i);
                NBA{i} = Population(i);
            end

            %% Optimization
            while Algorithm.NotTerminated(Population)
                NBA = UpdateNBA(NBA,n_NBA,PBA);
                Population = Operator(Problem,Population,PBA,NBA);
                PBA = UpdatePBA(Population,PBA,n_PBA);
                if Problem.FE >= Problem.maxFE
                    tempNBA = [];
                    for i = 1:Problem.N
                        tempNBA = [tempNBA,NBA{i}];
                    end
                    [tempNBA,FrontNo,~] = non_domination_scd_sort(tempNBA,Problem.N);
                    Algorithm.NotTerminated(tempNBA(FrontNo==1));
                end
            end
        end
    end
end
```

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Particle,PBA,NBA)
% Particle swarm optimization in MO_Ring_PSO_SCD

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
    [N,D]       = size(ParticleDec);
    PbestDec    = zeros(N,D);
    NbestDec    = zeros(N,D);
    for i = 1:N
        PbestDec(i,:) = PBA{i}(1).dec;
        NbestDec(i,:) = NBA{i}(1).dec;        
    end
    ParticleVel = Particle.adds(zeros(N,D));

    %% Particle swarm optimization
    W  = repmat(0.7298,N,D);
    r1 = rand(N,D);
    r2 = rand(N,D);
    C1 = repmat(2.05,N,D);
    C2 = repmat(2.05,N,D);
    OffVel = W.*ParticleVel + C1.*r1.*(PbestDec-ParticleDec) + C2.*r2.*(NbestDec-ParticleDec);
    delta  = repmat((Problem.upper-Problem.lower)/2,N,1);
    OffVel = max(min(OffVel,delta),-delta);
    OffDec = ParticleDec + OffVel;
    Lower  = repmat(Problem.lower,N,1);
    Upper  = repmat(Problem.upper,N,1);
    temp = OffDec<Lower;
    OffDec(temp) = Lower(temp)+0.25*(Upper(temp)-Lower(temp)).*rand(size(OffDec(temp)));
    temp = OffDec>Upper;
    OffDec(temp) = Upper(temp)-0.25*(Upper(temp)-Lower(temp)).*rand(size(OffDec(temp))); 
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `UpdateNBA.m`
```matlab
function NBA = UpdateNBA(NBA,n_NBA,PBA)
% Update NBA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for i = 1:length(NBA)
        if i == 1
            tempNBA = [PBA{end},PBA{1},PBA{2}];
        elseif i == length(NBA)
            tempNBA = [PBA{end-1},PBA{end},PBA{1}];
        else
            tempNBA = [PBA{i-1},PBA{i},PBA{i+1}];
        end
        tempNBA   = [tempNBA,NBA{i}];
        [tempNBA,FrontNo,SpCrowdDis] = non_domination_scd_sort(tempNBA,n_NBA);
        [~,index] = sortrows([FrontNo;-SpCrowdDis]');
        NBA{i}    = tempNBA(index);
    end
end
```

### `UpdatePBA.m`
```matlab
function PBA = UpdatePBA(Population,PBA,n_PBA)
% Update PBA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for i = 1 : length(PBA)
        tempPBA   = [PBA{i},Population(i)];
        [tempPBA,FrontNo,SpCrowdDis] = non_domination_scd_sort(tempPBA,n_PBA);
        [~,index] = sortrows([FrontNo;-SpCrowdDis]');
        PBA{i}    = tempPBA(index);
    end
end
```

### `non_domination_scd_sort.m`
```matlab
function [Population,FrontNo,SpCrowdDis] = non_domination_scd_sort(Population,N)
% Sort the population according to non-dominated relationship and special crowding 

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    %% Non-dominated sorting
    N = min(N,length(Population));
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the special crowding distance of each solution
    SpCrowdDis_Obj = ModifiedCrowdingDistance(Population.objs,FrontNo);
    SpCrowdDis_Dec = ModifiedCrowdingDistance(Population.decs,FrontNo);
    SpCrowdDis     = max(SpCrowdDis_Obj,SpCrowdDis_Dec);
    for i = 1 : MaxFNo
        Front   = find(FrontNo==i);
        Avg_Obj = mean(SpCrowdDis_Obj(Front));
        Avg_Dec = mean(SpCrowdDis_Dec(Front));
        replace = SpCrowdDis_Obj(Front)<=Avg_Obj & SpCrowdDis_Dec(Front)<=Avg_Dec;
        SpCrowdDis(Front(replace)) = min(SpCrowdDis_Obj(Front(replace)),SpCrowdDis_Dec(Front(replace)));
    end
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(SpCrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Limit the size of Population 
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    SpCrowdDis = SpCrowdDis(Next);
end

function CrowdDis = ModifiedCrowdingDistance(PopObj,FrontNo)
    [N,M]    = size(PopObj);
    CrowdDis = zeros(1,N);
    Fronts   = setdiff(unique(FrontNo),inf);
    for f = 1 : length(Fronts)
        Front = find(FrontNo==Fronts(f));
        Fmax  = max(PopObj(Front,:),[],1);
        Fmin  = min(PopObj(Front,:),[],1);
        for i = 1 : M
            [~,Rank] = sortrows(PopObj(Front,i));
            CrowdDis(Front(Rank(1))) = CrowdDis(Front(Rank(1))) + 1;
            for j = 2 : length(Front)-1
                if Fmax(i) == Fmin(i)
                    CrowdDis(Front(Rank(j))) = CrowdDis(Front(Rank(j)))+1;
                else
                    CrowdDis(Front(Rank(j))) = CrowdDis(Front(Rank(j)))+(PopObj(Front(Rank(j+1)),i)-PopObj(Front(Rank(j-1)),i))/(Fmax(i)-Fmin(i));
                end
            end
        end
    end
end
```
