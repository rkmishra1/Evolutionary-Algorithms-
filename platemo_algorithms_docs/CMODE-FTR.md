# CMODE-FTR

**Tags**: <2023> <multi> <real/integer> <constrained>

## Description
Constrained multiobjective differential evolution based on the fusion of two rankings

## Reference
Z. Zeng, X. Zhang, and Z. Hong. A constrained multiobjective differential evolution algorithm based on the fusion of two rankings. Information Sciences, 2023, 647, 119572.

## Source Code

### `CMODEFTR.m`
```matlab
classdef CMODEFTR < ALGORITHM
% <2023> <multi> <real/integer> <constrained>
% Constrained multiobjective differential evolution based on the fusion of two rankings

%------------------------------- Reference --------------------------------
% Z. Zeng, X. Zhang, and Z. Hong. A constrained multiobjective differential
% evolution algorithm based on the fusion of two rankings. Information
% Sciences, 2023, 647, 119572.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiqiang Zeng (email: zhiqiang.zeng@outlook.com)

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization();
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                Offspring  = DEgenerator2(Population,Problem);
                t          = Problem.FE/Problem.N;
                MaxGen     = Problem.maxFE/Problem.N;
                a          = 0.5*(1-cos((1-t/MaxGen)*pi));
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,a,Problem);
            end
        end
    end
end
```

### `DEgenerator2.m`
```matlab
function [Offspring] = DEgenerator2(Population,Problem)
%Search operator based on differential evolution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiqiang Zeng (email: zhiqiang.zeng@outlook.com)

    FrontNo = NDSort(Population.objs,Population.cons,1);
    [N,D]   = size(Population(1).decs);
    trial   = zeros(1*Problem.N,D);
    
    for i = 1 : Problem.N
    
        %Randomly select one individual from CDP ranking level 1 as the best
        index_No1  = find(FrontNo==1);
        r          = floor(rand*length(index_No1))+1;
        best_index = index_No1(r);
    
        %Randomly select three individuals who are not equal to each other
        indexset    = 1 : Problem.N;
        indexset(i) = [];
        r1  = floor(rand*(Problem.N-1))+1;
        xr1 = indexset(r1);
        indexset(r1) = [];
        r2  = floor(rand*(Problem.N-2))+1;
        xr2 = indexset(r2);
        indexset(r2) = [];
        r3  = floor(rand*(Problem.N-3))+1;
        xr3 = indexset(r3);
    
        if rand <= 0.5
            CR = 0.1;
            if rand < 0.5
                F_pool = [0.6,0.8,1.0];
                F      = F_pool(randi(3));
                v      = Population(xr1).decs+rand*(Population(best_index).decs-Population(xr1).decs)+F*(Population(xr2).decs-Population(xr3).decs);%Mutation operation 1
            else
                F_pool = [0.1,0.8,1.0];
                F      = F_pool(randi(3));
                v      = Population(xr1).decs+F*(Population(i).decs-Population(xr1).decs)+F*(Population(xr2).decs-Population(xr3).decs);%Mutation operation 2
            end
    
            %Boundary Repair
            Lower = repmat(Problem.lower,N,1);
            Upper = repmat(Problem.upper,N,1);
            v     = min(max(v,Lower),Upper);
    
            %Crossover operation
            Site   = rand(N,D) < CR;
            j_rand = floor(rand * D) + 1;
            Site(1, j_rand) = 1;
            Site_  = 1-Site;
            trial(i, :) = Site.*v+Site_.*Population(i).decs;
    
        else
            if rand < 0.5
                F_pool = [0.6,0.8,1.0];
                F      = F_pool(randi(3));
                v      = Population(i).decs+rand*(Population(xr1).decs-Population(i).decs)+F*(Population(xr2).decs-Population(xr3).decs);%Mutation operation 3
            else
                F_pool = [0.1,0.8,1.0];
                F      = F_pool(randi(3));
                v      = Population(i).decs+F*(Population(best_index).decs-Population(i).decs)+F*(Population(xr1).decs-Population(xr2).decs);%Mutation operation 4
            end
            %Boundary Repair
            Lower = repmat(Problem.lower,N,1);
            Upper = repmat(Problem.upper,N,1);
            trial(i, :) = min(max(v,Lower),Upper);
        end
    end
    Offspring = trial;
    Offspring = Problem.Evaluation(Offspring);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population] = EnvironmentalSelection(Population,N,a,Problem)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Zhiqiang Zeng (email: zhiqiang.zeng@outlook.com)

    [N1,~]       = size(Population.objs);
    [FrontNo1,~] = NDSort(Population.objs,Population.cons,inf);
    CrowdDis1    = CrowdingDistance(Population.objs,FrontNo1);
    [~,r1]       = sortrows([FrontNo1',-CrowdDis1']);
    Rc(r1)       = 1 : N1;   % Obtain the ranking of each individual based on CDP
    
    FrontNo2  = NDSort(Population.objs,0,inf);
    CrowdDis2 = CrowdingDistance(Population.objs,FrontNo2);
    [~,r2]    = sortrows([FrontNo2',-CrowdDis2']);
    Rp(r2)    = 1 : N1;   % Obtain the ranking of each individual based on non-dominated sorting
    
    
    pro_l = 1-length(find(sum(max(0,Population.cons),2)>0))/N1; % Calculate feasibility rate
    b     = 1.0/((Problem.FE/Problem.maxFE)^2+1)-0.5;
    r     = a*(1-b)+pro_l*b;
    R_sum = (1-r)*Rc+r*Rp; % To fuse two rankings.
    
    [~,Rank]   = sort(R_sum);
    Population = Population(Rank(1:N));
end
```
