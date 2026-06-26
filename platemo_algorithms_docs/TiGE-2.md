# TiGE-2

**Tags**: <2020> <many> <real/integer/label/binary/permutation> <constrained/none>

## Description
Tri-Goal Evolution Framework for CMaOPs

## Reference
Y. Zhou, Z. Min, J. Wang, Z. Zhang, and J.Zhang. Tri-goal evolution framework for constrained many-objective optimization. IEEE Transactions on Systems Man and Cybernetics Systems, 2020, 50(8): 3086-3099.

## Source Code

### `Calculate_fcv.m`
```matlab
function fcv = Calculate_fcv(Population)
% Calculate normalized  constraints violation(CV) measuring feasibility

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CV_Original = Population.cons;
    CV_Original(CV_Original<=0) = 0;
    CV = CV_Original./max(CV_Original);
    CV(:,isnan(CV(1,:))) = 0;
    fcv = sum(max(0,CV),2)./size(CV_Original,2);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,fitness] = EnvironmentalSelection(Population,PopObj,OffObj,N)
% The environmental selection of TiGE_1

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [FrontNo,~] = NDSort([PopObj;OffObj],N);
    fcv         = Calculate_fcv(Population);    % CV of hybrid population
    fitness     = FrontNo' + fcv./(fcv+1);
    [~,index]   = sort(fitness);
    Population  = Population(index(1:N));
    fitness     = fitness(index(1:N));
end
```

### `Estimation.m`
```matlab
function [fpr,fcd] = Estimation(PopObj,r)
% Estimate the proximity and crowding degree of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    
    %% Proximity estimation
    fmax   = repmat(max(PopObj,[],1),N,1);
    fmin   = repmat(min(PopObj,[],1),N,1);
    PopObj = (PopObj-fmin)./(fmax-fmin);
    fpr    = sum(PopObj,2);
    
    %% Crowding degree estimation
    d     = pdist2(PopObj,PopObj);
    d(logical(eye(length(d)))) = inf;
    fprm  = repmat(fpr,1,N);
    case1 = d<r & fprm<=fprm';
    case2 = d<r & fprm>fprm';
    sh        = zeros(N);
    sh(case1) = (0.5*(1-d(case1)/r)).^2;
    sh(case2) = (1.5*(1-d(case2)/r)).^2;
    fcd       = sqrt(sum(sh,2));
end
```

### `TiGE2.m`
```matlab
classdef TiGE2 < ALGORITHM
% <2020> <many> <real/integer/label/binary/permutation> <constrained/none>
% Tri-Goal Evolution Framework for CMaOPs

%------------------------------- Reference --------------------------------
% Y. Zhou, Z. Min, J. Wang, Z. Zhang, and J.Zhang. Tri-goal evolution
% framework for constrained many-objective optimization. IEEE Transactions
% on Systems Man and Cybernetics Systems, 2020, 50(8): 3086-3099.
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
            [Epsilon0,row] = Algorithm.ParameterSet(0.05,1.01);
            
            %% Generate random population
            Population = Problem.Initialization();
            [fpr,fcd]  = Estimation(Population.objs,1/Problem.N^(1/Problem.M));
            fcv        = Calculate_fcv(Population); 
            Epsilon    = Epsilon0;
            PopObj_1   = [fpr,fcd]; 
            [fm,~]     = NDSort(PopObj_1,Problem.N);
            PopObj     = [fm' + Epsilon * fcv,fcv];
            [frank,~]  = NDSort(PopObj,Problem.N);
            fitness    = frank' + fcv./(fcv+1);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,fitness);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [fpr,fcd]  = Estimation(Offspring.objs,1/Problem.N^(1/Problem.M));
                fcv = Calculate_fcv(Offspring); 
                OffObj_1   = [fpr,fcd]; 
                [fm,~]     = NDSort(OffObj_1,Problem.N);
                OffObj     = [fm' + Epsilon * fcv,fcv];
                [Population,fitness] = EnvironmentalSelection([Population,Offspring],PopObj,OffObj,Problem.N);
                [fpr,fcd]  = Estimation(Population.objs,1/Problem.N^(1/Problem.M));
                fcv        = Calculate_fcv(Population);
                PopObj_1   = [fpr,fcd]; 
                [fm,~]     = NDSort(PopObj_1,Problem.N);
                PopObj     = [fm' + Epsilon * fcv,fcv];
                Epsilon    = row * Epsilon;
            end
        end
    end
end
```
