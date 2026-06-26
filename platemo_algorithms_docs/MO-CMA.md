# MO-CMA

**Tags**: <2007> <multi> <real/integer>

## Description
Multi-objective covariance matrix adaptation evolution strategy

## Reference
C. Igel, N. Hansen, and S. Roth. Covariance matrix adaptation for multi- objective optimization. Evolutionary computation, 2007, 15(1): 1-28.

## Source Code

### `MOCMA.m`
```matlab
classdef MOCMA < ALGORITHM
% <2007> <multi> <real/integer>
% Multi-objective covariance matrix adaptation evolution strategy

%------------------------------- Reference --------------------------------
% C. Igel, N. Hansen, and S. Roth. Covariance matrix adaptation for multi-
% objective optimization. Evolutionary computation, 2007, 15(1): 1-28.
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
            %% Generate the initial individuals in CMA-ES
            Population = Problem.Initialization();
            ptarget    = 1/5.5;
            a          = struct('x',num2cell(Population.decs,2)','psucc',ptarget,'sigma',0.5,'pc',0,'C',eye(Problem.D),'Individual',num2cell(Population));

            %% Optimization
            while Algorithm.NotTerminated([a.Individual])
                % Generate new individuals
                for k = 1 : Problem.N
                    a1(k)            = a(k);
                    a1(k).x          = mvnrnd(a(k).x,a(k).sigma^2*a(k).C,1);
                    a1(k).Individual = Problem.Evaluation(a1(k).x);
                end

                % Update the fitness of each individual
                Q           = [a,a1];
                Population  = [Q.Individual];
                % Penalized fitness for handling box constraints
                PopObj      = Population.objs + repmat(1e-6*sum((cat(1,Q.x)-Population.decs).^2,2),1,Problem.M);
                % Calculate the fitness of each individual
                FrontNo     = NDSort(PopObj,inf);
                CrowdDis    = CrowdingDistance(PopObj,FrontNo);
                [~,rank]    = sortrows([FrontNo;-CrowdDis]');
                [~,fitness] = sort(rank);

                % Update the CMA models
                for k = 1 : Problem.N
                    a(k)  = updateStepSize(a(k),fitness(Problem.N+k)<fitness(k),ptarget);
                    a1(k) = updateStepSize(a1(k),fitness(Problem.N+k)<fitness(k),ptarget);
                    a1(k) = updateCovariance(a1(k),(a1(k).x-a(k).x)/a(k).sigma);
                end

                % Individuals for next generation
                Q = [a,a1];
                a = Q(rank(1:Problem.N));
            end
        end
    end
end
```

### `updateCovariance.m`
```matlab
function a = updateCovariance(a,xstep)
% Update the covariance of CMA model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % Constant of learning rate for evolution path
    cc = 2/(length(a.x)+2);
    % Constant of learning rate for covariance matrix
    ccov = 2/(length(a.x)^2+6);
    if a.psucc < 0.44
        % Update the evolution path
        a.pc = (1-cc)*a.pc + sqrt(cc*(2-cc))*xstep;
        % Update the covariance matrix
        a.C = (1-ccov)*a.C + ccov*a.pc'*a.pc;
    else
        % Update the evolution path
        a.pc = (1-cc)*a.pc;
        % Update the covariance matrix
        a.C = (1-ccov)*a.C + ccov*(a.pc'*a.pc+cc*(2-cc)*a.C);
    end
end
```

### `updateStepSize.m`
```matlab
function a = updateStepSize(a,psucc,ptarget)
% Update the step size of CMA model based on the success rate

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % Success rate averaging parameter
    cp = ptarget/(2+ptarget);
    % Step size damping
    d = 1 + length(a.x)/2;
    % Update the averaged success rate
    a.psucc = (1-cp)*a.psucc + cp*psucc;
    % Update the global step size
    a.sigma = a.sigma*exp((a.psucc-ptarget)/d/(1-ptarget));
end
```
