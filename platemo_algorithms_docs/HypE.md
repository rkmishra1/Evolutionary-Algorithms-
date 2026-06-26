# HypE

**Tags**: <2011> <multi/many> <real/integer/label/binary/permutation>

## Description
Hypervolume estimation algorithm

## Reference
J. Bader and E. Zitzler. HypE: An algorithm for fast hypervolume-based many-objective optimization. Evolutionary Computation, 2011, 19(1): 45-76.

## Source Code

### `CalHV.m`
```matlab
function F = CalHV(points,bounds,k,nSample)
% Calculate the hypervolume-based fitness value of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/

    [N,M] = size(points);
    if M > 2
        % Use the estimated method for three or more objectives
        alpha = zeros(1,N); 
        for i = 1 : k 
            alpha(i) = prod((k-[1:i-1])./(N-[1:i-1]))./i; 
        end
        Fmin = min(points,[],1);
        S    = unifrnd(repmat(Fmin,nSample,1),repmat(bounds,nSample,1));
        PdS  = false(N,nSample);
        dS   = zeros(1,nSample);
        for i = 1 : N
            x        = sum(repmat(points(i,:),nSample,1)-S<=0,2) == M;
            PdS(i,x) = true;
            dS(x)    = dS(x) + 1;
        end
        F = zeros(1,N);
        for i = 1 : N
            F(i) = sum(alpha(dS(PdS(i,:))));
        end
        F = F.*prod(bounds-Fmin)/nSample;
    else
        % Use the accurate method for two objectives
        pvec  = 1:size(points,1);
        alpha = zeros(1,k);
        for i = 1 : k 
            j = 1:i-1; 
            alpha(i) = prod((k-j)./(N-j))./i;
        end
        F = hypesub(N,points,M,bounds,pvec,alpha,k);
    end
end

function h = hypesub(l,A,M,bounds,pvec,alpha,k)
% The recursive function for the accurate method

    h     = zeros(1,l); 
    [S,i] = sortrows(A,M); 
    pvec  = pvec(i); 
    for i = 1 : size(S,1) 
        if i < size(S,1) 
            extrusion = S(i+1,M) - S(i,M); 
        else
            extrusion = bounds(M) - S(i,M);
        end
        if M == 1
            if i > k
                break; 
            end
            if alpha >= 0
                h(pvec(1:i)) = h(pvec(1:i)) + extrusion*alpha(i); 
            end
        elseif extrusion > 0
            h = h + extrusion*hypesub(l,S(1:i,:),M-1,bounds,pvec(1:i),alpha,k); 
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N,RefPoint,nSample)
% The environmental selection of HypE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = true(1,length(Last));
    while sum(Choose) > N-sum(Next)
        drawnow('limitrate');
        Remain  = find(Choose);
        F       = CalHV(Population(Last(Remain)).objs,RefPoint,sum(Choose)-N+sum(Next),nSample);
        [~,del] = min(F);
        Choose(Remain(del)) = false;
    end
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end
```

### `HypE.m`
```matlab
classdef HypE < ALGORITHM
% <2011> <multi/many> <real/integer/label/binary/permutation>
% Hypervolume estimation algorithm
% nSample --- 10000 --- Number of sampled points for HV estimation

%------------------------------- Reference --------------------------------
% J. Bader and E. Zitzler. HypE: An algorithm for fast hypervolume-based
% many-objective optimization. Evolutionary Computation, 2011, 19(1):
% 45-76.
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
            nSample = Algorithm.ParameterSet(10000);

            %% Generate random population
            Population = Problem.Initialization();
            % Reference point for hypervolume calculation
            RefPoint = zeros(1,Problem.M) + max(Population.objs)*1.2;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,-CalHV(Population.objs,RefPoint,Problem.N,nSample));
                Offspring  = OperatorGA(Problem,Population(MatingPool));    
                Population = EnvironmentalSelection([Population,Offspring],Problem.N,RefPoint,nSample);
            end
        end
    end
end
```
