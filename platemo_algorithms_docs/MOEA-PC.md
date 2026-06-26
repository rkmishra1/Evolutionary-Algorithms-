# MOEA-PC

**Tags**: <2015> <multi> <real/integer>

## Description
Multiobjective evolutionary algorithm based on polar coordinates

## Reference
R. Denysiuk, L. Costa, I. E. Santo, and J. C. Matos. MOEA/PC: Multiobjective evolutionary algorithm based on polar coordinates. Proceedings of the International Conference on Evolutionary Multi- Criterion Optimization, 2015, 141-155.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,Offspring,Z,nDivs)
% The environmental selection of MOEA/PC

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    % for offspring, determine grid index and polar coordinates
    [off.idx, off.theta, off.rho] = PolarCoordinatesGrid(Offspring.obj, Z, nDivs);
    
    % for population member corresponding to grid index of offspring,
    % determine true grid index and polar coordinates
    [pop.idx, pop.theta, pop.rho] = PolarCoordinatesGrid(Population(off.idx).obj, Z, nDivs);
    
    % update grid
    if (off.idx == pop.idx) 
        % pop member and off reside in the same grid     
        % offspring replaces pop member if it has a smaller radius
        if off.rho < pop.rho
            Population(off.idx) = Offspring; 
        end    
        return;
        
    elseif any(Offspring.obj < Population(off.idx).obj)
        % pop memebr and offspring are swapped because
        % pop member resides in other grid 
        tmp = Population(off.idx);
        Population(off.idx) = Offspring;
        Offspring  = tmp;
        Population = EnvironmentalSelection(Population,Offspring,Z,nDivs);
        
    end

end
```

### `GridStructure.m`
```matlab
function [G, nGrids, nDivs] = GridStructure(n, m)
%   GridStructure - Generate a grid deviding the first octant by polar
%   coordinates. Each element of grid is given by (m-1) integer numbers. 
%   The structure is similar to that given by polar coordinates.
%
%   [G, nGrids, nDivs] = GridStructure(n, m)
%
%   Input:
%   n - population size
%   m - number of objectives
%
%   Output:
%   G - grid structure
%   nGrids - number of grids (can be larger than provided population size)
%   nDivs - number of divisions in each right angle
%
%   Example:
%   [G, nGrids, nDivs] = GridStructure(225,3)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    % compute number of divisions
    nDivs = ceil( power(n, 1/(m-1)) );
    
    % compute number of grids
    nGrids = power(nDivs, m-1);
    
    % initialize grids
    G = zeros(nGrids, m-1);
    
    % compute grids
    D = m - 2;
    for j = 0 : D
        tmp      = repmat(1:nDivs, power(nDivs, j), 1);
        G(:,j+1) = repmat( tmp(:), power(nDivs, D-j), 1);
    end

end
```

### `MOEAPC.m`
```matlab
classdef MOEAPC < ALGORITHM
% <2015> <multi> <real/integer>
% Multiobjective evolutionary algorithm based on polar coordinates
% delta --- 0.8 --- The probability of choosing parents locally
% T     ---  20 --- Neighborhood size 

%------------------------------- Reference --------------------------------
% R. Denysiuk, L. Costa, I. E. Santo, and J. C. Matos. MOEA/PC:
% Multiobjective evolutionary algorithm based on polar coordinates.
% Proceedings of the International Conference on Evolutionary Multi-
% Criterion Optimization, 2015, 141-155.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [delta,T] = Algorithm.ParameterSet(0.8,20);

            %% Generate grid
            [G,nGrids,nDivs] = GridStructure(Problem.N,Problem.M);  
            Problem.N = nGrids;

            %% Detect the neighbours of each grid
            B = pdist2(G,G);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Select solution at random
                i = randi(nGrids);

                % Choose the parents
                if rand < delta
                    P = B(i,randperm(size(B,2),3));
                else
                    P = randperm(nGrids,3);
                end  

                % Generate an offspring
                Offspring = OperatorDE(Problem,Population(P(1)),Population(P(2)),Population(P(3)));

                % Update the ideal point
                Z = min(Z,Offspring.obj);

                % Update the solutions in P by Environmental Selection
                Population = EnvironmentalSelection(Population,Offspring,Z,nDivs);
            end
        end
    end
end
```

### `PolarCoordinatesGrid.m`
```matlab
function [idx, theta, rho] = PolarCoordinatesGrid(y, r, nDivs)
%   PolarCoordinatesGrid - Calculate index of grid cell and 
%   polar coordinates for a given objective vector
%
%   [idx, theta, rho] = PolarCoordinatesGrid(y, r, nDivs)
%
%   Input:
%   y - individual's objectives
%   r - reference point
%   nDivs - number of divisions in each right angle
%
%   Output:
%   idx - index of grid cell where individual resides
%   theta - polar angles
%   rho - radius

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Roman Denysiuk

    %% calculate polar coordinates
    
    % determine dimensionality
    m = numel(r);
    
    % calculate radius
    rho = norm(y-r);
    
    % calculate angles
    theta = zeros(m-1,1);
    
    for i = 0 : m-2
        
        u = (y(m-i) - r(m-i))/rho;
        
        for j = 0 : i-1
            u = u/cos(theta(j+1));
        end
        
        u = min( max(u, -1), 1); % ensure feasibility
        
        theta(i+1) = asin(u);
        
        theta(i+1) = min( max(theta(i+1), eps), pi/2-eps);
    end
    
    %% calculate grid index based on polar coordinates
    
    % calculate grid coordinates (integer values)
    G = ceil(2*nDivs*theta/pi);
    
    % calculate index of pop member using grid coordinates
    % formula: idx=G(1)*ndiv^0+(G(2)-1)*ndiv^1+(G(3)-1)*ndiv^2 ...
    
    idx = G(1);
    for i = 2 : numel(G)
        idx = idx + (G(i)-1)*power(nDivs, i-1);
    end

end
```
