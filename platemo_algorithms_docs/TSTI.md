# TSTI

**Tags**: <2022> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Two-stage evolutionary algorithm with three indicators

## Reference
J. Dong, W. Gong, F. Ming, and L. Wang. A two-stage evolutionary algorithm based on three indicators for constrained multi-objective optimization. Expert Systems with Applications, 2022, 195: 116499.

## Source Code

### `Calculate_fcv.m`
```matlab
function fcv = Calculate_fcv(Population)
% calculate normalized  constraints violation(CV) measuring feasibility

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

### `EnvironmentalSelectionStageI.m`
```matlab
function [Population,fitness] = EnvironmentalSelectionStageI(Population,PopObj,OffObj,N)
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
    fcv         = Calculate_fcv(Population);
    fitness     = FrontNo' + fcv./(fcv+1);
    [~,index]   = sort(fitness);
    Population  = Population(index(1:N));
    fitness     = fitness(index(1:N));
end
```

### `EnvironmentalSelectionStageII.m`
```matlab
function [Population,Fitness] = EnvironmentalSelectionStageII(Population,N)
% The environmental selection of Stage-II of the TSTI

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    Fitness = Fit(Population.objs,Population.cons);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
       
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population     = Population(rank);
end


function Del = Truncation(PopObj,K)
    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
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
    d         = pdist2(PopObj,PopObj);
    d(logical(eye(length(d)))) = inf;
    fprm      = repmat(fpr,1,N);
    case1     = d<r & fprm<=fprm';
    case2     = d<r & fprm>fprm';
    sh        = zeros(N);
    sh(case1) = (0.5*(1-d(case1)/r)).^2;
    sh(case2) = (1.5*(1-d(case2)/r)).^2;
    fcd       = sqrt(sum(sh,2));
end
```

### `Fit.m`
```matlab
function Fitness = Fit(PopObj,PopCon)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N  = size(PopObj,1);
    CV = sum(max(0,PopCon),2);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:))-any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)：
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `TSTI.m`
```matlab
classdef TSTI < ALGORITHM
% <2022> <multi> <real/integer/label/binary/permutation> <constrained>
% Two-stage evolutionary algorithm with three indicators
% epsilon --- 0.05 --- parameter for calculating frank
% row     --- 1.01 --- parameter for adjusting epsilon

%------------------------------- Reference --------------------------------
% J. Dong, W. Gong, F. Ming, and L. Wang. A two-stage evolutionary
% algorithm based on three indicators for constrained multi-objective
% optimization. Expert Systems with Applications, 2022, 195: 116499.
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
			[Epsilon0,row] = Algorithm.ParameterSet(0.05,1.01);
			Population     = Problem.Initialization();
			[fpr,fcd]      = Estimation(Population.objs,1/Problem.N^(1/Problem.M));
			fcv            = Calculate_fcv(Population); 
			Epsilon        = Epsilon0;
			PopObj_1       = [fpr,fcd]; 
			[fm,~]         = NDSort(PopObj_1,Problem.N);
			PopObj         = [fm' + Epsilon * fcv,fcv];
			[frank,~]      = NDSort(PopObj,Problem.N);
			fitness        = frank' + fcv./(fcv+1);

			%% Optimization
			while Algorithm.NotTerminated(Population)
				if Problem.FE <= 0.4*Problem.maxFE
					MatingPool = TournamentSelection(2,Problem.N,fitness);
					Offspring  = OperatorGA(Problem,Population(MatingPool));
					[fpr,fcd]  = Estimation(Offspring.objs,1/Problem.N^(1/Problem.M));
					fcv        = Calculate_fcv(Offspring); 
					OffObj_1   = [fpr,fcd]; 
					[fm,~]     = NDSort(OffObj_1,Problem.N);
					OffObj     = [fm' + Epsilon * fcv,fcv];
					[Population,fitness] = EnvironmentalSelectionStageI([Population,Offspring],PopObj,OffObj,Problem.N);
					[fpr,fcd]  = Estimation(Population.objs,1/Problem.N^(1/Problem.M));
					fcv        = Calculate_fcv(Population);
					PopObj_1   = [fpr,fcd]; 
					[fm,~]     = NDSort(PopObj_1,Problem.N);
					PopObj     = [fm' + Epsilon * fcv,fcv];
					Epsilon    = row * Epsilon;
				else
					[Population,fit2] = EnvironmentalSelectionStageII(Population,Problem.N);
					MatingPool        = TournamentSelection(2,Problem.N,fit2);
					Offspring         = OperatorGA(Problem,Population(MatingPool));
					[Population,fit2] = EnvironmentalSelectionStageII([Population,Offspring],Problem.N);
                end	
			end
		end
	end
end
```
