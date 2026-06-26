# MSCEA

**Tags**: <2023> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Multi-stage constrained multi-objective evolutionary algorithm

## Reference
Y. Zhang, Y. Tian, H. Jiang, X. Zhang, and Y. Jin. Design and analysis of helper-problem-assisted evolutionary algorithm for constrained multiobjective optimization. Information Sciences, 2023, 648: 119547.

## Source Code

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,CV)
% Calculate the fitness of each solution

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    if nargin < 2
        CV = zeros(size(PopObj,1),1);
    end
    
   %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
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
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
   %% Calculate D(i)
    Distance = pdist2(real(PopObj),real(PopObj),'cosine');
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
   %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalSDE.m`
```matlab
function SDE = CalSDE(PopObj)
% Calculate the value of SDE of each solution

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N      = size(PopObj,1);   
    Zmin   = min(PopObj,[],1);
    Zmax   = max(PopObj);
    PopObj = (PopObj-repmat(Zmin,N,1))./(repmat(Zmax,N,1)-repmat(Zmin,N,1));
    SDE    = zeros(1,N);
    for i = 1 : N
        SPopuObj = PopObj;
        Temp     = repmat(PopObj(i,:),N,1);
        Shifted  = PopObj < Temp;
        SPopuObj(Shifted) = Temp(Shifted);                                    
        Distance  = pdist2(real(PopObj(i,:)),real(SPopuObj));
        [~,index] = sort(Distance,2);
        Dk        = Distance(index(floor(sqrt(N))+1)); % Dk denotes the distance of solution i and its floor(sqrt(N)+1)-th nearest neighbour
        SDE(i)    = 1./(Dk+2);
    end
end
```

### `EnvironmentalSelection1.m`
```matlab
function Population = EnvironmentalSelection1(Population,N,epsn)
% The environmental selection of MSCEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
   
    [~, nCon] = size(Population.cons);
    PopCon    = max(0,Population.cons);
    if sum(sum(PopCon<=epsn, 2)==nCon) > N       
        tmp        = sum(PopCon<=epsn, 2)==nCon;
        Population = Population(1:end, tmp);
        CV         = sum(max(0,Population.cons),2);
        Fitness    = CalFitness(Population.objs,CV);
        Next       = Fitness < 1;
        if sum(Next) < N
            [~,Rank] = sort(Fitness);
            Next(Rank(1:N)) = true;
        elseif sum(Next) > N
            Del  = Truncation(Population(Next).objs,sum(Next)-N);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
    else       
        CV        = sum(max(0,Population.cons),2);
        [~, rank] = sort(CV);
        Next      = rank(1:N);       
    end   
    Population = Population(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

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

### `EnvironmentalSelection2.m`
```matlab
function [Population, Fitness] = EnvironmentalSelection2(Population,N,epsn)
% The environmental selection of MSCEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    [~, nCon] = size(Population.cons);
    PopCon    = max(0,Population.cons);
    if sum(sum(PopCon<=epsn, 2)==nCon) > N       
        tmp        = sum(PopCon<=epsn, 2)==nCon;
        Population = Population(1:end, tmp);
        CV         = sum(max(0,Population.cons),2);        
        Fitness    = CalFitness([Population.objs,CV]);        
    else       
        CV      = sum(max(0,Population.cons),2);
        Fitness = CalFitness([CalSDE(Population.objs)',CV]);        
    end
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

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

### `MSCEA.m`
```matlab
classdef MSCEA < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <constrained>
% Multi-stage constrained multi-objective evolutionary algorithm
% cp --- 5 --- Decrease trend of the dynamic constraint boundary

%------------------------------- Reference --------------------------------
% Y. Zhang, Y. Tian, H. Jiang, X. Zhang, and Y. Jin. Design and analysis of
% helper-problem-assisted evolutionary algorithm for constrained 
% multiobjective optimization. Information Sciences, 2023, 648: 119547.
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
            cp = Algorithm.ParameterSet(5);

            %% Generate the random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();    

            %% Calculate the initial dynamic constraint boundary
            [~, nCon]               = size(Population1.cons);
            [initialE1, ~]          = max(max(0,Population1.cons), [], 1);
            [initialE2, ~]          = max(max(0,Population2.cons), [], 1);
            initialE1(initialE1==0) = 1;            
            initialE2(initialE2==0) = 1; 
            epsn1                   = initialE1;
            epsn2                   = initialE2;    
            CV2                     = sum(max(0,Population2.cons),2);            
            Fitness2                = CalFitness([CalSDE(Population2.objs)',CV2]);
            MaxCV2                  = zeros(ceil(Problem.maxFE/Problem.N),1);
            MaxCV2(1)               = max(CV2);
            ASC2                    = 0;
            arch                    = archive([Population1,Population2],Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population1)                
                PopCon1   = max(0,Population1.cons);
                PopCon2   = max(0,Population2.cons);
                if sum(sum(PopCon1<=epsn1,2)==nCon) == length(Population1)
                    epsn1 = ReduceBoundary(initialE1,ceil(Problem.FE/Problem.N),ceil(Problem.maxFE/Problem.N)-1,cp);
                end
                CV2 = sum(PopCon2,2);
                MaxCV2(ceil(Problem.FE/Problem.N)) = max(CV2);
                if MaxCV2(ceil(Problem.FE/Problem.N))-MaxCV2(ceil((Problem.FE-Problem.N)/Problem.N))>0
                    ASC2  = ASC2 + 1;
                    epsn2 = ReduceBoundary(initialE2,ceil(Problem.FE/Problem.N)-ASC2,ceil(Problem.maxFE/Problem.N)-1,cp);
                elseif sum(sum(PopCon2<=epsn2,2)==nCon) == length(Population2)
                    ASC2  = 0;
                    epsn2 = ReduceBoundary(initialE2,ceil(Problem.FE/Problem.N),ceil(Problem.maxFE/Problem.N)-1,cp);
                end                                                        
                MatingPool1 = TournamentSelection(2,Problem.N,sum(max(0,Population1.cons-epsn1),2));
                MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                Population1            = EnvironmentalSelection1([Population1,Offspring1,Offspring2],Problem.N,epsn1);
                [Population2,Fitness2] = EnvironmentalSelection2([Population2,Offspring1,Offspring2],Problem.N,epsn2);                
                % Output the non-dominated and feasible solutions.
                arch = [arch,Population1,Population2];
                [~, Unduplicated] = unique(arch.objs,'rows');
                arch = arch(Unduplicated);
                arch = archive(arch,Problem.N);
                if Problem.FE >= Problem.maxFE
                    Population1 = arch;
                end                        
            end
        end
    end
end
```

### `ReduceBoundary.m`
```matlab
function epsn = ReduceBoundary(eF, k, MaxK, cp)
% The shrink of the dynamic constraint boundary

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
   
    z        = 1e-8;
    Nearzero = 1e-15;
    B        = MaxK./power(log((eF + z)./z), 1.0./cp);
    B(B==0)  = B(B==0) + Nearzero;
    f        = eF.* exp( -(k./B).^cp );
    tmp      = find(abs(f-z) < Nearzero);
    f(tmp)   = f(tmp).*0 + z;
    epsn     = f - z;
    epsn(epsn<=0) = 0;
end
```

### `archive.m`
```matlab
function Population = archive(Population,N)
% Select feasible and non-dominated solutions by SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select feasible solutions
    fIndex     = all(Population.cons <= 0,2);
    Population = Population(fIndex);
    if isempty(Population)
        return
    else
        Fitness = CalFitness(Population.objs);
        Next    = Fitness < 1;
        if sum(Next) > N
            Del  = Truncation(Population(Next).objs,sum(Next)-N);
            Temp = find(Next);
            Next(Temp(Del)) = false;
        end
        Population = Population(Next);
    end
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

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
