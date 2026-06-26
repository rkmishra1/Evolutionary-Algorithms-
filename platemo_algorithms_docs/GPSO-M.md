# GPSO-M

**Tags**: <2012> <multi/many> <real> <large/none> <constrained/none>

## Description
Gradient based particle swarm optimization algorithm (for multi-objective optimization)

## Reference
M. M. Noel. A new gradient based particle swarm optimization algorithm for accurate computation of global minimum. Applied Soft Computing, 2012, 12: 353-359.

## Source Code

### `GPSOM.m`
```matlab
classdef GPSOM < ALGORITHM
% <2012> <multi/many> <real> <large/none> <constrained/none>
% Gradient based particle swarm optimization algorithm (for multi-objective optimization)
% popsize --- 20 --- Population size of single run

%------------------------------- Reference --------------------------------
% M. M. Noel. A new gradient based particle swarm optimization algorithm
% for accurate computation of global minimum. Applied Soft Computing, 2012,
% 12: 353-359.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            popsize = Algorithm.ParameterSet(20);

            %% Generate random population 
            SubPops = cell(1,Problem.N);
            Pbests  = cell(1,Problem.N);
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            SubPops{1}    = Problem.Initialization(popsize);
            Pbests(1)     = SubPops(1);
            FrontNo       = NDSort(SubPops{1}.objs,SubPops{1}.cons,1);
            index = FrontNo == 1;
            FrontNoPops = SubPops{1}(index);
            Gbest(1)    = LocalSearch(Problem,FrontNoPops(1),W(1,:));
            for i = 2 : Problem.N
                SubPops{i}  =  Problem.Initialization(popsize);
                Pbests(i)   = SubPops(i);
                [FrontNo,~] = NDSort(SubPops{i}.objs,SubPops{i}.cons,1);
                index = FrontNo == 1;
                FrontNoPops = SubPops{1}(index);
                Gbest(i)    = LocalSearch(Problem,FrontNoPops(1),W(i,:));
            end
            
            %% Optimization
            while Algorithm.NotTerminated(Gbest)
                for i = 1 : Problem.N
                    for j = 1 : popsize
                        SubPops{i}(j) = OperatorPSO(Problem,SubPops{i}(j),Pbests{i}(j),Gbest(i));
                        Pbests{i}(j)  = UpdatePbest(Pbests{i}(j),SubPops{i}(j));
                    end
                    FrontNo     = NDSort(SubPops{i}.objs,SubPops{i}.cons,1);
                    index       = FrontNo==1;
                    FrontNoPops = SubPops{i}(index);
                    Gbest(i)    = LocalSearch(Problem,FrontNoPops(1),W(i,:));
                end
            end
        end
	end
end
```

### `LocalSearch.m`
```matlab
function optimo = LocalSearch(Problem,pos,w)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    MaxIter = 5;
    Tol     = 1e-3;
    step    = 1;
    k       = 1;
    error   = 10;
    while error>Tol && k<MaxIter
        grad(1,:)    = FiniteDifference(pos,w,Problem);
        offspringdec = pos.dec - step*grad(1,:);
        offspringdec = min(max(offspringdec,Problem.lower),Problem.upper);
        offspring    = Problem.Evaluation(offspringdec);
        grad(2,:)    = FiniteDifference(offspring,w,Problem);
        step         = abs((offspring.dec-pos.dec)*(grad(2,:)-grad(1,:))')/norm(grad(2,:)-grad(1,:))^2;
        error = norm(offspring.dec-pos.dec);
        pos   = offspring;
        k     = k + 1;
    end
    optimo = pos;
end

function df = FiniteDifference(X,W,Problem)
    feasible = X.con <= 0;
    if ~all(feasible)
        [~,df] = Problem.CalGrad(X.dec);
        df(feasible,:) = 0;
        df = sum(df',2);
    else
        df = Problem.CalGrad(X.dec)';
        df = df*W';
    end
end
```

### `UpdatePbest.m`
```matlab
function Pbest = UpdatePbest(Pbest,Population)
% Update the local best position of each particle

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    temp = Pbest.objs - Population.objs;
    Dominate = any(temp<0,2) - any(temp>0,2);
    Pbest(Dominate==-1) = Population(Dominate==-1);
    temp = rand(length(Dominate),1);
    Pbest(Dominate==0 & temp<0.5) = Population(Dominate==0 & temp<0.5);
end
```
