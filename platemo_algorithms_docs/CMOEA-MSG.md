# CMOEA-MSG

**Tags**: <2024> <multi> <real/integer> <constrained>

## Description
Multi-stage constrained multi-objective evolutionary algorithm

## Reference
Y. Tian, J. Chen, and X. Zhang. An optimizer combining evolutionary computation and gradient descent for constrained multi-objective optimization. Journal of Computer Applications (Chinese), 2024, 44(05): 1386-1392.

## Source Code

### `CMOEAMSG.m`
```matlab
classdef CMOEAMSG < ALGORITHM
% <2024> <multi> <real/integer> <constrained>
% Multi-stage constrained multi-objective evolutionary algorithm
% type --- 1 --- Type of operator (1. GA 2. DE)

%------------------------------- Reference --------------------------------
% Y. Tian, J. Chen, and X. Zhang. An optimizer combining evolutionary
% computation and gradient descent for constrained multi-objective
% optimization. Journal of Computer Applications (Chinese), 2024, 44(05):
% 1386-1392.
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
            type = Algorithm.ParameterSet(1);

            %% Initialization
            priority     = [];
            current_cons = 0;
            Population1  = Problem.Initialization(100);
            Population2  = Problem.Initialization(100);
            Population   = [Population1,Population2];
            Ar = Population1;       
            i  = 0;
            ii = 0;
            ab = 10;
            gen                = 0;
            change_threshold   = 0.1;
            change_rate        = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            constraint_handing = 0;
            [priority,~]       = Constraint_priority(Population,priority,current_cons);
            last_gen           = max(30,floor(200/(length(priority)+1)));
            
            %% Optimization
            Population1 = EnvironmentalSelection1(Population1,100,priority,current_cons,0);
            for h = 1 : length(Population1)
                while true
                    X    = Population1(h);
                    Grad = Gradient(Problem,X,current_cons,priority,0);
                    Grad = sign(Grad);
                    while true
                        step = abs(Population1(h).dec-Population1(randi(100,1)).dec);
                        i = i + 1;
                        H = X;
                        X = X.dec - step.*Grad*0.1  ;
                        X = Problem.Evaluation(X);
                        [X,Bool] = Compare([X,H],priority,current_cons,constraint_handing);
                        if i ==ab || Bool == false
                            break;
                        end
                    end
                    if i == 10 || Bool == false
                        i = 0;
                        break;
                    end
                end
                Population1 = [Population1,X];
            end
            [Population1,~] = EnvironmentalSelection(Population1,100,priority,current_cons,0);
            while Algorithm.NotTerminated(Ar)
                ii = ii + 1;
                if current_cons <= size(Population1.cons,2)
                    if constraint_handing == 0
                        change_rate(ii,:) = mean(Ar.objs);
                        if (ii-gen > last_gen) && (min(abs(change_rate(ii,:)-mean(change_rate(ii-30:ii-1,:))))<=change_threshold)
                            if current_cons < size(Population1.cons,2)
                                [priority,~] = Constraint_priority(Population1,priority,current_cons);
                                current_cons = current_cons+1;
                            elseif current_cons == size(Population1.cons,2)
                                constraint_handing = 1;
                            end
                            gen = ii + 1;
                        end
                    end
                end
                if constraint_handing == 0
                    if  mod(ii,1) == 0
                        Fitness1 = CalFitness(Population1.objs,Population1.cons,priority,current_cons,constraint_handing);
                        [Xx,oc]  = Selection(Population1,current_cons,Fitness1,priority);
                        if current_cons == 0
                            m = 3;
                        else
                            m = 1;
                        end
                        for h = 1 : m
                            X    = Xx(h);
                            Grad = Gradient(Problem,X,current_cons,priority,oc);
                            Grad = sign(Grad);
                            while true
                                i    = i + 1;
                                step = abs(Population1(h).dec-Population1(randi(100,1)).dec);
                                H    = X;
                                X    = X.dec - step.*Grad*0.1  ;
                                X    = Problem.Evaluation(X);
                                [X,Bool] = Compare([X,H],priority,current_cons,constraint_handing);
                                if i > ab || Bool == false
                                    i = 0;
                                    break;
                                end
                            end
                            Population1 = [Population1,X];
                        end
                    end
                end
                Fitness = CalFitness(Population1.objs,Population1.cons,priority,length(priority),0);
                if type == 1
                    MatingPool1 = TournamentSelection(2,100,Fitness);
                    Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                else
                    a = length(Population1);
                    MatingPool11 = TournamentSelection(2,a,Fitness);
                    MatingPool12 = TournamentSelection(2,a,Fitness);
                    Offspring1   = OperatorDE(Problem,Population1,Population1(MatingPool11),Population1(MatingPool12));
                end
                Fitness5 = CalFitness(Ar.objs,Ar.cons,priority,length(priority),1);
                if type == 1
                    MatingPool4 = TournamentSelection(2,Problem.N,Fitness5);
                    Offspring4  = OperatorGA(Problem,Ar(MatingPool4));
                    if constraint_handing == 0
                        [Population1,~] = EnvironmentalSelection([Population1,Offspring1],100,priority,current_cons,0);
                    else
                        [Population1,~] = EnvironmentalSelection1([Population1,Offspring1],100,priority,current_cons,0);
                    end
                    Ar = EnvironmentalSelection1([Ar,Offspring4,Offspring1],Problem.N,priority,length(priority),1);
                else
                    MatingPool41 = TournamentSelection(2,100,Fitness5);
                    MatingPool42 = TournamentSelection(2,100,Fitness5);
                    Offspring4   = OperatorDE(Problem,Ar,Ar(MatingPool41),Ar(MatingPool42));
                    if current_cons == 0
                        [Population1,~] = EnvironmentalSelection([Population1,Offspring1],100,priority,current_cons,0);
                    else
                        [Population1,~] = EnvironmentalSelection1([Population1,Offspring1],100,priority,current_cons,0);
                    end
                    Ar = EnvironmentalSelection1([Ar,Offspring4,Offspring1],Problem.N,priority,length(priority),1);
                end
            end
        end
    end
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(PopObj,PopCon,priority,current_cons,constraint_handing)
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
    if current_cons == 0
        CV = zeros(N,1);
    else
        CV = PopCon;
        CV = CV(:,priority);
        CV = CV(:,1:current_cons);
        CV = sum(max(0,CV),2);
    end
    
    %% Detect the dominance relation between each two solutions
    Dominate = false(N);  
    for i = 1 : N-1       
        for j = i+1 : N           
             if constraint_handing == 0
                z = [PopObj,CV];
                k = any(z(i,:)<z(j,:)) - any(z(i,:)>z(j,:));
                if k == 1
                   Dominate(i,j) = true;
                elseif k == -1
                   Dominate(j,i) = true;
                end
            else
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
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    Fitness = R + D';
end
```

### `Compare.m`
```matlab
function [X,Bool] = Compare(Pop,priority,current_cons,constraint_handing)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if current_cons == 0
        CV = zeros(2,1);
    else
        CV = Pop.cons;
        CV = CV(:,priority(1:current_cons));
        CV = sum(max(0,CV),2);
    end
    if constraint_handing == 0
        if CV(1) < CV(2)
            Bool = true;
            X    = Pop(1);
        elseif CV(1) > CV(2)
            Bool = false;
            X    = Pop(2);
        else
            k = any(Pop(1).objs<Pop(2).objs) - any(Pop(1).objs>Pop(2).objs);
            if k == 1 
                Bool = true;
                X = Pop(1);
            elseif k == -1 || k == 0
                Bool = false;
                X    = Pop(2);
            end
        end
    end
end
```

### `Constraint_priority.m`
```matlab
function [priority,Feasible_rate] = Constraint_priority(Population,priority,current_cons)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Feasible_rate = zeros(1,size(Population.cons,2));
    for j = 1 : size(Population.cons,2)
        CV = Population.cons;
        CV = CV(:,j);
        Feasible_rate(1,j) = length(find(CV<=0))/size(Population,2);
    end
    if isempty(priority)
        [~,temp] = sort(Feasible_rate);
        priority = temp;
    elseif current_cons+1 < size(Population.cons,2)
        Feasible_rate = Feasible_rate(:,priority);
        [~,temp] = sort(Feasible_rate(current_cons+1:end));
        priority(current_cons+1:end) = priority(temp +current_cons)   ;
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection(Population,N,index,count,c)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    a        = 0;
    Fitness1 = [];
    num      = 1;
    Pop      = [];
    PopObj   = Population.objs;
	Fitness  = CalFitness(Population.objs,Population.cons,index,count,c);
    M        = size(Population(1).objs,2);
    L        = size(Population,2);
    rank     = inf(1,L);
    
    z          = min(PopObj,[],1);
    [W,~]      = UniformPoint(100,M);
    [~,Region] = min(pdist2(PopObj-z,W,'cosine'),[],2);  
    for i = 1 : size(W,1)
        index = find(Region==i);
         if ~isempty(index)
            [~,index1] = sort(Fitness(index));
            index = index(index1);
            for j = 1 : size(index)
                rank(index(j)) = j;
            end
        end
    end
    while true
        [~,index3] = find(rank==num);
        a = a + length(index3);
        if a <= N
            Pop = [Pop,Population(index3)];
            Fitness1 = [Fitness1,Fitness(index3)];
            num = num + 1;
        else
            break;
        end
    end
    if length(Pop) == N
        Population = Pop;
        Fitness    = Fitness1;
    else
        [~,index3] = find(rank==num);
        Pop1       = Population(index3);
        Fitness2   = Fitness(index3);
        Del        = Truncation(Pop1.objs,a-N);
        Population = [Pop,Pop1(~Del)];
        Fitness    = [Fitness1,Fitness2(~Del)];
    end
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

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

### `EnvironmentalSelection1.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection1(Population,N,index,count,c)
% Environmental selection 1

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
	Fitness = CalFitness(Population.objs,Population.cons,index,count,c);
    
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
    Population = Population(rank);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

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

### `Gradient.m`
```matlab
function Gradient = Gradient(Problem,Pop,current_cons,priority,oc)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if oc == 1
        % Gradient of a constraint
        [~,ConGrad] = Problem.CalGrad(Pop.dec);
        Gradient    = ConGrad(priority(current_cons),:);
    else
        % Gradient of all objectives
        ObjGrad  = Problem.CalGrad(Pop.dec);
        Gradient = sum(ObjGrad,1);
        conflict = any(ObjGrad<0,1) & any(ObjGrad>0,1);
        Gradient(conflict) = 0;
    end
end
```

### `Selection.m`
```matlab
function [X,oc] = Selection(Population,current_cons,Fitness,priority)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    cons = Population.cons;
    cons = cons(:,priority);
    if current_cons == 0
        oc = 0;
        [~,temp] = sort(Fitness);
        X = [Population(temp(floor(end/10))),Population(randi(100,1,2))];
    elseif current_cons == 1
        temp = find(cons(:,current_cons)>0);
        if isempty(temp)
            oc = 0;
            [~,temp] = sort(Fitness);
            X = Population(temp(1));
        else
            oc = 1;
            [~,I] = sort(Fitness(temp));
            X = Population(temp(I(end)));
        end
    else
        temp  = find(max(cons(:,1:current_cons-1),[],2)<0);
        index = find(cons(temp,current_cons)>0);
        if isempty(index)
            oc = 0;
            [~,temp] = sort(Fitness);
            X = Population(temp(1));
        else
            oc    = 1;
            temp  = temp(index);
            [~,I] = sort(Fitness(temp));
            X     = Population(temp(I(end)));
        end
    end
end
```
