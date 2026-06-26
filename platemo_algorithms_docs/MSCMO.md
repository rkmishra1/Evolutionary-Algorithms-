# MSCMO

**Tags**: <2021> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Multi-stage constrained multi-objective evolutionary algorithm

## Reference
H. Ma, H. Wei, Y. Tian, R. Cheng, and X. Zhang. A multi-stage evolutionary algorithm for multi-objective optimization with complex constraints. Information Sciences, 2021, 560: 68-91.

## Source Code

### `Archive.m`
```matlab
function Population = Archive(varargin)
% Update the archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select feasible solutions
    [Population,N,index,count] = deal(varargin{1:4});
    if count == 0
        CV = zeros(1,size(Population,2));
    else
        CV = Population.cons;
        CV = CV(:,index(1:count));
    end
    if nargin == 4
        fIndex     = all(CV <= 0,2);
        Population = Population(fIndex);
        if isempty(Population)
            return;
        else
            %% Non-dominated sorting
            [FrontNo,~] = NDSort(Population.objs,1);
            Next = (FrontNo == 1);
            Population = Population(Next);
            Next = true(size(Population,2),1);
            if sum(Next) > N
                Del  = Truncation(Population(Next).objs,sum(Next)-N);
                Temp = find(Next);
                Next(Temp(Del)) = false;
                Population      = Population(Next);      
            end
        end
    else                  
        feasible_number = varargin{5};
        if feasible_number == 0
            return;
        end
        feasible_solutions = Population(1:feasible_number);
        remain_solutions   = Population(feasible_number+1:end);
        [W,~] = UniformPoint(N,size(Population.objs,2));
        itr   = 1;
        while size(feasible_solutions,2) < N
            for i = 1 : size(W,1)
                if size(feasible_solutions,2) == N
                    break;
                end
                [~,Region1] = max(1-pdist2(feasible_solutions.objs,W,'cosine'),[],2);
                [~,Region2] = max(1-pdist2(remain_solutions.objs,W,'cosine'),[],2);
                region1     = find(Region1==i);
                region2     = find(Region2==i);
                if length(region1)<itr && ~isempty(region2) 
                    feasible_solutions = [feasible_solutions remain_solutions(region2(1))];
                    remain_solutions(region2(1)) = [];
                end
            end
            itr = itr + 1;
        end
        Population = feasible_solutions;
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
             if constraint_handing==0
                z=[PopObj CV];
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
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `Constraint_priority.m`
```matlab
function [priority,Feasible_rate] = Constraint_priority(Population)
% Determine constraint-handling priority

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
    [~,temp] = sort(Feasible_rate);
    priority = temp;          
end
```

### `Convertion.m`
```matlab
function outcome = Convertion(Average,G,g,last_gen,change_threshold)
% Conversion condition for enter next stage

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if (G-g > last_gen) && (max(abs(Average(G,:)-Average(G-last_gen,:)))<=change_threshold)
        outcome = true;
    else
        outcome = false;
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
    Population     = Population(rank);
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

### `MSCMO.m`
```matlab
classdef MSCMO < ALGORITHM
% <2021> <multi> <real/integer/label/binary/permutation> <constrained>
% Multi-stage constrained multi-objective evolutionary algorithm
% type --- 1 --- Type of operator (1. GA 2. DE)

%------------------------------- Reference --------------------------------
% H. Ma, H. Wei, Y. Tian, R. Cheng, and X. Zhang. A multi-stage
% evolutionary algorithm for multi-objective optimization with complex
% constraints. Information Sciences, 2021, 560: 68-91.
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
            Population         = Problem.Initialization();
            gen                = 0;
            last_gen           = 100;
            change_threshold   = 1e-2;
            change_rate        = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            priority           = [];
            current_cons       = 0;
            flag               = 0;
            constraint_handing = 0;
            Fitness            = CalFitness(Population.objs,Population.cons,priority,current_cons,constraint_handing);
            archive            = Population;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Determine constraint-handling priority
                if flag == 0
                    change_rate = Normalization(Population,change_rate,ceil(Problem.FE/Problem.N));
                     if Convertion(change_rate,ceil(Problem.FE/Problem.N),gen,last_gen,change_threshold)
                        flag = 1;
                        [priority,Feasible_rate] = Constraint_priority(Population);
                        Population = Problem.Initialization();
                     end
                else
                    % Judge whether to enter next stage
                    if current_cons == 0
                       CV = Population.cons;
                       CV = CV(:,priority(1));
                       if length(find(CV>0))/Problem.N > 0
                          current_cons = current_cons + 1;
                          gen = ceil(Problem.FE/Problem.N) + 1;
                       end
                    elseif current_cons <= size(Population.cons,2)
                         if constraint_handing == 0
                             change_rate = Normalization(Population,change_rate,ceil(Problem.FE/Problem.N));
                             if Convertion(change_rate,ceil(Problem.FE/Problem.N),gen,last_gen,change_threshold)
                                 if current_cons<size(Population.cons,2) && Feasible_rate(priority(current_cons+1))~=1
                                    current_cons = current_cons+1;
                                 elseif current_cons<size(Population.cons,2) && Feasible_rate(priority(current_cons+1))==1
                                    current_cons = size(Population.cons,2);
                                 elseif current_cons == size(Population.cons,2)
                                    constraint_handing = 1;
                                 end
                                 if size(archive,2) == Problem.N
                                    Population = archive;
                                    Fitness    = CalFitness(Population.objs,Population.cons,priority,current_cons,constraint_handing);
                                 else
                                    archive    = Archive([archive,Population],Problem.N,priority,current_cons,size(archive,2));
                                    Population = archive;
                                    Fitness    = CalFitness(Population.objs,Population.cons,priority,current_cons,constraint_handing);
                                 end
                                 gen = ceil(Problem.FE/Problem.N) + 1;
                             end
                         end  
                    end     
                end
                % Reproduction
                if type == 1
                    MatingPool = TournamentSelection(2,Problem.N,Fitness);
                    Offspring  = OperatorGA(Problem,Population(MatingPool));
                elseif type == 2
                    MatingPool1 = TournamentSelection(2,Problem.N,Fitness);
                    MatingPool2 = TournamentSelection(2,Problem.N,Fitness);
                    Offspring   = OperatorDE(Problem,Population,Population(MatingPool1),Population(MatingPool2));
                end
                %  Update external archive
                if flag==1 && constraint_handing~=1
                    archive = Archive([Offspring,archive],Problem.N,priority,current_cons);
                end
                % Environmental selection 
                [Population,Fitness] = EnvironmentalSelection([Population,Offspring],Problem.N,priority,current_cons,constraint_handing);
            end
        end
    end
end
```

### `Normalization.m`
```matlab
function outcome = Normalization(Population,Average,G)
% Normalization

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    fmax = max(Population.objs,[],1);
    fmin = min(Population.objs,[],1);
    Average(G,:) = mean((Population.objs-fmin)./(fmax-fmin));
    outcome      = Average;
end
```
