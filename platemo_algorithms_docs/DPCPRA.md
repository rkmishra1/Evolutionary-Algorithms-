# DPCPRA

**Tags**: <2024> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Dual-population with dynamic constraint processing and resource allocating

## Reference
K. Qiao, Z. Chen, B. Qu, K. Yu, C. Yue, K. Chen, and J. Liang. A dual- population evolutionary algorithm based on dynamic constraint processing and resources allocation for constrained multi-objective optimization problems. Expert Systems With Applications, 2024, 238: 121707.

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

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

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
            FrontNo    = NDSort(Population.objs,1);
            Next       = (FrontNo == 1);
            Population = Population(Next);
            Next       = true(size(Population,2),1);
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
        itr = 1;
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

### `CalFitness_pop1.m`
```matlab
function Fitness = CalFitness_pop1(PopObj,PopCon)
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
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
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
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `CalFitness_pop2.m`
```matlab
function Fitness = CalFitness_pop2(PopObj,PopCon,priority,current_cons,constraint_handing)
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

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

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

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    if (G-g > last_gen) && (max(abs(Average(G,:)-Average(G-last_gen,:)))<=change_threshold)
        outcome = true;
    else
        outcome = false;
    end
end
```

### `DPCPRA.m`
```matlab
classdef DPCPRA < ALGORITHM
% <2024> <multi> <real/integer/label/binary/permutation> <constrained>
% Dual-population with dynamic constraint processing and resource allocating

%------------------------------- Reference --------------------------------
% K. Qiao, Z. Chen, B. Qu, K. Yu, C. Yue, K. Chen, and J. Liang. A dual-
% population evolutionary algorithm based on dynamic constraint processing
% and resources allocation for constrained multi-objective optimization
% problems. Expert Systems With Applications, 2024, 238: 121707.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Fitness1    = CalFitness_pop1(Population1.objs,Population1.cons);
            current_cons       = 0;
            gen                = 0;
            last_gen           = 100;
            change_threshold   = 1e-2;
            change_rate        = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            priority           = [];
            flag               = 0;
            constraint_handing = 0;
            archive            = Population2;
            Fitness2           = CalFitness_pop2(Population2.objs,Population2.cons,priority,current_cons,constraint_handing);
            success_rate1      = 0.5;

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                if flag == 0
                    change_rate = Normalization(Population2,change_rate,ceil(Problem.FE/Problem.N));
                    if Convertion(change_rate,ceil(Problem.FE/Problem.N),gen,last_gen,change_threshold)
                        flag = 1;
                        [priority,evaluatedasible_rate] = Constraint_priority(Population2);
                        Population2 = Problem.Initialization();
                    end
                else
                    % Judge whether to enter next stage
                    if current_cons == 0
                        CV = Population2.cons;
                        CV = CV(:,priority(1));
                        if length(find(CV>0))/Problem.N > 0
                            current_cons = current_cons + 1;
                            gen = ceil(Problem.FE/Problem.N) + 1;
                        end
                    elseif current_cons <= size(Population2.cons,2)
                        if constraint_handing == 0
                            change_rate = Normalization(Population2,change_rate,ceil(Problem.FE/Problem.N));
                            if Convertion(change_rate,ceil(Problem.FE/Problem.N),gen,last_gen,change_threshold)
                                if current_cons<size(Population2.cons,2) && evaluatedasible_rate(priority(current_cons+1))~=1
                                    current_cons = current_cons+1;
                                elseif current_cons<size(Population2.cons,2) && evaluatedasible_rate(priority(current_cons+1))==1
                                    current_cons = size(Population2.cons,2);
                                elseif current_cons == size(Population2.cons,2)
                                    constraint_handing = 1;
                                end
                                if size(archive,2) == Problem.N
                                    Population2 = archive;
                                    Fitness2    = CalFitness_pop2(Population2.objs,Population2.cons,priority,current_cons,constraint_handing);
                                else
                                    archive     = Archive([archive,Population2],Problem.N,priority,current_cons,size(archive,2));
                                    Population2 = archive;
                                    Fitness2    = CalFitness_pop2(Population2.objs,Population2.cons,priority,current_cons,constraint_handing);
                                end
                                gen = ceil(Problem.FE/Problem.N) + 1;
                            end
                        end
                    end
                end
                % Optimization
                if flag == 0
                    MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                    MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                    Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                    Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                else
                    if mod(ceil(Problem.N*success_rate1),2)~=0
                        MatingPool1 = TournamentSelection(2,ceil(Problem.N*success_rate1)+1,Fitness1);
                    else
                        MatingPool1 = TournamentSelection(2,ceil(Problem.N*success_rate1),Fitness1);
                    end

                    Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool1));
                    MatingPool2 = TournamentSelection(2,Problem.N-2*length(Offspring1),Fitness2);
                    Offspring2  = OperatorGAhalf(Problem,Population2(MatingPool2));
                end
                %  Update external archive
                if flag==1 && constraint_handing~=1
                    archive = Archive([Offspring2,archive],Problem.N,priority,current_cons);
                end
                [Population1,Fitness1,success_rate1] = EnvironmentalSelection_pop1([Population1,Offspring1,Offspring2],Problem.N,true,length(Offspring1));
                [Population2,Fitness2,success_rate2] = EnvironmentalSelection_pop2([Population2,Offspring2,Offspring1],Problem.N,priority,current_cons,constraint_handing,length(Offspring2));
                success_rate1 = success_rate1/(success_rate1+success_rate2);
            end
        end
    end
end
```

### `EnvironmentalSelection_pop1.m`
```matlab
function [Population,Fitness,success_rate1] = EnvironmentalSelection_pop1(Population,N,isOrigin,p1_length)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    if isOrigin
        Fitness = CalFitness_pop1(Population.objs,Population.cons);
    else
        Fitness = CalFitness(Population.objs);
    end

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
    success_rate1 = sum(Next(N+1:N+p1_length))/p1_length;
    if success_rate1 < 0.1
       success_rate1 = 0.1;
    elseif success_rate1 > 0.9
       success_rate1 = 0.9; 
    end
    % Population for next generation
    Population1 = Population(Next);
    Fitness     = Fitness(Next);
    % Sort the population
    [Fitness,rank] = sort(Fitness);
    Population     = Population1(rank);
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

### `EnvironmentalSelection_pop2.m`
```matlab
function [Population,Fitness,success_rate2] = EnvironmentalSelection_pop2(Population,N,index,count,c,p2_length)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    %% Calculate the fitness of each solution
	Fitness = CalFitness_pop2(Population.objs,Population.cons,index,count,c);
     
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
    %success_rate2 = sum(Next(1:N))/N;
     success_rate2 = sum(Next(N+1:N+ p2_length))/ p2_length;
     if success_rate2 < 0.1
       success_rate2 = 0.1;
    elseif success_rate2 > 0.9
       success_rate2 = 0.9; 
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

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    fmax = max(Population.objs,[],1);
    fmin = min(Population.objs,[],1);
    Average(G,:) = mean((Population.objs-fmin)./(fmax-fmin));
    outcome      = Average;
end
```
