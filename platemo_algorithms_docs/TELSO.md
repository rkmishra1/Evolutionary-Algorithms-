# TELSO

**Tags**: <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>

## Description
Two-layer encoding learning swarm optimizer

## Reference
S. Qi, R. Wang, T. Zhang, X. Yang, R. Sun, and L. Wang. A two-layer encoding learning swarm optimizer based on frequent itemsets for sparse large-scale multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2024, 11(6): 1342-1357.

## Source Code

### `EnvironmentalSelection.m`
```matlab
function [Population,Mask2] = EnvironmentalSelection(Population,V,theta,Mask)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Sheng Qi (email: 2745679162@qq.com)
  
    Mask   = repmat(Mask, 2, 1);
    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    NV     = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the degree of violation of each solution
    CV = sum(max(0,Population.cons),2);
    
    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current1 = find(associate==i & CV==0);
        current2 = find(associate==i & CV~=0);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD = (1+M*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
            % Select the one with the minimum APD value
            [~,best] = min(APD);
            Next(i)  = current1(best);
        elseif ~isempty(current2)
            % Select the one with the minimum CV value
            [~,best] = min(CV(current2));
            Next(i)  = current2(best);
        end
    end
    % Population for next generation
    Population = Population(Next(Next~=0));
    j = length(Next(Next~=0));
    a = Next(Next~=0);
    for i = 1 : j
        b = a(i);
        Mask2(i, :) = Mask(b, :);
    end
end
```

### `Initial_EnvironmentalSelection.m`
```matlab
function [Population,Mask] = Initial_EnvironmentalSelection(Population,Dec,Mask,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Sheng Qi (email: 2745679162@qq.com)

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `Operator.m`
```matlab
function [Offspring,Mask] = Operator(Population,Mask,Problem)
% The learning swarm optimizer of TELSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Sheng Qi (email: 2745679162@qq.com)

    %% Parameter setting
    PopulationDec = Population.decs;
    [N,D]         = size(PopulationDec);
	PopulationVel = Population.adds(zeros(N,D));
    OffDec        = [];
    OffVel        = []; 
    
    %% Refactoring the real variable    
    for i = 1 : (N-2)
        rand_indices       = randi([i+1,N],1,2);
        selected_particles = Population(rand_indices); 
        r1       = repmat(rand(1,1),1,D);
        r2       = repmat(rand(1,1),1,D);
        r3       = repmat(rand(1,1),1,D);
        p_OffVel = r1.*PopulationVel(i)+r2.*(selected_particles(1).dec-Population(i).dec)+r3.*(selected_particles(2).dec-Population(i).dec);
        p_OffDec = Population(i).dec+p_OffVel;
        OffDec   = [OffDec;p_OffDec];
        OffVel   = [OffVel;p_OffVel];

        p1_mask = Mask(rand_indices(1), :);
        p2_mask = Mask(rand_indices(2), :);
        if rand < 0.5
            same_elems = (p1_mask == p2_mask & p2_mask ==1);
        else
            same_elems = (p1_mask == p2_mask & p2_mask ==0);
        end
        same_cols = find(same_elems);
        D_number  = numel(same_cols);  
        keep      = floor(D_number * Problem.FE/Problem.maxFE);  
        same_cols = same_cols(1:keep);  
        if same_cols ~= 0
            for j = same_cols
                Mask(i,j) = Mask(rand_indices(1),j); 
            end
        end
    end

	%% Add the two winners
    OffDec    = [OffDec;Population(N-1).dec;Population(N).dec];
    OffVel    = [OffVel;Population(N-1).add;Population(N).add];
    Offspring = Problem.Evaluation(OffDec.*Mask,OffVel);
end
```

### `TELSO.m`
```matlab
classdef TELSO < ALGORITHM
% <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Two-layer encoding learning swarm optimizer

%------------------------------- Reference --------------------------------
% S. Qi, R. Wang, T. Zhang, X. Yang, R. Sun, and L. Wang. A two-layer
% encoding learning swarm optimizer based on frequent itemsets for sparse
% large-scale multi-objective optimization. IEEE/CAA Journal of Automatica
% Sinica, 2024, 11(6): 1342-1357.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Sheng Qi (email: 2745679162@qq.com)

    methods
        function main(Algorithm,Problem)
            %% Initialization reference vector
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            
            %% Generate random population
            % Calculate the fitness of each decision variable
            TDec    = [];
            TMask   = [];
            TempPop = [];
            DF      = zeros(1,Problem.D);	% The fitness of decision variables
            REAL    = all(Problem.encoding==1);
            for i = 1 : 1+4*REAL
                if REAL
                    Dec = unifrnd(repmat(Problem.lower,Problem.D,1),repmat(Problem.upper,Problem.D,1));
                else
                    Dec = ones(Problem.D,Problem.D);
                end
                Mask       = eye(Problem.D);
                Population = Problem.Evaluation(Dec.*Mask);
                TDec       = [TDec;Dec];
                TMask      = [TMask;Mask];
                TempPop    = [TempPop,Population];
                DF         = DF + NDSort([Population.objs,Population.cons],inf);
            end
            
            %% Generate initial population
            if REAL
                Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            else
                Dec = ones(Problem.N,Problem.D);
            end
            Mask = zeros(Problem.N,Problem.D);
            for i = 1 : Problem.N
                Mask(i,TournamentSelection(2,ceil(rand*Problem.D),DF)) = 1;
            end
            Population = Problem.Evaluation(Dec.*Mask);
            [Population,Mask] = Initial_EnvironmentalSelection([Population,TempPop],[Dec;TDec],[Mask;TMask],Problem.N);
            [Population,Mask] = EnvironmentalSelection(Population,V,(Problem.FE/Problem.maxFE)^2,Mask);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                Fitness    = CalFitness(Population.objs);
                [~,index]  = sort(Fitness);
                Population = Population(index);
                Mask       = Mask(index,:);
                [Offspring,Mask]  = Operator(Population,Mask,Problem);
                [Population,Mask] = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2,Mask);
            end
        end
    end
end

function Fitness = CalFitness(PopObj)
% Calculate the fitness by shift-based density
    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end
```
