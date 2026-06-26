# CA-MOEA

**Tags**: <2019> <multi> <real/integer/label/binary/permutation>

## Description
Clustering based adaptive multi-objective evolutionary algorithm

## Reference
Y. Hua, Y. Jin, and K. Hao. A clustering-based adaptive evolutionary algorithm for multiobjective optimization with irregular Pareto fronts. IEEE Transactions on Cybernetics, 2019, 49(7): 2758-2770.

## Source Code

### `CAMOEA.m`
```matlab
classdef CAMOEA < ALGORITHM
% <2019> <multi> <real/integer/label/binary/permutation>
% Clustering based adaptive multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Hua, Y. Jin, and K. Hao. A clustering-based adaptive evolutionary
% algorithm for multiobjective optimization with irregular Pareto fronts.
% IEEE Transactions on Cybernetics, 2019, 49(7): 2758-2770.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yicun Hua

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Generate offspring randomly
                MatingPool = randperm(Problem.N);
                Offspring  = OperatorGA(Problem,Population(MatingPool));

                % Elitism strategy
                UniPop = [Population,Offspring];
                PopObj = UniPop.objs;
                [FrontNo,MaxFNo] = NDSort(PopObj,Problem.N);

                % The number of individuals to be selected in the last
                % non-dominated front
                K = Problem.N - sum(FrontNo<MaxFNo);

                if K ~= 0
                    % Normalization
                    pareto_population = find(FrontNo<MaxFNo);
                    last_population   = find(FrontNo == MaxFNo);
                    Zmin = min(PopObj(FrontNo == MaxFNo,:));
                    Zmax = max(PopObj(FrontNo == MaxFNo,:));
                    S = sum(FrontNo == MaxFNo);
                    MaxFnorm = (PopObj(last_population,:)-repmat(Zmin,S,1))./repmat(Zmax-Zmin,S,1);

                    % Clustering-based reference points generation
                    [Ref] = Reference_Generation( MaxFnorm,Problem.M,K);

                    % Clustering-based environmental selection
                    [reference_population] = Reference_Point_Selection(MaxFnorm,last_population,Ref,K,Problem.M);
                else
                    pareto_population    = find(FrontNo<=MaxFNo);
                    reference_population = [];
                end
                Population = UniPop([pareto_population,reference_population]);
            end
        end
    end
end
```

### `Reference_Generation.m`
```matlab
function [Ref] = Reference_Generation(MaxFnorm, M, K)
% Clustering-based reference points generation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yicun Hua

    % Calculate boundary individuals: extreme
    [~,maxobj] = max(MaxFnorm,[],1);
    [~,minobj] = min(MaxFnorm,[],1);
    ex = unique([maxobj,minobj]);
    extreme = unique(MaxFnorm(ex,:),'rows');
    
    E = size(extreme,1);    
    Nc = K-E;
    
    if Nc>0
        % Clustering
        T = clusterdata(MaxFnorm,'maxclust',Nc,'distance','euclidean','linkage','ward');
        
        % Calculate cluster centers
        for i = 1:Nc

            p = find(T == i);
            pn = length(p);
 
            Ref(i,1:M) = sum(MaxFnorm(p,:),1)/pn;

            Ref(i,M+1) = i;   
            
        end
        
        Ref = unique(Ref,'rows');
        rpr = size(Ref,1);
        
        % If cluster centers are not enough, 
        % randomly select some individuals as reference points
        if rpr < Nc
            Ncrp = Nc-rpr;
            Ref(rpr+1:Nc,:) = MaxFnorm(randperm(size(MaxFnorm,1),Ncrp),:);
        end
       
        Ref(:,M+1) = [];
        
        % Use boundary individuals as reference points
        for er = 1:E
            Ref(Nc+er,1:M) = extreme(er,1:M);
        end
    else
        % Use boundary individuals as reference points
        for er = 1:K
            Ref(er,:) = extreme(er,:);
        end
    end
end
```

### `Reference_Point_Selection.m`
```matlab
function  [reference_population]=Reference_Point_Selection(MaxFnorm,last_population,Ref,K,M)
% Clustering-based environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yicun Hua

    last_Num = size(MaxFnorm,1);
    Ref_Num = size(Ref,1);
    dis = pdist2(MaxFnorm,Ref);
    
    for i = 1:last_Num
        [~,dmin_p] = min(dis(i,:),[],2);
        
        % Individual i assigned to Ref(dmin_p,:)
        MaxFnorm(i,(M+1)) = dmin_p; 
        
        % Distance between individual i and Ref(dmin_p,:)
        MaxFnorm(i,(M+2)) = dis(i,dmin_p);
    end
    
    % Individuals are labeled 3(to distinguish from 1 and 2)
    MaxFnorm(:,(M+3)) = 3;
    for i = 1:Ref_Num
        if sum(MaxFnorm(:,(M+1)) == i) ~= 0
            a = find(MaxFnorm(:,(M+1)) == i);
            a = a';
            [~,b] = sort(MaxFnorm(a,(M+2)));
            
            % Individuals closest to reference points are labeled 1
            MaxFnorm(a(b(1)),(M+3)) = 1;   
            
            % Individuals in crowd area are labeled 2
            if sum(MaxFnorm(:,(M+1)) == i) > 3
                a(b(1)) = [];
                MaxFnorm(a,(M+3)) = 2; 
            end
        end
    end
    
    t1 = sum(MaxFnorm(:,(M+3)) == 1);
    t2 = sum(MaxFnorm(:,(M+3)) == 2); 
    
    if t1 <= K
        % If individuals labeled 1 are not enough, we choose some 2, then 3
        d = find(MaxFnorm(:,(M+3))==1);
        d = d';
        reference_population(1:t1) = d;
        if t2 >= (K-t1)
            % If individuals labeled 2 are enough
            d = find(MaxFnorm(:,(M+3))==2);
            d = d';
            y = randperm(length(d));
            reference_population((t1+1):K) = d(y(1:(K-t1)));
        else
            % If individuals labeled 2 are not enough, we choose some 3
            d = find(MaxFnorm(:,(M+3)) == 2);
            d = d';
            reference_population((t1+1):(t1+t2)) = d;
            d = find(MaxFnorm(:,(M+3)) == 3);
            d = d';
            y = randperm(length(d));
            reference_population((t1+t2+1):K) = d(y(1:(K-t1-t2)));
        end
    else
        % If individuals labeled 1 are enough
        d = find(MaxFnorm(:,(M+3)) == 1);
        d = d';
        y = randperm(length(d));
        reference_population(1:K) = d(y(1:K));
    end
    reference_population = last_population(reference_population);
end
```
