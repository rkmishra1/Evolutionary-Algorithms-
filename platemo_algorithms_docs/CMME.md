# CMME

**Tags**: <2023> <many> <real/integer/label/binary/permutation> <constrained>

## Description
Constrained many-objective evolutionary algorithm with enhanced mating and environmental selections

## Reference
F. Ming, W. Gong, L. Wang, and L. Gao. A constrained many-objective optimization evolutionary algorithm with enhanced mating and environmental selections. IEEE Transactions on Cybernetics, 2023, 53(8): 4934-4946.

## Source Code

### `CMME.m`
```matlab
classdef CMME < ALGORITHM
% <2023> <many> <real/integer/label/binary/permutation> <constrained>
% Constrained many-objective evolutionary algorithm with enhanced mating and environmental selections

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, L. Wang, and L. Gao. A constrained many-objective
% optimization evolutionary algorithm with enhanced mating and
% environmental selections. IEEE Transactions on Cybernetics, 2023, 53(8):
% 4934-4946.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    methods
        function main(Algorithm,Problem)
            %% Generate Uniform Reference Points
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);

            %% Generate random population
            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                density    = DensityEstimate(Population.objs,W);
                Fitness    = CalFitness2(Population.objs,Population.cons);
                MatingPool = TournamentSelection(2,Problem.N,Fitness,density);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                Population = EnvironmentalSelection([Population,Offspring],W,Problem.N);
            end
        end
    end
end
```

### `CalFitness1.m`
```matlab
function Fitness = CalFitness1(PopObj,PopCon)
% Calculate the fitness of each solution for environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

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

### `CalFitness2.m`
```matlab
function Fitness = CalFitness2(PopObj,PopCon)
% Calculate the fitness of each solution for mating selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    N  = size(PopObj,1);
    CV = sum(max(0,PopCon),2);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
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
    
    %% Calculate the fitnesses
    for i=1:N
        if CV(i)==0||R(i)==0
            Fitness(i)=0;
        else
            Fitness(i)=CalFitness1(PopObj(i,:),PopCon(i,:));
        end
    end
end
```

### `DensityEstimate.m`
```matlab
function density = DensityEstimate(PopObj,W)
% Estimate the density of each individual

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    Zmax=max(PopObj,[],1);
    Zmin=min(PopObj,[],1);
    SPopObj=(PopObj-repmat(Zmin,size(PopObj,1),1))./(repmat(Zmax,size(PopObj,1),1)-repmat(Zmin,size(PopObj,1),1));
    [~,Region] = max(1-pdist2(SPopObj,W,'cosine'),[],2);
    [value,~]=sort(Region,'ascend');
    flag=max(value);
    counter=histc(value,1:flag);
    density=counter(Region);
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(OffSpring,W,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    S       = []; 	% S is the set used for output (replaced by population as output)
    Sc      = [];  	% Sc is used to collect feasible solutions
    PopObj  = OffSpring.objs;
    PopCon  = OffSpring.cons;
    Fitness = CalFitness1(PopObj,PopCon);
    Sc=[Sc,OffSpring(Fitness<1)];

    if length(Sc)==N
        Population=Sc;
    elseif length(Sc)>N
        [FrontNO,MaxNO]=NDSort(Sc.objs,inf);
        if MaxNO==1
           %% -dominated sorting
            FrontNO = tNDSort(Sc.objs,W);
            MaxNO=max(FrontNO);
            for i=1:MaxNO
                S=cat(2,S,Sc(FrontNO==i));
                if length(S)>=N
                    break;
                end
            end
        else
            for i=1:MaxNO
                S=cat(2,S,Sc(FrontNO==i));
                if length(S)>=N
                    break;
                end
            end
        end
        while length(S)>N
            %normalization
            Zmax=max(S.objs,[],1);
            Zmin=min(S.objs,[],1);
            SPopObj=(S.objs-repmat(Zmin,size(S.objs,1),1))./(repmat(Zmax,size(S.objs,1),1)-repmat(Zmin,size(S.objs,1),1));
            
            [~,Region] = max(1-pdist2(SPopObj,W,'cosine'),[],2);% associate each solution in S with their corresponding subregion
            [value,~]=sort(Region,'ascend');
            flag=max(value);
            counter=histc(value,1:flag);                        % counter denotes the number of indiviudals in each subregion
            [~,most_crowded]=max(counter);
            S_crowdest=S(Region==most_crowded);                 % S_crowdest is the set of individuals from the most crowded subregion
            dist=pdist2(S_crowdest.objs,S_crowdest.objs);
            dist(dist==0)=inf;
            [row,~]=find(min(min(dist))==dist);
            St=S_crowdest(row);                                 % St is the set of individuals having the smallest distance in S_crowdest
            [~,Region_St] = max(1-pdist2(St.objs,W,'cosine'),[],2);
            Z = min(St.objs,[],1);
            g_tch=max(abs(St.objs-repmat(Z,length(St),1))./W(Region_St,:),[],2);
            [~,order]=max(g_tch);
            x_wrost=St(order);
            S=setdiff(S,x_wrost);
        end
        Population=S;
    elseif length(Sc)<N
        SI=setdiff(OffSpring,Sc);                                      % SI is the set of infeasible solutions in Hc
        f1=sum(max(0,SI.cons),2);
        [~,Region_SI] = max(1-pdist2(SI.objs,W,'cosine'),[],2);
        Z = min(SI.objs,[],1) ;
        f2=max(abs(SI.objs-repmat(Z,length(SI),1))./W(Region_SI,:),[],2);
        PopObj=[f1,f2];
        [FrontNO,MaxNO]=NDSort(PopObj,inf);
        S=[S,Sc];
        for i=1:MaxNO
            S=cat(2,S,SI(FrontNO==i));
            if length(S)>=N
                last=i;
                break;
            end
        end
        F_last=SI(FrontNO==last);                               % find the individuals in the last front joined into S
        delete_n=size(S,2)-N;
        CV=sum(max(0,F_last.cons),2);
        [~,index]=sort(CV,'descend');
        x_wrost=F_last(index(1:delete_n));
        S=setdiff(S,x_wrost);
        Population=S;
    end
end
```

### `tNDSort.m`
```matlab
function tFrontNo = tNDSort(PopObj,W)
% theta-non-dominated sorting

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming

    N  = size(PopObj,1);
    NW = size(W,1);
    
    [z,znad] = deal(min(PopObj),max(PopObj));
    %% Normalization
    PopObj = Normalization(PopObj,z,znad);

    %% Calculate the d1 and d2 values for each solution to each weight
    normP  = sqrt(sum(PopObj.^2,2));
    Cosine = 1 - pdist2(PopObj,W,'cosine');
    d1     = repmat(normP,1,size(W,1)).*Cosine;
    d2     = repmat(normP,1,size(W,1)).*sqrt(1-Cosine.^2);
    
    %% Clustering
    [~,class] = min(d2,[],2);
    
    %% Sort
    theta = zeros(1,NW) + 5;
    theta(sum(W>1e-4,2)==1) = 1e6;
    tFrontNo = zeros(1,N);
    for i = 1 : NW
        C = find(class==i);
        [~,rank] = sort(d1(C,i)+theta(i)*d2(C,i));
        tFrontNo(C(rank)) = 1 : length(C);
    end
end

function [PopObj,z,znad] = Normalization(PopObj,z,znad)
% Normalize the population and update the ideal point and the nadir point

    [N,M] = size(PopObj);

    %% Update the ideal point
    z = min(z,min(PopObj,[],1));
    
    %% Update the nadir point
    % Identify the extreme points
    W = zeros(M) + 1e-6;
    W(logical(eye(M))) = 1;
    ASF = zeros(N,M);
    for i = 1 : M
        ASF(:,i) = max(abs((PopObj-repmat(z,N,1))./(repmat(znad-z,N,1)))./repmat(W(i,:),N,1),[],2);
    end
    [~,extreme] = min(ASF,[],1);
    % Calculate the intercepts
    Hyperplane = (PopObj(extreme,:)-repmat(z,M,1))\ones(M,1);
    a = (1./Hyperplane)' + z;
    if any(isnan(a)) || any(a<=z)
        a = max(PopObj,[],1);
    end
    znad = a;
    
    %% Normalize the population
    PopObj = (PopObj-repmat(z,N,1))./(repmat(znad-z,N,1));
end
```
