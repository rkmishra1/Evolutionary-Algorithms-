# MOEA-CKF

**Tags**: <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>

## Description
Multi-objective evolutionary algorithm based on cross-scale knowledge fusion

## Reference
Z. Ding, L. Chen, D. Sun, and X. Zhang. Efficient sparse large-scale multi-objective optimization based on cross-scale knowledge fusion. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(11): 6989-7001.

## Source Code

### `BinaryCrossover.m`
```matlab
function Offspring = BinaryCrossover(Parent1,Parent2)
% Unbalanced binary crossover

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Offspring = logical(Parent1);
    for i = 1 : size(Offspring,1)
        diff = find(Parent1(i,:)~=Parent2(i,:));
        MOne = mean(Offspring(i,diff));
        r    = min(min(0.5,2*MOne),2*(1-MOne));
        rate = zeros(1,length(diff));
        rate(Offspring(i,diff))     = r/2/MOne;
        rate(~Offspring(i,diff))    = r/2/(1-MOne);
        exchange                    = rand(1,length(diff)) < rate;
        Offspring(i,diff(exchange)) = ~Offspring(i,diff(exchange));
    end
end
```

### `BinaryMutation.m`
```matlab
function Offspring = BinaryMutation(Offspring)
% Unbalanced binary mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,D] = size(Offspring);
    MOne  = mean(Offspring,2);
    r     = min(min(1/D,2*MOne),2*(1-MOne));
    rate1 = repmat(r./2./MOne,1,D);
    rate0 = repmat(r./2./(1-MOne),1,D);
    rate  = zeros(N,D);
    rate(Offspring)     = rate1(Offspring);
    rate(~Offspring)    = rate0(~Offspring);
    exchange            = rand(N,D) < rate;
    Offspring(exchange) = ~Offspring(exchange);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,sRatio] = EnvironmentalSelection(Population,Dec,Mask,N,len,num)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions
    success = false(1,length(Population));
    [~,uni] = unique(Population.objs,'rows');
    if length(uni) == 1
        [~,uni] = unique(Population.decs,'rows');
    end
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));
    
    %% Calculate the fitness of each solution
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    PopObj = Population.objs;
    fmax   = max(PopObj(FrontNo==1,:),[],1);
    fmin   = min(PopObj(FrontNo==1,:),[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);  

    %% Environmental selection
    Last = find(FrontNo==MaxFNo);
    del  = Truncation(PopObj(Last,:),length(Last)-N+sum(Next));
    Next(Last(~del)) = true;
    
    % Population for next generation 
    success(uni(Next)) = true;
    s1     = sum(success(len+1:len+num));
    s2     = sum(success(len+num+1:end));
    sRatio = (s1+1e-6)./(s1+s2+1e-6);
    sRatio = max(min(sRatio,0.9),0.1);
    Population = Population(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
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

### `MOEACKF.m`
```matlab
classdef MOEACKF < ALGORITHM
% <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Multi-objective evolutionary algorithm based on cross-scale knowledge fusion

%------------------------------- Reference --------------------------------
% Z. Ding, L. Chen, D. Sun, and X. Zhang. Efficient sparse large-scale
% multi-objective optimization based on cross-scale knowledge fusion. IEEE
% Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(11):
% 6989-7001.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Lei Chen (email: soarfx@163.com)

    methods
        function main(Algorithm,Problem)
            %% Population initialization
            REAL = ~strcmp(Problem.encoding,'binary');
            [Dec,Mask,Population,Fitness] = PriorAnalysis_initialization(Problem,REAL);
            [Population,Dec,Mask]         = EnvironmentalSelection(Population,Dec,Mask,Problem.N,0,0);
            rho   = 0.5;
            GROUP = [];     % In order to prevent outliers in Kmeans algorithm
           
            %% Optimization
            while Algorithm.NotTerminated(Population)  
                [FrontNo,~] = NDSort(Population.objs,Population.cons,inf);
                CrowdDis    = CrowdingDistance(Population.objs,FrontNo);
                % Divide the population, Pop1 is used as the population for 
                % dual dimension reduction to generate offspring, and Pop2 
                % is used as the population of dual grouping.
                Pop1_Site = rho > rand(1,length(Population));                
                if sum(Pop1_Site) >= 2
                    Pop1  = Population(Pop1_Site);
                    Dec1  = Dec(Pop1_Site,:);
                    Mask1 = Mask(Pop1_Site,:);
                    Pop2  = Population(~Pop1_Site);
                    Dec2  = Dec(~Pop1_Site,:);
                    Mask2 = Mask(~Pop1_Site,:);
                elseif sum(~Pop1_Site) < 1
                    Pop1  = Population;
                    Dec1  = Dec;
                    Mask1 = Mask;
                    [Pop2,Dec2,Mask2] = deal([]);                
                else
                    [Pop1,Dec1,Mask1] = deal([]);
                    Pop2  = Population;
                    Dec2  = Dec;
                    Mask2 = Mask;     
                end
                % decision variable Sparsity analysis
                [Local_Knowlege,Global_Knowledge,Fitness,NSV,SV,theta] = SparsityAnalysis(Problem,Mask,FrontNo,Fitness,GROUP);
                GROUP = [NSV;SV];
                % dual dimension reduction
               if ~isempty(Pop1)
                   [OffDec1,OffMask1,len1] = Reproduction1(Problem,Pop1,Dec1,Mask1,FrontNo,CrowdDis,Pop1_Site,Local_Knowlege,Global_Knowledge,NSV,SV,theta,REAL);
                   Offspring1 = Problem.Evaluation(OffDec1.*OffMask1);
               else
                   [OffDec1,OffMask1,Offspring1] = deal([]);
                   len1 = 0;
               end
               % dual grouping 
               if ~isempty(Pop2)
                   MatingPool = TournamentSelection(2,(Problem.N-len1)*2,FrontNo(~Pop1_Site),-CrowdDis(~Pop1_Site));
                   [OffDec2,OffMask2] = Reproduction2(Problem,Dec2(MatingPool,:),Mask2(MatingPool,:),SV,NSV,REAL);
                   Offspring2 = Problem.Evaluation(OffDec2.*OffMask2);
               else
                  [OffDec2,OffMask2,Offspring2] = deal([]);
               end
               [Population,Dec,Mask,sRatio] = EnvironmentalSelection([Population,Offspring1,Offspring2],[Dec;OffDec1;OffDec2],[Mask;OffMask1;OffMask2],Problem.N,length(Population),len1);
               rho = (rho+sRatio)/2;
            end
        end
    end
end
```

### `PriorAnalysis_initialization.m`
```matlab
function [TDec,TMask,TempPop,Fitness] = PriorAnalysis_initialization(Problem,REAL)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    TDec    = [];
    TMask   = [];
    TempPop = [];
    Fitness = zeros(1,Problem.D);
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
        Fitness    = Fitness + NDSort([Population.objs,Population.cons],inf);
    end 
    
    % Generate initial population
    if REAL
        Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
    else
        Dec = ones(Problem.N,Problem.D);
    end
    Mask = zeros(Problem.N,Problem.D);
    for i = 1 : Problem.N
        Mask(i,TournamentSelection(2,ceil(rand*Problem.D),Fitness)) = 1;
    end
    Population = Problem.Evaluation(Dec.*Mask);
    TDec       = [TDec;Dec];
    TMask      = [TMask;Mask];
    TempPop    = [TempPop,Population];
end
```

### `Reproduction1.m`
```matlab
function [OffDec,OffMask,len1] = Reproduction1(Problem,Pop1,Dec1,Mask1,FrontNo,CrowdDis,Pop1_Site,Local_Knowlege,Global_Knowlege,NSV,SV,theta,REAL)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    len1 = length(Pop1);
    MatingPool = TournamentSelection(2,len1*2,FrontNo(Pop1_Site),-CrowdDis(Pop1_Site));
    
    % Mask evolution of offspring
    Parent1Mask = Mask1(MatingPool(1:end/2),:);
    Parent2Mask = Mask1(MatingPool((end/2+1):end),:);
    [N,D] = size(Parent1Mask); 
  
    OffMask = false(N,D);
    
    for i = 1 : N
        if rand < theta
            allOne   = Local_Knowlege(1,:);
            other    = Local_Knowlege(3,:);
            TOffMask = false(1,D);          
            TOffMask(other(NSV)) = BinaryCrossover(Parent1Mask(i,other(NSV)),Parent2Mask(i,other(NSV)));
            TOffMask(other(NSV)) = BinaryMutation(TOffMask(other(NSV)));
            
            TOffMask(other(SV)) = BinaryCrossover(Parent1Mask(i,other(SV)),Parent2Mask(i,other(SV)));
            TOffMask(other(SV)) = BinaryMutation(TOffMask(other(SV)));            
            TOffMask(allOne)    = true;  
        else
            allOne   = Global_Knowlege(1,:);
            other    = Global_Knowlege(3,:);
            TOffMask = false(1,D);                                  
            TOffMask(other(NSV)) = BinaryCrossover(Parent1Mask(i,other(NSV)),Parent2Mask(i,other(NSV)));
            TOffMask(other(NSV)) = BinaryMutation(TOffMask(other(NSV)));
            
            TOffMask(other(SV)) = BinaryCrossover(Parent1Mask(i,other(SV)),Parent2Mask(i,other(SV)));
            TOffMask(other(SV)) = BinaryMutation(TOffMask(other(SV)));
            TOffMask(allOne)    = true;  
        end        
        OffMask(i,:) = TOffMask;
    end

    if REAL
        allOne  = Global_Knowlege(1,:);
        allZero = Global_Knowlege(2,:);
        other   = Global_Knowlege(3,:);
        % data preprocessing
        T_Best_Dec       = Dec1(:,~allZero);
        VariableMean     = mean(T_Best_Dec,1);
        VariableNormlize = max(T_Best_Dec)-min(T_Best_Dec)+1e-6;  
        Temp_PopDec      = (T_Best_Dec-repmat(VariableMean,N,1))./repmat(VariableNormlize,N,1);
        % SVD decomposition
        [U,~,~] = svd((1/N).*(Temp_PopDec'*Temp_PopDec));
        % Guide subspace size
        K = sum(other(NSV)) + sum(allOne);
        
        U_reduce      = U(:,1:K);
        Pop_reduce    = Temp_PopDec*U_reduce;
        OffDec_reduce = GAhalfCross(Pop_reduce(MatingPool,:));
        % Restore to original space
        T_OffDec = OffDec_reduce*U_reduce';
        T_OffDec = (T_OffDec.*repmat(VariableNormlize,N,1))+repmat(VariableMean,N,1);
        % Perform polynomial variation
        T_OffDec = PM(Problem,T_OffDec,~allZero);
        %  to reduce the computational complexity of PCA
        if sum(allZero) > 0
            OffDec = zeros(N,D);
            OffDec(:,allZero)  = GAhalf(Problem,Dec1(MatingPool,allZero),allZero);
            OffDec(:,~allZero) = T_OffDec;
        else
            OffDec = T_OffDec;
        end
    else
        OffDec = ones(N,D);
    end
end

function Offspring = PM(Problem,Offspring,site)
    [~,~,proM,disM] = deal(1,20,1,20);
    [N,D] = size(Offspring);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Lower = Lower(:,site);
    Upper = Upper(:,site);
    
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end

function Offspring = GAhalfCross(Parent)
    [proC,disC,~,~] = deal(1,20,1,20);
    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);
    beta    = zeros(N,D);
    mu      = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
end

function Offspring = GAhalf(Problem,Parent,allZero)
    [proC,disC,proM,disM] = deal(1,20,1,20);
    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);

    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;

    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Lower = Lower(:,allZero);
    Upper = Upper(:,allZero);

    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end
```

### `Reproduction2.m`
```matlab
function [OffDec,OffMask] = Reproduction2(Problem,ParentDec,ParentMask,NSV,SV,REAL)
% The operator of MOEA/CKF

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    Parent1Mask = ParentMask(1:end/2,:);
    Parent2Mask = ParentMask(end/2+1:end,:);
    [N,D] = size(Parent1Mask);
    
    %% Binary variation
    OffMask = false(N,D);
    OffMask(:,SV) = BinaryCrossover(Parent1Mask(:,SV),Parent2Mask(:,SV));
    OffMask(:,SV) = BinaryMutation(OffMask(:,SV));
    
    OffMask(:,NSV) = BinaryCrossover(Parent1Mask(:,NSV),Parent2Mask(:,NSV));
    OffMask(:,NSV) = BinaryMutation(OffMask(:,NSV));
    
    %% Real variation
    if REAL
        Parent1 = ParentDec(1:floor(end/2),:);
        Parent2 = ParentDec(floor(end/2)+1:floor(end/2)*2,:);             
        OffDec  = zeros(N,D);
        for i = 1 : N
            NonZero = OffMask(i,:);
            OffDec(i,NonZero)  = SBXhalf(Parent1(i,NonZero),Parent2(i,NonZero));
            OffDec(i,NonZero)  = PM(Problem,OffDec(i,NonZero),NonZero);
            OffDec(i,~NonZero) = SBXhalf(Parent1(i,~NonZero),Parent2(i,~NonZero));
            OffDec(i,~NonZero) = PM(Problem,OffDec(i,~NonZero),~NonZero);
        end                       
    else
        OffDec = ones(size(OffMask));
    end
end

function Offspring = SBXhalf(Parent1,Parent2)
    [proC,disC,~,~] = deal(1,20,1,20);
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
end

function Offspring = PM(Problem,Offspring,indexSite)
    [N,D] = size(Offspring);
    [proM,disM] = deal(1,20);
    % Polynomial mutation
    Lower = repmat(Problem.lower(indexSite),N,1);
    Upper = repmat(Problem.upper(indexSite),N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
 end
```

### `SparsityAnalysis.m`
```matlab
function [Local_Knowlege,Gobal_Knowlege,Fitness,NSV,SV,theta] = SparsityAnalysis(Problem,Mask,FrontNo,Fitness,GROUP)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Local_Knowlege = false(3,Problem.D);
    Gobal_Knowlege = false(3,Problem.D);
    EliteMask      = Mask(FrontNo==1,:);
    Local_Knowlege(1,:) = all(EliteMask,1);
    Local_Knowlege(2,:) = all(~EliteMask,1);
    Local_Knowlege(3,:) = ~ Local_Knowlege(1,:) & ~ Local_Knowlege(2,:);
    
    Gobal_Knowlege(1,:) = all(Mask,1);
    Gobal_Knowlege(2,:) = all(~Mask,1);
    Gobal_Knowlege(3,:) = ~ Gobal_Knowlege(1,:) & ~ Gobal_Knowlege(2,:);
    
    theta   = size(EliteMask,1)/size(Mask,1);    
    Fitness = (Fitness-min(Fitness)+1e-6)./(max(Fitness)-min(Fitness)+1e-6)+1e-6;
    
    if size(Fitness',1) < 2
        NSV = GROUP(1,:);
        SV  = GROUP(2,:);
    else
        Cluster = kmeans(Fitness',2);  
        Fitness = Fitness + (1-mean(EliteMask)).*(0.5*theta+0.5*(Problem.FE/Problem.maxFE)).*Fitness;       
        if mean(Fitness(Cluster==1)) > mean(Fitness(Cluster==2))
            NSV = Cluster == 2;
            SV  = Cluster == 1;
        else
            NSV = Cluster == 1;
            SV  = Cluster == 2;
        end
    end        
end
```
