# PIEA

**Tags**: <2024> <multi/many> <real> <expensive>

## Description
Performance indicator-based evolutionary algorithm

## Reference
Y. Li, W. Li, S. Li, and Y. Zhao. A performance indicator-based evolutionary algorithm for expensive high-dimensional multi-/many- objective optimization. Information Sciences, 2024: 121045.

## Source Code

### `NDSort_SDR.m`
```matlab
function [FrontNo,MaxFNo] = NDSort_SDR(Population,nSort)
% Do non-dominated sorting by strengthened dominance relation (SDR)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    zmax   = max(Population.objs);
    zmin   = min(Population.objs);
    PopObj = (Population.objs - zmin)./(zmax - zmin);
    N      = size(PopObj,1);
    NormP  = sum(PopObj,2);
    cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    Angle  = acos(cosine);
    temp   = sort(unique(min(Angle,[],2)));
    minA   = temp(min(ceil(N/2),end));
    Theta  = max(1,(Angle./minA).^1);
    
    dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if NormP(i)*Theta(i,j) < NormP(j)
                dominate(i,j) = true;
            elseif NormP(j)*Theta(j,i) < NormP(i)
                dominate(j,i) = true;
            end
        end
    end
    
    FrontNo = inf(1,N);
    MaxFNo  = 0;
    while sum(FrontNo~=inf) < min(nSort,N)
        MaxFNo  = MaxFNo + 1;
        current = ~any(dominate,1) & FrontNo==inf;
        FrontNo(current)    = MaxFNo;
        dominate(current,:) = false;
    end
end
```

### `PIEA.m`
```matlab
classdef PIEA < ALGORITHM
% <2024> <multi/many> <real> <expensive>
% Performance indicator-based evolutionary algorithm
% eta   ---  5 --- Number of pre-selected survivors
% R_max --- 20 --- Maximum repeat time of offspring generation
% tau   --- 20 --- Window width for history list

%------------------------------- Reference --------------------------------
% Y. Li, W. Li, S. Li, and Y. Zhao. A performance indicator-based 
% evolutionary algorithm for expensive high-dimensional multi-/many-
% objective optimization. Information Sciences, 2024: 121045.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yang Li (email: liyangwust@163.com)
    
    methods
        function main(Algorithm,Problem)
            warning off
            %% Parameter setting
            [eta,R_max,tau] = Algorithm.ParameterSet(5,20,20);

            %% Initialize population
            NI         = Problem.N;
            P          = UniformPoint(NI,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1),zeros(NI,1));
            A          = Population;
            
             %% Initialize history list
            indicator(1) = struct('method','Shift-based density','Choose_record',ones(1,tau),'Win_record',ones(1,tau),'Pw',1/3);
            indicator(2) = struct('method','I_epsilon+','Choose_record',ones(1,tau),'Win_record',ones(1,tau),'Pw',1/3);
            indicator(3) = struct('method','Minkowski distance','Choose_record',ones(1,tau),'Win_record',ones(1,tau),'Pw',1/3);
            
            %% Optimization
            while Algorithm.NotTerminated(A)
                Lp = Shape_Estimate(A,Problem.N);
                % Choose the work indicator
                randtemp = rand;
                if randtemp < indicator(1).Pw
                    Fitness = calFitness_SDE(A.objs,Lp);
                    flag    = 1;
                elseif randtemp < indicator(1).Pw + indicator(2).Pw
                    Fitness = CalFitness_epsilon(A.objs,0.05);
                    flag    = 2;
                else
                    Fitness = calFitness_MD(A.objs,Lp);
                    flag    = 3;
                end

                % Train the surrogate model
                Model = fitrsvm(A.decs,Fitness,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
                
                Dec = A.decs;
                Arc = Dec(randperm(end,NI),:);
                
                % Model-based optimization
                for r = 1 : R_max
                    MatingPool   = TournamentSelection(2,Problem.N,-Fitness);
                    OffspringDec = OperatorDE(Problem, Dec(MatingPool,:),  Arc,  Dec(randperm(end,NI),:));
                    Offspringfit = predict(Model,OffspringDec);
                    if r == 1
                        Arc    = OffspringDec;
                        ArcFit = Offspringfit;
                    else
                        temp = ArcFit<Offspringfit;
                        Arc(temp,:)  = OffspringDec(temp,:);
                        ArcFit(temp) = Offspringfit(temp);
                    end
                end
                
                %Pre-selection
                [~,order] = sort(ArcFit,'descend');
                Arc = Arc(order(1:eta),:);
                
                %Difference comparison
                normADec  = (A.decs - Problem.lower)./(Problem.upper - Problem.lower);
                normDec   = (Arc - Problem.lower)./(Problem.upper - Problem.lower);
                distance  = min(pdist2(normDec,normADec),[],2);
                [~,index] = max(distance);
                
                %Expensive evaluation
                New = Problem.Evaluation(Arc(index,:));
                A   = [A,New];
                
                %Hierarchical evaluation
                [FrontNo,~] = NDSort(A.objs,1);
                score = 0;
                if FrontNo(end) == 1
                    score = 1;
                    [FrontNo,~] = NDSort_SDR(A(FrontNo==1),1);
                    if FrontNo(end) == 1
                        score = 2;
                    end
                end
                indicator = UpdateInformation(flag,score,indicator);
            end
        end
    end
end

function Fitness = calFitness_SDE(PopObj,Lp)
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
    Fitness = 3/(max(Fitness)+eps-min(Fitness))*(Fitness-min(Fitness));
    dis     = pdist2(PopObj, min(PopObj), 'minkowski', Lp);
    dis     = -3/(max(dis)+eps-min(dis))*(dis-min(dis));
    Fitness(Fitness<10^-4) = dis(Fitness<10^-4);
    Fitness = tansig(Fitness);
end

function [Fitness,I,C] = CalFitness_epsilon(PopObj,kappa)
% Calculate the fitness by I_epsilon+

    N      = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end
    
function Fitness = calFitness_MD(PopObj,Lp)
% Calculate the fitness by Minkowski distance
    
    N       = size(PopObj,1);
    fmax    = max(PopObj,[],1);
    fmin    = min(PopObj,[],1);
    PopObj  = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    dis     = pdist2(PopObj, min(PopObj), 'minkowski', Lp);
    Fitness = -dis;
end
```

### `Shape_Estimate.m`
```matlab
function p= Shape_Estimate(Population,N)
% Estimate the shape of PF

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yang Li (email: liyangwust@163.com)

    %% Normalization
    [FrontNo,~] = NDSort(Population.objs,N);
    Population  = Population(FrontNo<=1);
    if length(Population) < 20
        p = 1;
        return;
    end
    PopObj = Population.objs ;
    [N,~]  = size(PopObj);
    PopObj = normalization(PopObj);
    k      = 1.5;
    
    CP = [ 0.27 0.36 0.43 0.5 0.57 0.66 0.75 0.86 1 1.15 1.35 1.6 2 2.4 3.1 4.2 6.5];
    Vp = zeros(1,length(CP));
    for i = 1 : length(CP)
        Gp   = (sum(PopObj.^CP(i),2)).^(1/CP(i));
        temp = sort(Gp);
        Q1   = temp(max(fix(N*0.25),1));
        Q3   = temp(max(fix(N*0.75),1));
        Max  = Q3+k*(Q3-Q1);
        Gp(Gp>Max) = []; % Gp is denoised using box plot
        Vp(i)      = std(Gp./max(Gp));
    end
    [~,index] = min(Vp);
    p         = CP(index);
end
```

### `UpdateInformation.m`
```matlab
function indicator = UpdateInformation(flag,score,indicator)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yang Li (email: liyangwust@163.com)

    a    = indicator(1).Choose_record;
    b    = indicator(2).Choose_record;
    c    = indicator(3).Choose_record;
    a(1) = [];
    b(1) = [];
    c(1) = [];
    
    if flag == 1
        a = [a 1];
        b = [b 0];
        c = [c 0];
    elseif flag == 2
        a = [a 0];
        b = [b 1];
        c = [c 0];
    else
        a = [a 0];
        b = [b 0];
        c = [c 1];
    end
    indicator(1).Choose_record = a;
    indicator(2).Choose_record = b;
    indicator(3).Choose_record = c;
    
    a = indicator(1).Win_record;
    b = indicator(2).Win_record;
    c = indicator(3).Win_record;
    
    a(1) = [];
    b(1) = [];
    c(1) = [];
    
    if score == 0
        a = [a 0];
        b = [b 0];
        c = [c 0];
    else
        if flag == 1
            a = [a score/2];
            b = [b 0];
            c = [c 0];
        elseif flag == 2
            a = [a 0];
            b = [b score/2];
            c = [c 0];
        else
            a = [a 0];
            b = [b 0];
            c = [c score/2];
        end
    end
    indicator(1).Win_record = a;
    indicator(2).Win_record = b;
    indicator(3).Win_record = c;
    
    p = [(eps+sum(indicator(1).Win_record))/(eps+sum(indicator(1).Choose_record))...
         (eps+sum(indicator(2).Win_record))/(eps+sum(indicator(2).Choose_record))...
         (eps+sum(indicator(3).Win_record))/(eps+sum(indicator(3).Choose_record))];
    
    p = p/sum(p);
    indicator(1).Pw = p(1);
    indicator(2).Pw = p(2);
    indicator(3).Pw = p(3);
end
```
