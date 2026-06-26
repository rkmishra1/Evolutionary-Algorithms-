# NSGA-II+ARSBX

**Tags**: <2021> <multi> <real/integer> <constrained/none>

## Description
NSGA-II with adaptive rotation based simulated binary crossover

## Reference
L. Pan, W. Xu, L. Li, C. He, and R. Cheng. Adaptive simulated binary crossover for rotated multi-objective optimization. Swarm and Evolutionary Computation, 2021, 60: 100759.

## Source Code

### `ARSBX.m`
```matlab
function Offspring = ARSBX(Problem,Parent,Parameter)
% Rotation based simulated binary crossover and polynomial mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [B,Centroid,pc]       = deal(Parameter{:});
    [proC,disC,proM,disM] = deal(1,2,1,20);

    %% Simulated binary crossover
    ParentDec = Parent.decs;
    [N,D]     = size(ParentDec);  
    Norigin   = round(N*pc/2)*2;
    Neig      = round(N*(1-pc)/2)*2;
    ParentDecOrigin = ParentDec(1:Norigin,:);
    ParentDecEig    = ParentDec(end-Neig+1:end,:);
    Parent1Dec      = ParentDecOrigin(1:Norigin/2,:);
    Parent2Dec      = ParentDecOrigin(Norigin/2+1:end,:);
    beta = zeros(Norigin/2,D);
    mu   = rand(Norigin/2,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],Norigin/2,D);
    beta(rand(Norigin/2,D)<0.5) = 1;
    beta(repmat(rand(Norigin/2,1)>proC,1,D)) = 1;
    OffspringDec = [(Parent1Dec+Parent2Dec)/2+beta.*(Parent1Dec-Parent2Dec)/2
                    (Parent1Dec+Parent2Dec)/2-beta.*(Parent1Dec-Parent2Dec)/2];
    Flag = ones(Norigin,1);

    NormalParentDec = ParentDecEig - Centroid;
    [~,D]           = size(B);
    ParentDecEig    = NormalParentDec*B;
    Parent1Dec      = ParentDecEig(1:Neig/2,:);
    Parent2Dec      = ParentDecEig(Neig/2+1:end,:);
    beta = zeros(Neig/2,D);
    mu   = rand(Neig/2,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],Neig/2,D);
    beta(rand(Neig/2,D)<0.5) = 1;
    beta(repmat(rand(Neig/2,1)>proC,1,D)) = 1;
    OffspringRDec = [(Parent1Dec+Parent2Dec)/2+beta.*(Parent1Dec-Parent2Dec)/2
                    (Parent1Dec+Parent2Dec)/2-beta.*(Parent1Dec-Parent2Dec)/2];
    OffspringRDec = OffspringRDec*B' + Centroid;
    Flag          = [Flag;2*ones(Neig,1)];       
    OffspringDec  = [OffspringDec;OffspringRDec];
    
    %% Polynomial mutation
    [~,D] = size(ParentDec);
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    OffspringDec(temp) = OffspringDec(temp)+(Upper(temp)-Lower(temp)).*(nthroot(2.*mu(temp)+(1-2.*mu(temp)).*...
                         (1-(OffspringDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1),disM+1)-1);              
    temp = Site & mu>0.5; 
    OffspringDec(temp) = OffspringDec(temp)+(Upper(temp)-Lower(temp)).*(1-nthroot(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                        (1-(Upper(temp)-OffspringDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1),disM+1));
    Offspring = Problem.Evaluation(OffspringDec,Flag);
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

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
end
```

### `NSGAIIARSBX.m`
```matlab
classdef NSGAIIARSBX < ALGORITHM
% <2021> <multi> <real/integer> <constrained/none>
% NSGA-II with adaptive rotation based simulated binary crossover

%------------------------------- Reference --------------------------------
% L. Pan, W. Xu, L. Li, C. He, and R. Cheng. Adaptive simulated binary
% crossover for rotated multi-objective optimization. Swarm and
% Evolutionary Computation, 2021, 60: 100759.
%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            B  = eye(Problem.D);
            m  = 0.5*(Problem.upper - Problem.lower);
            ps = 0.5;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = ARSBX(Problem,Population(MatingPool),{B,m,ps});
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                [B,m,ps,Population] = UpdateParameter(Problem,Population);
            end
        end
    end
end
```

### `UpdateParameter.m`
```matlab
function [B,m,ps,Population] = UpdateParameter(Problem,Population)
% Update the parameters in ARSBX

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopDec   = Population.decs;
    Flag     = Population.adds(zeros(length(Population),1));
    Ori      = sum(Flag == 1);
    Eig      = sum(Flag == 2);
    ps       = 1/(1+exp(-Problem.M*sqrt(Problem.D)*((Ori+1)/(Eig+Ori+2)-0.5)*Problem.FE/Problem.maxFE));
    C        = cov(PopDec);
    [B,E]    = eig(C);
    E        = diag(E);
    E        = sqrt(E);
    [~,Rank] = sort(E,'descend');
    B        = B(:,Rank);
    m        = mean(PopDec);
end
```
