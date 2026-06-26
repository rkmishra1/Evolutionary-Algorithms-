# TriMOEA-TA&R

**Tags**: <2019> <multi> <real/integer> <multimodal>

## Description
Multi-modal MOEA using two-archive and recombination strategies

## Reference
Y. Liu, G. G. Yen, and D. Gong. A multi-modal multi-objective evolutionary algorithm using two-archive and recombination strategies. IEEE Transactions on Evolutionary Computation, 2019, 23(4): 660-674.

## Source Code

### `DecisionVariableAnalysis.m`
```matlab
function [Xic,Xre] = DecisionVariableAnalysis(Problem,NCA,NIA)
% Decision variable analysis
% This code is modified from ControlVariableAnalysis.m and
% DividingDistanceVariables.m in MOEA/DVA

%------------------------------- Copyright --------------------------------
% Copyright 2017-2018 Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------
   
    Xic = false(1,Problem.D);
    Xre = false(1,Problem.D);
    
    %% Find convergence-related variable
    for i = 1 : Problem.D
        x      = rand(1,Problem.D).*(Problem.upper-Problem.lower) + Problem.lower;
        S      = repmat(x,NCA,1);
        S(:,i) = ((1:NCA)'-1+rand(NCA,1))/NCA*(Problem.upper(i)-Problem.lower(i)) + Problem.lower(i);
        S = Problem.CalObj(S);
        S = unique(S,'rows'); % Delete the duplicate
        [~,MaxFNo] = NDSort(S,inf);
        if MaxFNo == size(S,1)
            Xic(i) = true;
        else
            Xre(i) = true;
        end
    end
    
    %% Interdependence 
    % Generate the initial population        
    PopDec = rand(Problem.N,Problem.D);
    PopDec = PopDec.*repmat(Problem.upper-Problem.lower,Problem.N,1) + repmat(Problem.lower,Problem.N,1);   
    PopObj = Problem.CalObj(PopDec);
    % Interdependence analysis
    interaction = false(Problem.D);
    interaction(logical(eye(Problem.D))) = true;
    for i = 1 : Problem.D-1
        for j = i+1 : Problem.D
            for time2try = 1 : NIA
                % Detect whether the i-th and j-th decision variables are interacting
                x    = randi(Problem.N);
                a2   = rand*(Problem.upper(i)-Problem.lower(i)) + Problem.lower(i);
                b2   = rand*(Problem.upper(j)-Problem.lower(j)) + Problem.lower(j);
                Decs = repmat(PopDec(x,:),3,1);
                Decs(1,i) = a2;
                Decs(2,j) = b2;
                Decs(3,[i,j]) = [a2,b2];
                F      = Problem.CalObj(Decs);
                delta1 = F(1,:) - PopObj(x,:);
                delta2 = F(3,:) - F(2,:);
                interaction(i,j) = interaction(i,j) | any(delta1.*delta2<0);
                interaction(j,i) = interaction(i,j);                
            end
        end
    end
    
    %% Group based on Interdependence
    while sum(sum(interaction(Xic,Xre)))
        for i = find(Xic==1)
            fprintf('i=%d\n',i);
            if sum(interaction(i,Xre))
                Xic(i) = false;
                Xre(i) = true;
            end
        end      
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population] = EnvironmentalSelection(Population,N)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright 2017-2018 Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    N = min(N,length(Population));
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
end
```

### `Recombination.m`
```matlab
function FS = Recombination(AC,AD,Xic,Xre,eps_peak,fS,RankC)
% Recombination

%------------------------------- Copyright --------------------------------
% Copyright 2017-2018 Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    ACDec = AC.decs;
    ADDec = AD.decs;     
    [N,D] = size(ADDec);
    peak  = find( ((fS - min(fS)) < eps_peak) & RankC == 1 );    
    FS    = NaN(length(peak).*N,D);
    FS(:,Xic) =  kron( ACDec(peak',Xic),ones(N,1));
    FS(:,Xre) =  repmat(ADDec(:,Xre),length(peak),1);          
end
```

### `TriMOEATAR.m`
```matlab
classdef TriMOEATAR < ALGORITHM
% <2019> <multi> <real/integer> <multimodal>
% Multi-modal MOEA using two-archive and recombination strategies
% p_con       --- 0.5  --- Probability of selecting parents from the convergence archive
% sigma_niche --- 0.1  --- Niche radius in the decision space
% eps_peak    --- 0.01 --- Accuracy level to detect peaks
% NR          --- 100  --- Number of refernece points
% NCA         --- 20   --- Number of sampling solutions in control variable analysis
% NIA         --- 6    --- Maximum number of tries required to judge the interaction

%------------------------------- Reference --------------------------------
% Y. Liu, G. G. Yen, and D. Gong. A multi-modal multi-objective
% evolutionary algorithm using two-archive and recombination strategies.
% IEEE Transactions on Evolutionary Computation, 2019, 23(4): 660-674.
%------------------------------- Copyright --------------------------------
% Copyright 2017-2018 Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [p_con,sigma_niche,eps_peak,NR,NCA,NIA] = Algorithm.ParameterSet(0.5,0.1,0.01,100,20,6);  

            %% Decision Variable Analysis and Generate random population
            [Xic,Xre] = DecisionVariableAnalysis(Problem,NCA,NIA);

            %% Generate the reference points    
            R = UniformPoint(NR,Problem.M);

            %% Initialization
            Population = Problem.Initialization();
            NC = Problem.N;
            ND = Problem.N;
            AC = [];
            AD = [];
            Pc = floor(p_con.*Problem.N);
            Pd = Problem.N - Pc;

            %% Update Archives
            Z            = min(Population.objs,[],1);  
            [AC,RankC,~] = UpdateConvergenceArchive(AC,Population,NC,Z,Xic,sigma_niche,Problem);
            [AD,RankD]   = UpdateDiversityArchive(AD,Population,ND,R,Z,Xre,sigma_niche,Problem);

            %% Optimization
            while Algorithm.NotTerminated(AD)
                % Generate new Population based on two Archives
                MatingPoolC = TournamentSelection(2,Pc,RankC);
                MatingPoolD = TournamentSelection(2,Pd,RankD);
                Parents     = [AC(MatingPoolC),AD(MatingPoolD)];
                Parents     = Parents(randperm(Problem.N));
                Population  = OperatorGA(Problem,Parents);        
                % Update Archives
                Z             = min([Z;Population.objs],[],1);
                [AC,RankC,fS] = UpdateConvergenceArchive(AC,Population,NC,Z,Xic,sigma_niche,Problem);        
                [AD,RankD]    = UpdateDiversityArchive(AD,Population,ND,R,Z,Xre,sigma_niche,Problem);                
                % Recombination
                if Problem.FE >= Problem.maxFE && sum(Xic) > 0
                    FS = Recombination(AC,AD,Xic,Xre,eps_peak,fS,RankC);
                    AD = Problem.Evaluation(FS);
                end
            end
        end
    end
end
```

### `UpdateConvergenceArchive.m`
```matlab
function [AC,Rank,fS] = UpdateConvergenceArchive(AC,Population,NC,Z,Xic,sigma_niche,Problem)
% Update the Convergence Archive

%------------------------------- Copyright --------------------------------
% Copyright 2017-2018 Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    Population = [Population,AC];
    PopObj = Population.objs;
    PopDec = Population.decs;
    N      = size(PopObj,1);   
    Rank   = inf(1,N);      % Rank of each solution
    nrank  = 1;             % Current rank
    
    %% Normalize
    PopObj = PopObj - repmat(Z,N,1);
    PopDec = (PopDec - repmat(Problem.lower,N,1))./repmat(Problem.upper-Problem.lower,N,1);
    
    %% Convergence indicator
    fS = mean(PopObj');
    
    %% Calculate distance between every two solutions in the IC decsion subspace
    d = pdist2(PopDec(:,Xic),PopDec(:,Xic),'chebychev');
       
    %% Rank
    Choose = false(1,N);
    Q      = true(1,N);
    Q1     = false(1,N);
    while sum(Choose) < NC
        if sum(Q) == 0
           Q     = Q1;
           Q1    = false(1,N);
           nrank = nrank+1;
        end
        % Choose x with min FS 
        temp1 = fS == min(fS(Q));
        xmin  = find(and(temp1,Q));
        xmin  = xmin(1);
        Rank(xmin)   = nrank;
        Choose(xmin) = true;
        Q(xmin)      = false;        
        % Delete solution near x_min
        temp3      = d(xmin,:);
        temp2      = temp3<sigma_niche;
        Delete     = and(temp2,Q);
        Q(Delete)  = false;
        Q1(Delete) = true;      
    end
    AC   = Population(Choose);
    Rank = Rank(Choose);
    fS   = fS(Choose);
end
```

### `UpdateDiversityArchive.m`
```matlab
function [AD,Rank] = UpdateDiversityArchive(AD,Population,ND,R,Z,Xre,sigma_niche,Problem)
% Update the Diversity Archive

%------------------------------- Copyright --------------------------------
% Copyright 2017-2018 Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    Population = [Population,AD];
    [FrontNo,MaxFNo] = NDSort(Population.objs,ND);

    %% Select the solutions in the last front
    if length(find(FrontNo<=MaxFNo))== ND
       Next = FrontNo<=MaxFNo;
    else
       Next   = FrontNo < MaxFNo ;
       NQ     = ND-sum(Next);
       Last   = find(FrontNo==MaxFNo);
       Choose = LastSelection(Population(Last).objs,Population(Last).decs,NQ,R,Z,Xre,sigma_niche,Problem); %Algorithm 4, lines 5-30
       Next(Last(Choose)) = true;
    end
    
    %% Population for next generation
    AD   = Population(Next);
    Rank = FrontNo(Next);
end

function Choose = LastSelection(PopObj,PopDec,NQ,R,Z,Xre,sigma_niche,Problem)
% Select solutions with good diversity in the last front    
    N  = size(PopObj,1);
    NR = size(R,1);
    
    %% Normalize
    PopObj = PopObj - repmat(Z,N,1);
    PopDec = (PopDec - repmat(Problem.lower,N,1))./repmat(Problem.upper-Problem.lower,N,1);
  
    %% Calculate distance between every solution and reference point in the objective space
    theta = pdist2(R,PopObj,'cosine');
    
    %% Calculate distance between every two solutions in the reminder decsion subspace
    d = pdist2(PopDec(:,Xre),PopDec(:,Xre),'chebychev');   
    
    %% Cluster
    [thmin,label] = min(theta,[],1);   
    C1 = false(NR,N);
    C2 = false(NR,N);
    for j = 1 : NR
        member   = find(label==j);
        member1  = label==j;
        [~,temp] = sort(thmin(member));
        for i = temp
           if any(d(member(i),and(C1(j,:),member1))<sigma_niche) 
               C2(j,member(i))=true;
           else
               C1(j,member(i))=true;
           end
        end        
    end    
    
    %% Make selected solution == NQ
    while sum(sum(C1))>NQ
        cmax     = max(sum(C1,2));
        jmax     = sum(C1,2)== cmax;
        temp1    = find(sum(C1(jmax,:),1)>0);        
        [~,xmax] = max(thmin(temp1));
        xmax     = temp1(xmax);
        C1(label(xmax),xmax) = false;
    end   
    while sum(sum(C1))<NQ
        c2       = sum(C2,2)>0;
        cmin     = min(sum(C1(c2,:),2));
        jmin     = and(sum(C1,2)==cmin,c2);
        temp1    = find(sum(C2(jmin,:),1)>0);
        [~,xmin] = min(thmin(temp1));
        xmin     = temp1(xmin);
        C2(label(xmin),xmin) = false;
        C1(label(xmin),xmin) = true;
    end    
    Choose = sum(C1)>0;  
end
```
