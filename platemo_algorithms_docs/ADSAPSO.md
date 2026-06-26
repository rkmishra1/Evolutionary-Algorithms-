# ADSAPSO

**Tags**: <2022> <multi/many> <real/integer> <expensive>

## Description
Adaptive dropout based surrogate-assisted particle swarm optimization

## Reference
J. Lin, C. He, and R. Cheng. Adaptive dropout for high-dimensional expensive multiobjective optimization. Complex & Intelligent Systems, 2022, 8(1): 271C285.

## Source Code

### `ADSAPSO.m`
```matlab
classdef ADSAPSO < ALGORITHM
% <2022> <multi/many> <real/integer> <expensive>
% Adaptive dropout based surrogate-assisted particle swarm optimization
% k    ---   5 --- Number of re-evaluated solutions
% beta --- 0.5 --- Percentage of Dropout

%------------------------------- Reference --------------------------------
% J. Lin, C. He, and R. Cheng. Adaptive dropout for high-dimensional 
% expensive multiobjective optimization. Complex & Intelligent Systems,
% 2022, 8(1): 271C285.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [k,beta] = Algorithm.ParameterSet(5,0.5);
            Init_Num = 100;     % Initial number of solutions
            N_a      = 200;     % The number of solutions for building surrogate models
            N_s      = 50;      % The number of the well- and poorly performing solutions

            %% Generate initial population
            InitDec  = repmat((Problem.upper - Problem.lower),Init_Num, 1).* lhsdesign(Init_Num, Problem.D) + repmat(Problem.lower, Init_Num, 1);  % lhs design
            Arc      = Problem.Evaluation(InitDec);
            
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                Offspring  = Operator(Problem,Arc,k,beta,N_a,N_s);
                Offspring  = Problem.Evaluation(Offspring);
                Arc        = [Arc,Offspring];
            end
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
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

### `Operator.m`
```matlab
function Offspring = Operator(Problem,Arc,k,beta,N_a,N_s)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
 
% This function is written by Jianqing Lin
    
    %% Environmental Selection
    [Arc,FrontNo,CrowdDis] = EnvironmentalSelection(Arc,min(N_a,size(Arc,2)));

    %% Parameter setting     
    PopDec = Arc.decs;
    D      = Problem.D;
    M      = Problem.M;
    NP     = Problem.N;

    %% Solution Sort & Selection
    [~,index_FNCD] = sortrows([FrontNo;CrowdDis]',[1,-2]);
    Candidate_D    = PopDec(index_FNCD(1:k),:);
    
    Index_Well = index_FNCD(1:N_s);
    Index_Poor = index_FNCD(end-N_s+1:end);
    
    %% Statistics Mean Value
    Model_Dif      = mean(PopDec(Index_Well,:))-mean(PopDec(Index_Poor,:)); 
    Model_Dif_sort = sort(abs(Model_Dif),'descend');

    %% Selected beta*D Decision Variables
    Index_dif = find(abs(Model_Dif) >= Model_Dif_sort(ceil(beta*D)));
    
    %% Build RBF Models for Low-dimensional Decision Space
    Decs_Surrogate = PopDec(:,Index_dif);
    Objs_Surrogate = Arc.objs;
    
    for i = 1 : M
        RBF_para{i} = RBFCreate(Decs_Surrogate, Objs_Surrogate(:,i), 'gaussian');
    end
   
    %% Reproduction by RBF-assisted PSO
    Population      = EnvironmentalSelection(Arc,NP);
    PopDec          = Population.decs;
    Pop_Surrogate   = Surrogate_individual(PopDec(:,Index_dif),Population.objs);
    Offspring_d     = Reproduction(Problem,Pop_Surrogate,RBF_para,Index_dif);
    
    %% Replacement 
    Offspring_d              = EnvironmentalSelection(Offspring_d,k);
    Candidate_D(:,Index_dif) = Offspring_d.decs;
    Offspring                = Candidate_D;
end
```

### `Reproduction.m`
```matlab
function Offspring = Reproduction(Problem,Particle,RBF_para,Index_dif)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin

    [proM,disM] = deal(1,20);
    [N,~]     = size(Particle.decs);
    [~,M]     = size(Particle.objs);
    Pbest     = Particle;
    Archive   = UpdateArchive(Particle(NDSort(Particle.objs,1)==1),[],N);
    Gbest     = Archive(randi(ceil(length(Archive)/10),1,N));

    for gen = 1 : 100

       %% Parameter setting
        W = 0.5;
        ParticleDec = Particle.decs;
        PbestDec    = Pbest.decs;
        GbestDec    = Gbest.decs;
        [N,D]       = size(ParticleDec);
        ParticleVel = Particle.adds(zeros(N,D));

      %% Particle swarm optimization
        r1        = repmat(rand(N,1),1,D);
        r2        = repmat(rand(N,1),1,D);
        OffVel    = W.*ParticleVel + r1.*(PbestDec-ParticleDec) + r2.*(GbestDec-ParticleDec);
        OffDec    = ParticleDec + OffVel;
        
        Off_Objs  = Surrogate_Predictor(OffDec, RBF_para, M);
        Offspring = Surrogate_individual(OffDec,Off_Objs,OffVel);
        Particle  = Offspring;
        
       %% Update
        Pbest      = UpdatePbest(Pbest,Offspring);
        [Gbest,~]  = UpdateGbest(Gbest,N);

    end
    
    %% Polynomial mutation
    if Problem.FE >= Problem.maxFE*0.75
        
        Lower = repmat(Problem.lower,N,1);
        Upper = repmat(Problem.upper,N,1);
        Lower = Lower(:,Index_dif);
        Upper = Upper(:,Index_dif);
        Site  = rand(N,D) < proM/D;
        mu    = rand(N,D);
        
        temp  = Site & mu<=0.5;
        OffDec = min(max(OffDec,Lower),Upper);
        OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
        
        temp = Site & mu>0.5; 
        OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

        Off_Objs  = Surrogate_Predictor(OffDec, RBF_para, M);
        Offspring = Surrogate_individual(OffDec,Off_Objs,OffVel);
    end

end

function [Gbest,CrowdDis] = UpdateGbest(Gbest,N)
    % Update the Problem best set
    Gbest    = Gbest(NDSort(Gbest.objs,1)==1);
    CrowdDis = CrowdingDistanceSameFront(Gbest.objs);
    [~,rank] = sort(CrowdDis,'descend');
    Gbest    = Gbest(rank(1:min(N,length(Gbest))));
    CrowdDis = CrowdDis(rank(1:min(N,length(Gbest))));
end

function Pbest = UpdatePbest(Pbest,Population)
    % Update the local best position of each particle
    replace        = ~all(Population.objs>=Pbest.objs,2);
    Pbest(replace) = Population(replace);
end

function A = UpdateArchive(A,S,K)
% Update the external archive
    %% Combine A and S and normalize the objective values
    Combine = [A,S];
    fmin    = repmat(min(A.objs,[],1),length(Combine),1);
    fmax    = repmat(max(A.objs,[],1),length(Combine),1);
    PopObj  = (Combine.objs-fmin)./(fmax-fmin);
    [N,M]   = size(PopObj);
    
    %% Calculate the shifted distance between each two solutions
    sde = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            sde(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    
    %% Calculate Cv
    dis = sqrt(sum(PopObj.^2,2));
    % Use 1-dis instead of 1-dis/sqrt(M)
    Cv  = 1 - dis;
    
    %% Calculate d1 and d2
    Cosine = 1 - pdist2(PopObj,ones(1,M),'cosine');
    d1     = dis.*Cosine;
    d2     = dis.*sqrt(1-Cosine.^2);

    %% Insert each solution in S into the archive A
    Choose = 1 : length(A); % Indices of selected solutions in [A,S]
    for i = 1 : length(S)
        % Check the dominance relations between S(i) and solutions in A
        mark = false(1,length(Choose)+1);
        for j = 1 : length(Choose)
            flag = any(S(i).obj<Combine(Choose(j)).obj,2) - any(S(i).obj>Combine(Choose(j)).obj,2);
            if flag == 1
                mark(j) = true;
            elseif flag == -1
                mark(end) = true;
                break;
            end
        end
        Choose(mark(1:end-1)) = [];
        if ~mark(end)
            % Insert S(i) into A
            Choose = [Choose,length(A)+i];
            if length(Choose) > K
                % Delete the one in A with the lowest BFE value
                [~,worst]     = min(CalBFE(sde(Choose,Choose),Cv(Choose),d1(Choose),d2(Choose)));
                Choose(worst) = [];
            end
        end
    end
    
    %% Sort the solutions in A according to their BFE values
    A        = Combine(Choose);
    [~,rank] = sort(CalBFE(sde(Choose,Choose),Cv(Choose),d1(Choose),d2(Choose)),'descend');
    A        = A(rank);
end

function BFE = CalBFE(sde,Cv,d1,d2)
% Calculate the BFE value of each solution in the population

% This function is modified from the code in
% http://security.szu.edu.cn/people.aspx?p=QiuzhenLin
% Where the parameters are a little different from the ones in the paper

    %% Calculate Cd
    SDE = min(sde,[],2);
    Cd  = (SDE-min(SDE))./(max(SDE)-min(SDE));
    
    %% Determine the value of alpha and beta of each solution
    alpha   = zeros(length(Cv),1);
    beta    = zeros(length(Cv),1);
    meanCd  = mean(Cd);
    meanCv  = mean(Cv);
    meand1  = mean(d1);
    meand2  = mean(d2);
    case111 = Cv >  meanCv & d1 <= meand1 & Cd <= meanCd;
    case112 = Cv >  meanCv & d1 <= meand1 & Cd >  meanCd;
    case121 = Cv >  meanCv & d1 >  meand1 & Cd <= meanCd;
    case122 = Cv >  meanCv & d1 >  meand1 & Cd >  meanCd;
    case211 = Cv <= meanCv & d1 <= meand1 & d2 >  meand2 & Cd <= meanCd;
    case212 = Cv <= meanCv & d1 <= meand1 & d2 >  meand2 & Cd >  meanCd;
    case221 = Cv <= meanCv &(d1 >  meand1 | d2 <= meand2)& Cd <= meanCd;
    case222 = Cv <= meanCv &(d1 >  meand1 | d2 <= meand2)& Cd >  meanCd;
    alpha(case111) = rand(sum(case111),1)*0.3+0.8; beta(case111) = 1;
    alpha(case112) = 1;   beta(case112) = 1;
    alpha(case121) = 0.6; beta(case121) = 1;
    alpha(case122) = 0.9; beta(case122) = 1;
    alpha(case211) = rand(sum(case211),1)*0.3+0.8; beta(case211) = rand(sum(case211),1)*0.3+0.8;
    alpha(case212) = 1;   beta(case212) = 1;
    alpha(case221) = 0.2; beta(case221) = 0.2;
    alpha(case222) = 1;   beta(case222) = 0.2;

    %% The BFE value of each solution
    BFE = alpha.*Cd + beta.*Cv;
end

function CrowdDis = CrowdingDistanceSameFront(PopObj)
% Calculate the crowding distance of each solution in the same front

    [N,M]    = size(PopObj);
    
    CrowdDis = zeros(1,N);
    Fmax     = max(PopObj,[],1);
    Fmin     = min(PopObj,[],1);
    for i = 1 : M
        [~,rank] = sortrows(PopObj(:,i));
        CrowdDis(rank(1))   = inf;
        CrowdDis(rank(end)) = inf;
        for j = 2 : N-1
            CrowdDis(rank(j)) = CrowdDis(rank(j))+(PopObj(rank(j+1),i)-PopObj(rank(j-1),i))/(Fmax(i)-Fmin(i));
        end
    end
end
```
