# LCMEA

**Tags**: <2025> <multi> <real> <large/none> <constrained>

## Description
Large-scale constrained multi-objective evolutionary algorithm

## Reference
L. Si, X. Zhang, Y. Zhang, S. Yang, and Y. Tian. An efficient sampling approach to offspring generation for evolutionary large-scale constrained multi-objective optimization. IEEE Transactions on Emerging Topics in Computational Intelligence, 2025, 9(3): 2080-2092.

## Source Code

### `AdaFitness.m`
```matlab
function Fitness = AdaFitness(PopObj,PopCon, VAR)
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
    elseif nargin == 2
        CV = sum(max(0,PopCon),2);
    else
        CV = sum(max(0,PopCon),2);
        CV(CV<=VAR) = 0;
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

### `ESP.m`
```matlab
function Offspring = ESP(Problem, Population)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Number of promising directions
    RefNo = 10;
    SubN  = min(ceil(Problem.N/RefNo), Problem.N);

    %% Population info
    CV     = max(Population.cons, 0);
    Cons   = sum(CV, 2);
    FNum   = sum(Cons==0);
    FIndex = find(Cons==0);
    
    %% Decision boundary
    Lower = repmat(Problem.lower, Problem.N, 1);
    Upper = repmat(Problem.upper, Problem.N, 1);

    %% Select starting and ending solutions
    Fitness   = AdaFitness(Population.objs, Cons);
    [~,Index] = sort(Fitness, 'ascend');
    if rand() < exp(-RefNo*FNum/Problem.N) 
        NDIndex = find(Fitness<=1);
        %% Starting solutions
        if FNum < RefNo
            SIndex = Index(1:RefNo);
        else
            SIndex = FIndex(randperm(FNum, RefNo));
        end

        %% Ending solutions
        EIndex = Index(floor(Problem.N/2)+randperm(Problem.N-floor(Problem.N/2), RefNo));

        if FNum > 1
            RIndex = NDIndex;
        else
            RIndex = Index(1:RefNo);
        end
    else
        [FrontNo, ~] = NDSort(Population.objs, Population.cons, inf);
        NIndex = find(FrontNo==1);
        DIndex = find(FrontNo>1);
        NNum   = numel(NIndex);
        DNum   = numel(DIndex);
        if NNum < 2*RefNo || DNum < RefNo
            %% Convergence-biased selection
            if NNum < RefNo
                SIndex = Index(1:RefNo);
                EIndex = Index(floor(Problem.N/2)+randperm(Problem.N-floor(Problem.N/2), RefNo));
            else
                if DNum < RefNo
                    Select = randperm(NNum, 2*RefNo-DNum);
                    SIndex = NIndex(Select(1:RefNo));
                    EIndex = [DIndex, NIndex(Select(RefNo+1:end))];
                else
                    SIndex = NIndex(randperm(NNum, RefNo));
                    EIndex = DIndex(randperm(DNum, RefNo));
                end
            end
        else
            if rand() < exp(-0.5*RefNo*NNum/Problem.N)
                %% Diversity-biased selection
                Select = randperm(NNum, 2*RefNo);
                SIndex = NIndex(Select(1:RefNo));
                EIndex = NIndex(Select(RefNo+1:end));

            else
                %% Convergence-biased selection
                SIndex = NIndex(randperm(NNum, RefNo));
                EIndex = DIndex(randperm(DNum, RefNo));
            end 
        end
        if FNum > 1
            RIndex = NIndex;
        else
            RIndex = SIndex;
        end
    end

    PopDec = [];
    Vector = Population(EIndex).decs - Population(SIndex).decs;
    Direct = Vector./repmat(sum(Vector.^2,2).^(1/2),1,Problem.D);
    % Sampling in the promising direction
    for i = 1 : 1 : RefNo
        lambda = (Population(RIndex).decs - repmat(Population(SIndex(i)).decs,numel(RIndex),1))*Direct(i,:)';
        beta   = std(lambda);
        r      = normrnd(0, beta, [SubN,1])*(1/(1+((Problem.N-FNum+1)/Problem.N))^ 2);
        OffDec = repmat(r,1,Problem.D).*repmat(Direct(i,:),SubN,1)+repmat(Population(SIndex(i)).decs,SubN,1);
        PopDec = [PopDec;OffDec];
    end
    
    % Perturbation
    BU    = max(Population.decs, [], 1);
    BD    = min(Population.decs, [], 1);
    delta = ((Problem.maxFE-Problem.FE+1)/Problem.maxFE)^ 2;
    if delta < 1e-1
        delta = 0;
    end

    p      = ceil(0.1*Problem.N);
    Pindex = randperm(Problem.N,Problem.N);
    
    % Scaling: perturbation % For LSCM
    if rand() < 0.1
        RNum  = Problem.N - 2*p;
        PopDec(Pindex(1+RNum:RNum+p),:) = PopDec(Pindex(1+RNum:RNum+p),:)+repmat(randn(p,1),1,Problem.D).*PopDec(Pindex(1+RNum:RNum+p),:);
        [~,Index] = sort(Fitness, 'ascend');
        PopDec2   = PopDec(Pindex(1+RNum+p:end),:);
        F         = 0.5;
        ParDec    = Population(randperm(Problem.N,p)).decs;
        SiteS     = rand(p,Problem.D) < repmat(rand(p,1),1,Problem.D);
        BDec      = repmat(Population(Index(randperm(RefNo,1))).decs, p, 1);
        PopDec2(SiteS) = PopDec2(SiteS) + F.*(BDec(SiteS)-ParDec(SiteS));
        PopDec(Pindex(1+RNum+p:end),:) = PopDec2;
    else
        p    = 2*p;
        RNum = Problem.N - p;
        [~, Index] = sort(Fitness, 'ascend');
        PopDec2    = PopDec(Pindex(1+RNum:end),:);
        F      = 0.5;
        ParDec = Population(randperm(Problem.N,p)).decs;
        SiteS  = rand(p,Problem.D) < repmat(rand(p,1),1,Problem.D);
        BDec   = repmat(Population(Index(randperm(RefNo,1))).decs, p, 1);
        PopDec2(SiteS) = PopDec2(SiteS) + F.*(BDec(SiteS)-ParDec(SiteS));
        PopDec(Pindex(1+RNum:end),:) = PopDec2;
    end

    % Region: local perturbation
    SiteR   = rand(RNum, Problem.D) < 2*repmat(rand(RNum,1), 1, Problem.D);
    PR      = repmat((BU-BD)/RefNo, RNum, 1) .* repmat(2*randn(RNum,1),1,Problem.D) * delta;
    PopDec1 = PopDec(Pindex(1:RNum),:);
    PopDec1(SiteR) = PopDec1(SiteR) + PR(SiteR);
    PopDec(Pindex(1:RNum),:) = PopDec1;

    PopDec      = max(min(PopDec,Upper),Lower);
    [proM,disM] = deal(1,20);
    OffspringT  = PopDec;
       
    % Polynomial mutation
    Site = rand(Problem.N,Problem.D) < proM/Problem.D;
    mu   = rand(Problem.N,Problem.D);
    temp = Site & mu<=0.5;
    OffspringT       = min(max(OffspringT,Lower),Upper);
    OffspringT(temp) = OffspringT(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(OffspringT(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffspringT(temp) = OffspringT(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-OffspringT(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffspringT);
end
```

### `EnvCDP.m`
```matlab
classdef EnvCDP
% The environmental selection of SPEA2 (constraint dominance principle)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name = 'Env1';  % environemtal selection name
    end

    methods
        function Population = do(obj, Population, N)
            %% Calculate the fitness of each solution
            Fitness = AdaFitness(Population.objs,Population.cons);

            %% Environmental selection
            Next = Fitness < 1;
            if sum(Next) < N
                [~,Rank] = sort(Fitness);
                Next(Rank(1:N)) = true;
            elseif sum(Next) > N
                Del  = obj.Truncation(Population(Next).objs,sum(Next)-N);
                Temp = find(Next);
                Next(Temp(Del)) = false;
            end
            % Population for next generation
            Population = Population(Next);
            Fitness    = Fitness(Next);
            % Sort the population
            [~,rank]   = sort(Fitness);
            Population = Population(rank);
        end

        function Del = Truncation(obj, PopObj,K)
            % Select part of the solutions by truncation
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
    end
end
```

### `EnvEpsilon.m`
```matlab
classdef EnvEpsilon
% The environmental selection of SPEA2 (epsilon-constraint method)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name = 'Env2';  % environemtal selection name
    end

    methods
        function Population = do(obj, Population, N, epsilon)
            %% Calculate the fitness of each solution
            Fitness = AdaFitness(Population.objs,Population.cons,epsilon);

            %% Environmental selection
            Next = Fitness < 1;
            if sum(Next) < N
                [~,Rank] = sort(Fitness);
                Next(Rank(1:N)) = true;
            elseif sum(Next) > N
                Del  = obj.Truncation(Population(Next).objs,sum(Next)-N);
                Temp = find(Next);
                Next(Temp(Del)) = false;
            end
            % Population for next generation
            Population = Population(Next);
            Fitness    = Fitness(Next);

            % Sort the population
            [~,rank]   = sort(Fitness);
            Population = Population(rank);
        end

        function Del = Truncation(obj, PopObj,K)
            % Select part of the solutions by truncation
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
    end
end
```

### `EnvMOP.m`
```matlab
classdef EnvMOP
% The environmental selection of SPEA2 (multi-objective method with epsilon-relaxation)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name = 'Env3';  % environemtal selection name
    end

    methods
        function Population = do(obj, Population, N, epsilon)
            %% Calculate the fitness of each solution
            [~, nCon] = size(Population.cons);
            PopCon    = sum(max(0,Population.cons),2);
            if sum(sum(PopCon<=epsilon, 2)==nCon) > N
                tmp        = sum(PopCon<=epsilon, 2)==nCon;
                Population = Population(1:end, tmp);
                CV         = sum(max(0,Population.cons),2);
                Fitness    = AdaFitness([Population.objs,CV]);
            else
                CV      = sum(max(0,Population.cons),2);
                Fitness = AdaFitness([CalSDE(Population.objs)',CV]);
            end

            %% Environmental selection
            Next = Fitness < 1;
            if sum(Next) < N
                [~,Rank] = sort(Fitness);
                Next(Rank(1:N)) = true;
            elseif sum(Next) > N
                Del  = obj.Truncation(Population(Next).objs,sum(Next)-N);
                Temp = find(Next);
                Next(Temp(Del)) = false;
            end
            % Population for next generation
            Population = Population(Next);
            Fitness    = Fitness(Next);

            % Sort the population
            [~,rank]   = sort(Fitness);
            Population = Population(rank);
        end

        function Del = Truncation(obj, PopObj,K)
            % Select part of the solutions by truncation
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
    end
end
```

### `EnvRL.m`
```matlab
classdef EnvRL < handle
% Determine environemental selection via reinforcement learning

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        name;       % environemtal selection name
        problem;
        NumEnv;     % candidate environmental selection number
        NumState;   % state number
        Envs;
        ppo;

        % Basic parameter
        ArcSize;
        Zmin;
        Zmax;
        Cmax;
        Zmin1;
        Zmax1;
        Cmax1;
        Center;
        CenterI;
        CenterII;
        CenterIV;

        NRI;
        NRII;
        NRIV;
    end

    methods
        function obj = EnvRL(problem, Archive, MaxGen)
            obj.name     = 'EnvRL';
            obj.problem  = problem;
            obj.NumEnv   = 3;
            obj.NumState = 10;
            obj.Envs     = {EnvCDP(), EnvEpsilon(), EnvMOP()};
            obj.ppo      = PPO(obj.NumEnv, obj.NumState, MaxGen);
            obj.ArcSize  = length(Archive);
            obj.Zmin     = min(Archive.objs, [], 1);
            obj.Zmax     = max(Archive.objs, [], 1);
            obj.Cmax     = max(max(Archive.cons, 0), [], 1);
            obj.Cmax     = max(obj.Cmax, 1e-6);

            % Normalization
            SConsN1 = max(Archive.cons, 0) ./ repmat(obj.Cmax, obj.ArcSize, 1);
            SConsN1 = sum(SConsN1, 2);
            SObjsN1 = (Archive.objs-repmat(obj.Zmin, obj.ArcSize, 1)) ./ repmat(obj.Zmax-obj.Zmin, obj.ArcSize, 1);
            SObjsN1 = sum(SObjsN1, 2);

            obj.Zmin1 = min(SObjsN1);
            obj.Zmax1 = max(SObjsN1);
            obj.Cmax1 = max(SConsN1);

            % Gap solutinon
            CValue     = min(SConsN1);
            OValue     = min(SObjsN1(SConsN1==CValue));
            obj.Center = [OValue, CValue];

            % Re-normalization
            [Sobjs, Scons] = obj.Norm(SObjsN1,SConsN1);

            %% Centroid point
            % I Quadrant: +obj, +con
            IndexI = Sobjs >= 0 & Scons > 0;
            if any(IndexI)
                obj.CenterI = mean([Sobjs(IndexI), Scons(IndexI)],1);
            else
                obj.CenterI = [1,1];
            end

            % II Quadrant: -obj, +con
            IndexII = Sobjs < 0 & Scons > 0;
            if any(IndexII)
                obj.CenterII = mean([Sobjs(IndexII), Scons(IndexII)],1);
            else
                obj.CenterII = [0,0];
            end

            % III Quadrant: -obj, -con
            % empty set

            % IV Quadrant: +obj, -con
            IndexIV = Sobjs > 0 & Scons <= 0;
            if any(IndexIV)
                obj.CenterIV = min([Sobjs(IndexIV), Scons(IndexIV)],[],1);
            else
                obj.CenterIV = [1, 0];
            end
        end

        function [obj, Population, action] = do(obj, Population, VAR, Archive)
      	    % Choose environmental selection
            [state, reward]   = obj.GetStateReward(Archive);            
            [obj.ppo, action] = obj.ppo.GetAction(state, reward);
            % Population, N
            if action == 1
                Population = obj.Envs{1}.do(Population, obj.problem.N);
            elseif action == 2
                Population = obj.Envs{2}.do(Population, obj.problem.N, VAR);
            else
                Population = obj.Envs{3}.do(Population, obj.problem.N, VAR);
            end
        end


        function [state, reward] = GetStateReward(obj, Archive)
            %% Calculate Reward
            % Normalization
            SConsN1 = max(Archive.cons, 0) ./ repmat(obj.Cmax, obj.ArcSize, 1);
            SConsN1 = sum(SConsN1, 2);
            SObjsN1 = (Archive.objs-repmat(obj.Zmin, obj.ArcSize, 1)) ./ repmat(obj.Zmax-obj.Zmin, obj.ArcSize, 1);
            SObjsN1 = sum(SObjsN1, 2);

            [Sobjs, Scons] = obj.Norm(SObjsN1, SConsN1);

            % I Quadrant: +obj, +con
            IndexI = Sobjs >= 0 & Scons > 0;
            if any(IndexI)
                CenterI = mean([Sobjs(IndexI), Scons(IndexI)],1);
            else
                CenterI = [1, 1];
            end
            Reward1 = (obj.CenterI(2) - CenterI(2)) + (obj.CenterI(1) - CenterI(1));
            
            % II Quadrant: -obj, +con
            IndexII = Sobjs < 0 & Scons > 0;
            if any(IndexII)
                CenterII = mean([Sobjs(IndexII), Scons(IndexII)],1);
            else
                CenterII = [0,0];
            end
            Reward2 = (obj.CenterII(2) - CenterII(2));
            
            % III Quadrant: -obj, -con
            IndexIII = Sobjs <= 0 & Scons <= 0;
            IndexIII(Sobjs==0 & Scons==0) = false;
            if any(IndexIII)
                CenterIII = min([Sobjs(IndexIII), Scons(IndexIII)],[],1);
            else
                CenterIII = [0,0];
            end
            Reward3 = -(CenterIII(1))-(CenterIII(2));

            % IV Quadrant: +obj, -con
            IndexIV = Sobjs > 0 & Scons <= 0;
            if any(IndexIV)
                CenterIV = min([Sobjs(IndexIV), Scons(IndexIV)],[],1);
            else
                CenterIV = [1, 0];
            end
            Reward4 = (obj.CenterIV(1)-CenterIV(1));

            reward = Reward1 + Reward2 + Reward3 + Reward4;
            reward = tanh(reward);

            %% Update properties
            obj.ArcSize = length(Archive);
            obj.Zmin    = min(Archive.objs, [], 1);
            obj.Zmax    = max(Archive.objs, [], 1);
            obj.Cmax    = max(max(Archive.cons, 0), [], 1);
            obj.Cmax    = max(obj.Cmax, 1e-6);

            % Normalization
            SConsN1 = max(Archive.cons, 0) ./ repmat(obj.Cmax, obj.ArcSize, 1);
            SConsN1 = sum(SConsN1, 2);
            SObjsN1 = (Archive.objs-repmat(obj.Zmin, obj.ArcSize, 1)) ./ repmat(max(obj.Zmax-obj.Zmin, eps), obj.ArcSize, 1);
            SObjsN1 = sum(SObjsN1, 2);

            obj.Zmin1 = min(SObjsN1);
            obj.Zmax1 = max(SObjsN1);
            obj.Cmax1 = max(SConsN1);

            % Gap solutinon
            CValue     = min(SConsN1);
            OValue     = min(SObjsN1(SConsN1==CValue));
            obj.Center = [OValue, CValue];
           
           %% Centroid point
           [Sobjs, Scons] = obj.Norm(SObjsN1, SConsN1);
            % I Quadrant: +obj, +con
            IndexI = Sobjs >= 0 & Scons > 0;
            if any(IndexI)
                obj.CenterI = mean([Sobjs(IndexI), Scons(IndexI)],1);
            else
                obj.CenterI = [1, 1];
            end
            obj.NRI = sum(IndexI) / obj.ArcSize;

            % II Quadrant: -obj, +con
            IndexII = Sobjs < 0 & Scons > 0;
            if any(IndexII)
                obj.CenterII = mean([Sobjs(IndexII), Scons(IndexII)],1);
            else
                obj.CenterII = [0,0];
            end
            obj.NRII = sum(IndexII) / obj.ArcSize;

            % III Quadrant: -obj, -con
            % empty set

            % IV Quadrant: +obj, -con
            IndexIV = Sobjs > 0 & Scons <= 0;
            if any(IndexIV)
                obj.CenterIV = min([Sobjs(IndexIV), Scons(IndexIV)],[],1);
            else
                obj.CenterIV = [1, 0];
            end
            obj.NRIV = sum(IndexIV) / obj.ArcSize;

            NumPRel = corrcoef(Sobjs,Scons);
            NumPRel = NumPRel(1,2);
            if isnan(NumPRel)
                NumPRel = 0;
            end

            state = [obj.CenterI, obj.CenterII, obj.CenterIV, obj.NRI, obj.NRII, obj.NRIV, NumPRel];
        end


        function [SObjs, SCons] = Norm(obj, Objs, Cons)
            % obj.Center  = [OValue, CValue];
            SObjs = Objs;
            SCons = Cons;
            %% Normalization
            % I Quadrant
            IIndex = Objs>=obj.Center(1) & Cons>obj.Center(2);
            if sum(IIndex)>1
                SObjs(IIndex) = (Objs(IIndex) - obj.Center(1)) / (max(obj.Zmax1 - obj.Center(1), eps));
                SCons(IIndex) = (Cons(IIndex) - obj.Center(2)) / (max(obj.Cmax1 - obj.Center(2), eps));
            end

            % II Quadrant
            IIIndex = Objs<obj.Center(1) & Cons>obj.Center(2);
            if sum(IIIndex)>1
                SObjs(IIIndex) = (Objs(IIIndex) - obj.Zmin1) / (max(obj.Center(1) - obj.Zmin1, eps)) - 1;
                SCons(IIIndex) = (Cons(IIIndex) - obj.Center(2)) / (max(obj.Cmax1 - obj.Center(2), eps));
            end

            % III Quadrant
            IIIIndex = Objs<=obj.Center(1) & Cons<=obj.Center(2);
            IIIIndex(Objs==obj.Center(1) & Cons==obj.Center(2)) = false;
            if sum(IIIIndex)>1
                SObjs(IIIIndex) = (Objs(IIIIndex) - obj.Zmin1) / (max(obj.Center(1) - obj.Zmin1, eps)) - 1;
                SCons(IIIIndex) = (Cons(IIIIndex) + obj.Cmax1) / (max(obj.Center(2) + obj.Cmax1, eps)) - 1;
            end

            % IV Quadrant
            IVIndex = SObjs<obj.Center(1) & Cons<=obj.Center(2);
            if sum(IVIndex)>1
                SObjs(IVIndex) = (Objs(IVIndex) - obj.Center(1)) / (max(obj.Zmax1 - obj.Center(1), eps));
                SCons(IVIndex) = (Cons(IVIndex) + obj.Cmax1) / (max(obj.Center(2) + obj.Cmax1, eps)) - 1;
            end
            if any(SObjs==inf)
                SObjs(SObjs==inf) = 1;
            elseif any(SObjs==-inf)
                SObjs(SObjs==-inf) = 0;
            end

            % Orignal point
            R = find(Objs==obj.Center(1) & Cons==obj.Center(2));
            SCons(R) = 0;
            SObjs(R) = 0;
        end

        function [Population, PopIndex] = FormPop(obj, Archive)
            % Normalization
            SConsN1 = max(Archive.cons, 0) ./ repmat(obj.Cmax, obj.ArcSize, 1);
            SConsN1 = sum(SConsN1, 2);
            SObjsN1 = (Archive.objs-repmat(obj.Zmin, obj.ArcSize, 1)) ./ repmat(obj.Zmax-obj.Zmin, obj.ArcSize, 1);
            SObjsN1 = sum(SObjsN1, 2);
            [Sobjs, Scons] = obj.Norm(SObjsN1, SConsN1);

            % Reverse solutions in II Quadrant
            IndexII         = Sobjs < 0 & Scons > 0;
            Sobjs(IndexII)  = -Sobjs(IndexII);
            [FrontNo, MaxF] = NDSort([Sobjs, Scons], obj.ArcSize);
            Del             = find(FrontNo==MaxF);
            IndexTotal      = 1 : obj.ArcSize;
            PopIndex        = RandDel(FrontNo, IndexTotal, obj.ArcSize/2);
            if isscalar(Del)
                PopIndex(end-numel(Del)+1:end) = Del;
            end

            Population = Archive(PopIndex);
        end
    end
end

function PopIndex = RandDel(FrontNo, Index, N)
    Total = length(FrontNo);
    Num   = Total - N;
    if Total >= 2*Num
        Candiates  = randperm(Total, Num*2);
        Candiates1 = Candiates(1:Num);
        Candiates2 = Candiates(Num+1:end);
        Select     = [Candiates1(FrontNo(Candiates1)<=FrontNo(Candiates2)),Candiates2(FrontNo(Candiates2)<FrontNo(Candiates1))];
        PopIndex   = Index(Select);
        Index(Candiates) = [];
        PopIndex   = [PopIndex;Index];
    else
        Candiates  = randperm(Total, N*2);
        Candiates1 = Candiates(1:N);
        Candiates2 = Candiates(N+1:end);
        Select     = [Candiates1(FrontNo(Candiates1)<=FrontNo(Candiates2)),Candiates2(FrontNo(Candiates2)<FrontNo(Candiates1))];
        PopIndex   = Index(Select);
    end
end
```

### `LCMEA.m`
```matlab
classdef LCMEA < ALGORITHM
% <2025> <multi> <real> <large/none> <constrained>
% Large-scale constrained multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% L. Si, X. Zhang, Y. Zhang, S. Yang, and Y. Tian. An efficient sampling
% approach to offspring generation for evolutionary large-scale constrained
% multi-objective optimization. IEEE Transactions on Emerging Topics in
% Computational Intelligence, 2025, 9(3): 2080-2092.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evoluationary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Set the default parameters
            % Initialize archive
            Archive = Problem.Initialization(2*Problem.N);

            % Environmental selection
            Env = EnvRL(Problem, Archive, floor((Problem.maxFE - Problem.FE) / Problem.N));

            % Parameter of relaxation method
            VAR0 = max(sum(max(0,Archive.cons),2));
            if VAR0 == 0
                VAR0 = 1;
            end
            X   = 0;
            cp  = (-log(VAR0)-6)/log(1-0.5);
            VAR = VAR0*(1-X)^cp;

            [Population, PopIndex] = Env.FormPop(Archive);
            PreAction = inf;
            while Algorithm.NotTerminated(Archive)
                % Generate offspring
                Offspring = ESP(Problem, Population);

                % Environmental selection         
                [Env, Population, Action] = Env.do([Population, Offspring], VAR, Archive);

                % Update Archive
                Archive(PopIndex) = Population;

                % Update Population
                if Action ~= PreAction
                    [Population, PopIndex] = Env.FormPop(Archive);
                end
                PreAction = Action;

                cp  = (-log(VAR0)-6)/log(1-0.5);
                VAR = VAR0*(1-X)^cp;
                if VAR < 1e-6
                    VAR = 0;
                end
                X = X+1/(Problem.maxFE/Problem.N);
            end
        end
    end
end
```

### `PPO.m`
```matlab
classdef PPO < handle
% Proximal policy optimization

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        % Actor and Critic model
        CriticNetwork;
        ActorNetwork;

        % Environmental parameters
        NumAction;
        NumState;

        % Memory pool
        StateBuffer;
        ActionBuffer;
        LogProbBuffer;
        RewardBuffer;
        DoneBuffer;
        ValueBuffer;

        % Step counter used for updating model
        StepStamp = 1;
        Flag      = false;
        MaxGen    = 1;
        Gen       = 1;

        % Loss buffer
        acLoss = [];
        crLoss = [];
        reward = [];
    end

    properties(Access=private)
        % Model parameter
        ActorLearnRate  = 1e-2;
        CriticLearnRate = 2e-2;
        NumEnvs         = 1;
        NumSteps        = 20;
        BatchSize       = 20;
        MiniBatchSize   = 4;
        Gamma           = 0.99;
        GAELambda       = 0.95;
        UpdateEpochs    = 2;
        ClipCoef        = 0.1;
        EntCoef         = 0.01;
        VFCoef          = 0.5;
        MaxGradNorm     = 0.5;
        % Optimizer parameter (the SGDM solver)
        Momentum = 0.9;
    end
    methods
        function obj = PPO(NumAction, NumState, MaxGen)
            obj.MaxGen = MaxGen;

            %% Environmental parameter
            obj.NumAction = NumAction;
            obj.NumState  = NumState;

            %% Define Critic model
            criticPath = [featureInputLayer(obj.NumState,'Name','CriticInput')
                batchNormalizationLayer
                tanhLayer
                fullyConnectedLayer(60,'Name','CriticHidden', 'WeightsInitializer',@(sz) randn(sz) * sqrt(2))
                batchNormalizationLayer
                tanhLayer
                fullyConnectedLayer(1,'Name','CriticOutput', 'WeightsInitializer',@(sz) randn(sz) * sqrt(1))
                batchNormalizationLayer
                tanhLayer];            
            obj.CriticNetwork = dlnetwork(criticPath,Initialize=true);

            %% Define Actor modhel
            actorPath = [featureInputLayer(obj.NumState,'Name','ActorInput')
                batchNormalizationLayer
                tanhLayer
                fullyConnectedLayer(60,'Name','ActorHidden', 'WeightsInitializer',@(sz) randn(sz) * sqrt(2))
                batchNormalizationLayer
                tanhLayer
                fullyConnectedLayer(sum(obj.NumAction),'Name','ActorOutput', 'WeightsInitializer',@(sz) randn(sz) * sqrt(0.01))
                batchNormalizationLayer
                softmaxLayer];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            obj.ActorNetwork = dlnetwork(actorPath,Initialize=true);

            %% Memory pool initialization
            obj.StateBuffer   = dlarray(zeros(obj.NumState, obj.NumSteps),"CB");
            obj.ActionBuffer  = dlarray(zeros(sum(obj.NumAction), obj.NumSteps),"CB");
            obj.LogProbBuffer = dlarray(zeros(1, obj.NumSteps),"CB");
            obj.RewardBuffer  = dlarray(zeros(1, obj.NumSteps),"CB");
            obj.ValueBuffer   = dlarray(zeros(1, obj.NumSteps),"CB");
            obj.DoneBuffer    = dlarray(zeros(1, obj.NumSteps),"CB");
        end

        function [obj, action] = GetAction(obj, State, Reward)
            State = dlarray(State',"CB");

            if mod(obj.StepStamp, obj.NumSteps) == 1 && obj.Gen > obj.NumSteps
                % Reset stepstamp
                stateNext = State;
                doneNext  = 0;

                %% Update reward buffer
                obj.RewardBuffer(obj.StepStamp-1) = Reward;
                
                %% Update actor and critic
                obj = obj.UpdateModel(stateNext, doneNext);

                %% Reset stepstamp
                obj.StepStamp = 1;
            else
                %% Record reward
                if obj.StepStamp > 2
                    obj.RewardBuffer(obj.StepStamp-1) = Reward;
                end                
            end
            %% record path
            obj.StateBuffer(:, obj.StepStamp)   = State;
            obj.DoneBuffer(obj.StepStamp)       = 0;
            [action, logProb, value] = obj.GetActionValue(State);
            obj.ActionBuffer(:, obj.StepStamp)  = action;
            obj.LogProbBuffer(:, obj.StepStamp) = logProb;
            obj.ValueBuffer(obj.StepStamp)      = value;

            %% Update generation recorder
            obj.StepStamp = obj.StepStamp + 1;
            obj.Gen       = obj.Gen + 1;
            obj.reward    = [obj.reward,Reward];
        end

        function [action, logProb, value] = GetActionValue(obj, State)
            % Predict value
            value = obj.GetValue(State);

            % Predict action and other parameters
            logits = forward(obj.ActorNetwork, State);
            if obj.Gen == 1 || rand() < 0.2
                action = randperm(obj.NumAction, 1);
            else
                action = obj.SampleAction(logits);
            end
            logProb = obj.CalLogProbs(logits);
            logProb = sum(logProb, 1);
        end

        function value = GetValue(obj, State)
            State = dlarray(State, "CB");
            value = forward(obj.CriticNetwork, State);
        end

        function obj = UpdateModel(obj, StateNext, DoneNext)
            %% Boostrap value
            valueNext = obj.GetValue(StateNext);
            returns   = zeros(size(obj.RewardBuffer));
            for i = obj.NumSteps : -1 : 1
                if i == obj.NumSteps
                    nextNonTerminal = 1 - DoneNext;
                    nextReturn = valueNext;
                else
                    nextNonTerminal = 1 - obj.DoneBuffer(i+1);
                    nextReturn = returns(i+1);
                end
                returns(i) = obj.RewardBuffer(i) + obj.Gamma*nextNonTerminal*nextReturn;
            end
            advantages = returns - obj.ValueBuffer;

            %% Optimize the actor and critic network
            frac = 1 - (obj.Gen-1) / (2*obj.MaxGen) ;
            acLearnRate = frac * obj.ActorLearnRate;
            crLearnRate = frac * obj.CriticLearnRate;
            for epoch = 1 : 1 : obj.UpdateEpochs
                randIndex = randperm(obj.BatchSize);
                velActor  = [];
                velCritic = [];
                for start = 1 : obj.MiniBatchSize : obj.BatchSize
                    trainIndex      = randIndex(start:1:start-1+obj.MiniBatchSize);
                    trainAdvantages = advantages(trainIndex);
                    trainAdvantages = (trainAdvantages-mean(trainAdvantages)) ./ (std(trainAdvantages)+1e-8);

                    % Calculate gradient and loss
                    [acloss, acGradients] = dlfeval(@obj.ActorLoss,obj.ActorNetwork, obj.StateBuffer(:, trainIndex), trainAdvantages, obj.LogProbBuffer(:,trainIndex));
                    % Optimize actor using the SGDM optimizer
                    [obj.ActorNetwork, velActor] = sgdmupdate(obj.ActorNetwork, acGradients, velActor, acLearnRate, obj.Momentum);
                    % Calculate gradient and loss
                    [crloss, crGradients] = dlfeval(@obj.CriticLoss,obj.CriticNetwork, obj.StateBuffer(:, trainIndex), obj.ValueBuffer(trainIndex), returns(trainIndex));
                    % Optimize critic using the SGDM optimizer
                    [obj.CriticNetwork, velCritic] = sgdmupdate(obj.CriticNetwork, crGradients, velCritic, crLearnRate, obj.Momentum);
                end
            end
        end

        function [crLoss, crGradients] = CriticLoss(obj, CriticNet, State, Values, Returns)
            value = forward(CriticNet, State);

            vLossUnclipped = (value-Returns).^2;
            vClipped       = Values + min(max(value-Values, -obj.ClipCoef), obj.ClipCoef);
            vLossClipped   = (vClipped-Returns).^2;
            vLossMax       = max(vLossClipped,vLossUnclipped);
            crLoss         = 0.5*mean(vLossMax);
            crLoss         = crLoss*obj.VFCoef;

            % Calculate gradient
            crGradients = dlgradient(crLoss,CriticNet.Learnables);
        end

        function [acLoss, acGradients] = ActorLoss(obj, ActorNet, State, Advantages, LogProb)
            logits = forward(ActorNet, State);

            newLogProb = obj.CalLogProbs(logits);
            newLogProb = sum(newLogProb,1);
            entropy    = obj.CalEntropy(logits);
            entropy    = sum(entropy,1);
            logRatio   = newLogProb - LogProb;
            ratio      = exp(logRatio);

            acLoss1     = -Advantages .* ratio;
            acLoss2     = -Advantages .* min(max(ratio, 1-obj.ClipCoef),1+obj.ClipCoef);
            acLoss      = mean(max(acLoss1, acLoss2)) - obj.EntCoef*mean(entropy);
            acGradients = dlgradient(acLoss, ActorNet.Learnables);
        end

        function logProbs = CalLogProbs(obj, logits)
            logProbs = logits - log(sum(exp(logits), 1));
        end

        function entropy = CalEntropy(obj, logits)
            probs   = exp(logits) ./ sum(exp(logits), 1);
            entropy = - sum(probs.*log(probs),1);
        end

       function action = SampleAction(obj, logits)
           probs      = extractdata(exp(logits) ./ sum(exp(logits), 1))';
           probs(end) = 1-sum(probs(1:end-1));
           % sample
           action  = randsrc(1,1,[1:obj.NumAction;probs]);
        end
    end
end
```
