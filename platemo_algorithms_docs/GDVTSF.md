# GDVTSF

**Tags**: <2025> <multi> <real> <large/none>

## Description
Generational difference vector based tri-entropy structure framework

## Reference
Y. Xu, Y. Zhang, and W. Hu. A generational difference vector based tri-entropy structure optimizer for large-scale multiobjective optimization. Swarm and Evolutionary Computation, 2025, 98: 102079.

## Source Code

### `GDVTSF.m`
```matlab
classdef GDVTSF < ALGORITHM
% <2025> <multi> <real> <large/none>
% Generational difference vector based tri-entropy structure framework
% delta --- 0.3 --- The threshold ratio for WOF framework
% K     ---   5 --- The number of clusters
% B     ---  20 --- The number of sampling points
% G     --- 0.5 --- Global control factor
% T_clu ---  20 --- Max iterations for clustering

%------------------------------- Reference --------------------------------
% Y. Xu, Y. Zhang, and W. Hu. A generational difference vector based
% tri-entropy structure optimizer for large-scale multiobjective
% optimization. Swarm and Evolutionary Computation, 2025, 98: 102079.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Hannyx Xu
% If you have any questions, please open an issue in our GitHub repository:
% https://github.com/yuzhang576/GDVTSF

    methods
        function main(Algorithm,Problem)
            %% Core Parameters Setting (Analyzed in the paper)
            [delta, K, B, G, T_clu] = Algorithm.ParameterSet(0.3, 5, 20, 0.5, 20);
            
            %% Default Settings for WOF and FDV
            [Rate, Acc]          = deal(0.6, 0.4);
            [gamma, groups, psi] = deal(4, 2, 3);
            [t1, t2]             = deal(1000, 500);
            q                    = Problem.M + 1;
            t_pop                = 10;
            sel_m                = 3;

            %% Generate a set of uniformly distributed points
            [V, Problem.N] = UniformPoint(Problem.N, Problem.M);
            N              = Problem.N;
            D              = Problem.D;
            
            %% Initialization
            % Generate random population
            population = Problem.Initialization();
            LpopObj    = population.objs;
            LpopDec    = population.decs;
            swarmN     = floor(N/3);
            % Global best solution
            gBest = GDVTSF_Utils.update_gbest(population, swarmN, V, 0);
            % Clustering decision vectors
            [Lidx, Lcenter] = kmeans(LpopDec, K, "MaxIter", T_clu);
            Lrd = zeros(K,D);
            for i = 1 : K
                ch       = Lidx == i;
                Lrd(i,:) = mean(abs(Lcenter(i,:)-LpopDec(ch,:)),1);
            end
            % Generational difference vector
            GDV = zeros(N,D);
            % Depart to 3 swarms randomly
            if length(population) >= 3
                Rank = randperm(length(population),swarmN*3);
            else
                Rank = [1,1,1];
            end
            p1 = Rank(1:swarmN);
            p2 = Rank(swarmN+1:2*swarmN);
            p3 = Rank(2*swarmN+1:end);
            % Initialize WOF Environment
            WOF_WeightIndividual.Current(Problem);
            
            %% Optimization loop
            while Algorithm.NotTerminated(gBest)
                if Problem.FE < delta*Problem.maxFE
                    population = WOF_Utils.fill_population(population, Problem);
                    % Normal optimisation step for t1 evaluations
                    population = GDVTSF_Utils.WOF_optimizer_GDV_TSO(Problem, population, t1, K, B, G, T_clu); 
                    % Selection of xPrime solutions 
                    xPrimeList = WOF_Utils.select_xprimes(population, q, sel_m); 
                    WList      = [];
                    % Do for each xPrime
                    for c = 1 : size(xPrimeList,2)
                        xPrime = xPrimeList(c);
                        % Create variable groups 
                        [Group, gamma_n] = WOF_Utils.create_groups(Problem, gamma, xPrime, groups);
                        % A dummy object is needed to simulate the global class
                        GlobalDummy = WOF_Utils.create_global_dummy(gamma_n, xPrime, Group, Problem, t_pop, psi, 2);
                        % Create initial population for the transformed problem
                        WeightPopulation = WOF_Utils.create_initial_weight_population(GlobalDummy.N, gamma_n, GlobalDummy);
                        % Optimise the transformed problem 
                        WeightPopulation = WOF_Utils.optimise_by_MOEAD(GlobalDummy, WeightPopulation, GlobalDummy.uniW, t2-t_pop, true);
                        % Extract the population 
                        W     = WOF_Utils.extract_population(WeightPopulation, Problem, population, Group, psi, xPrime, q, sel_m);
                        WList = [WList,W];
                    end
                    % Join populations. Duplicate solutions need to be removed. 
                    population = WOF_Utils.eliminate_duplicates([population,WList]);
                    population = WOF_Utils.fill_population(population, Problem);
                    % Environmental Selection
                    [population,~,~] = WOF_Utils.environmental_selection(population,Problem.N);
                else
                    % Generate new population
                    popDec = GDVTSF_Utils.operator_GDV_TSO(Problem, population, gBest, GDV, p1, p2, p3, B, G);
                    popObj = population.objs;
                    % Calculate GDV
                    [Lcenter, Lidx, Lrd, GDV] = GDVTSF_Utils.cal_GDV(N, D, K, T_clu, popDec, popObj, Lrd, LpopObj, Lcenter, Lidx);
                    LpopObj                   = popObj;
                    % FDV (Fuzzy Difference Vector operation)
                    iter = Problem.FE/Problem.maxFE;
                    if iter <= Rate
                        population = GDVTSF_Utils.operator_FDV(Problem,Rate,Acc,popDec);
                    else
                        population = Problem.Evaluation(popDec);
                    end
                end
                % Update gBest
                gBest = GDVTSF_Utils.update_gbest([gBest, population], swarmN, V, (Problem.FE/Problem.maxFE)^2);
            end
        end
    end
end
```

### `GDVTSF_Utils.m`
```matlab
classdef GDVTSF_Utils
% GDVTSF_Utils - Static class for GDVTSF specific utility and auxiliary functions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    methods(Static)
        %% 1. Calculate Fitness
        function fitness = cal_fitness(PopObj)
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
            fitness = min(Dis,[],2);
        end
        
        %% 2. Calculate GDV
        function [center,idx,rd,GDV] = cal_GDV(N, D, K, T_clu, popDec, popObj, Lrd, LpopObj, Lcenter, Lidx)
            GDV = zeros(N,D);
            rd  = zeros(K,D);
            R   = rand(N,1);
            
            [idx,center] = kmeans(popDec, K, "MaxIter", T_clu);
            
            for i = 1 : K
                dis      = dist(center(i,:),Lcenter'); 
                [~,mi]   = min(dis,[],2);
                ch       = idx==i;
                C1popObj = popObj(ch,:);
                C1num    = sum(ch);
                C2popObj = LpopObj(Lidx==mi,:);
                
                fitness = GDVTSF_Utils.cal_fitness([C1popObj;C2popObj]);
                fmax    = max(fitness,[],1)+0.001;
                fmin    = min(fitness,[],1);
                fitness = (fitness-repmat(fmin,size(fitness,1),1))./repmat(fmax-fmin,size(fitness,1),1);
                
                f1      = max(fitness(1:C1num));
                f2      = max(fitness(C1num+1:end));
                rd(i,:) = mean(abs(center(i,:)-popDec(ch,:)), 1);
                
                if xor(f1 > f2, rd(i,:) < Lrd(mi,:))
                    om = 0;
                else
                    om = 1;
                end
                vec       = sign(f1-f2)*(center(i,:)-Lcenter(mi,:))+om*(center(i,:)-popDec(ch,:));
                GDV(ch,:) = R(ch,:).*rd(i,:).*vec./norm(vec);
            end
            
            if isnan(GDV)
                GDV = zeros(K,D);
            end
        end
        
        %% 3. Operator FDV
        function Offspring = operator_FDV(Problem,Rate,Acc,OffDec,OffVel)
            % Fuzzy Evolution Sub-stages Division
            Total = 1;
            S     = floor(sqrt(2*Rate*Total/Acc));
            Step  = zeros(1,S+2);  
            for i = 1 : S
                Step(i+1) = (S*i-i*i/2)*Acc;
            end
            Step(S+2) = Rate*Total;  
            
            % Fuzzy Operation
            R    = Problem.upper-Problem.lower;
            iter = Problem.FE/Problem.maxFE;  
            for i = 1 : S+1
                if iter>Step(i) && iter<=Step(i+1)
                    gamma_a = R*10^-i.*floor(10^i*R.^-1.*(OffDec-Problem.lower)) + Problem.lower;
                    gamma_b = R*10^-i.*ceil(10^i*R.^-1.*(OffDec-Problem.lower)) + Problem.lower;
                    miu1    = 1./(OffDec-gamma_a);
                    miu2    = 1./(gamma_b-OffDec);
                    logical = miu1-miu2>0;
                    OffDec  = gamma_b;
                    OffDec(logical) = gamma_a(logical);
                end
            end
            
            if nargin > 4
                Offspring = Problem.Evaluation(OffDec,OffVel);
            else
                Offspring = Problem.Evaluation(OffDec);
            end
        end
        
        %% 4. Operator GDV TSO
        function popDec = operator_GDV_TSO(Problem, population, gBest, GDV, p1, p2, p3, B, G)
            N = Problem.N;
            D = Problem.D;
            
            fitness  = GDVTSF_Utils.cal_fitness(population.objs);
            gBestDec = gBest.decs;
            popDec   = population.decs;
            popVel   = zeros(N,D);
            
            swarmN       = floor(N/3);
            [p1, p2, p3] = GDVTSF_Utils.triple_swarm_select(fitness, p1, p2, p3);
            gBestSelDec  = zeros(swarmN,D);
            
            for i = 1 : swarmN
                V      = zeros(B, D);
                H      = zeros(B, 1);
                dis    = dist(popDec(p1(i),:),gBestDec');
                [~,mi] = min(dis,[],2);
                
                % Use Global control factor G from hyperparameters
                if rand() <= G
                    gBestSelDec(i,:) = gBestDec(mi,:);
                else
                    gBestSelDec(i,:) = gBestDec(randi(length(gBest), 1),:);
                end
                
                for j = 1 : B
                    t      = rand();
                    r      = rand();
                    V(j,:) = popDec(p1(i),:) + r.*(t.*(popDec(p2(i),:)-popDec(p1(i),:)+(1-t).*(popDec(p3(i),:)-popDec(p1(i),:))));
                    L1     = norm(popDec(p1(i),:)-popDec(p2(i),:));
                    L2     = norm(popDec(p2(i),:)-popDec(p3(i),:));
                    L3     = norm(popDec(p3(i),:)-popDec(p1(i),:));
                    L      = (L1+L2+L3)/3;
                    H(j)   = log(2*pi*exp(1)*(1-exp(-norm(V(j,:)-popDec(p1(i),:)).^2/L^2)))/fitness(p1(i));
                    H(j)   = H(j) + log(2*pi*exp(1)*(1-exp(-norm(V(j,:)-popDec(p2(i),:)).^2/L^2)))/fitness(p2(i));
                    H(j)   = H(j) + log(2*pi*exp(1)*(1-exp(-norm(V(j,:)-popDec(p3(i),:)).^2/L^2)))/fitness(p3(i));
                end
                
                [~,idx]         = sort(H, 'descend');
                popVel(p2(i),:) = V(idx(1),:) - popDec(p2(i),:);
                popVel(p3(i),:) = V(idx(2),:) - popDec(p3(i),:);
            end
            
            rate = Problem.FE/Problem.maxFE;
            C1   = (1.5-rate)*repmat(rand(N,1),1,D);
            C2   = (1.5-rate)*repmat(rand(N,1),1,D);
            
            popVel(p1,:) = (gBestSelDec - popDec(p1,:));
            popVel(p2,:) = C1(p2,:).*popVel(p2,:) + C2(p2,:).*(popDec(p1,:)-popDec(p2,:));
            popVel(p3,:) = C1(p3,:).*popVel(p3,:) + C2(p3,:).*(popDec(p1,:)-popDec(p3,:));
            
            velMax = repmat((Problem.upper-Problem.lower)/1.001, N, 1);
            popVel = max(min(popVel,velMax),-velMax);
            
            try
                popDec = popDec + popVel + GDV;
            catch
            end
            
            Lower  = repmat(Problem.lower,N,1);
            Upper  = repmat(Problem.upper,N,1);
            popDec = max(min(popDec,Upper),Lower);
            
            disM  = 20;
            Site1 = repmat(rand(N,1)<0.999,1,D);
            Site2 = rand(N,D) < 1/D;
            mu    = rand(N,D);
            
            temp  = Site1 & Site2 & mu<=0.5;
            popDec(temp) = popDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                           (1-(popDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
            
            temp  = Site1 & Site2 & mu>0.5; 
            popDec(temp) = popDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                           (1-(Upper(temp)-popDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
        end
        
        %% 5. Triple Swarm Select
        function [p1,p2,p3] = triple_swarm_select(fitness, p1, p2, p3)
            Change1     = fitness(p3) > fitness(p1);
            Temp        = p1(Change1);
            p1(Change1) = p3(Change1);
            p3(Change1) = Temp;
            
            Change2     = (fitness(p2) > fitness(p1)) & (fitness(p2) > fitness(p3));
            Temp        = p1(Change2);
            p1(Change2) = p2(Change2);
            p2(Change2) = Temp;
        end
        
        %% 6. Update GBest
        function gBest = update_gbest(population, N, V, theta)
            population = population(NDSort(population.objs,1)==1);
            popObj     = population.objs;
            [NN,M]     = size(popObj);
            NV         = size(V,1);
            
            popObj = popObj - repmat(min(popObj,[],1),NN,1);
            CV     = sum(max(0,population.cons),2);
            
            cosine = 1 - pdist2(V,V,'cosine');
            cosine(logical(eye(length(cosine)))) = 0;
            gamma  = min(acos(cosine),[],2);
            
            Angle         = acos(1-pdist2(popObj,V,'cosine'));
            [~,associate] = min(Angle,[],2);
            
            Next = zeros(1,NV);
            for i = unique(associate)'
                current1 = find(associate==i & CV==0);
                current2 = find(associate==i & CV~=0);
                if ~isempty(current1)
                    APD      = (1+M*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(popObj(current1,:).^2,2));
                    [~,best] = min(APD);
                    Next(i)  = current1(best);
                elseif ~isempty(current2)
                    [~,best] = min(CV(current2));
                    Next(i)  = current2(best);
                end
            end
            gBest = population(Next(Next~=0));
        end
        
        %% 7. WOF Optimizer GDVTSO
        function population = WOF_optimizer_GDV_TSO(Problem, population, t, K, B, G, T_clu)
            [V, Problem.N] = UniformPoint(Problem.N, Problem.M);    
            N              = Problem.N;
            D              = Problem.D;
            swarmN         = floor(N/3);
            gBest          = GDVTSF_Utils.update_gbest(population, swarmN, V, 0);
            
            if length(population) >= 3
                Rank = randperm(length(population),swarmN*3);
            else
                Rank = [1,1,1];
            end
            
            p1 = Rank(1:swarmN);
            p2 = Rank(swarmN+1:2*swarmN);
            p3 = Rank(2*swarmN+1:end);
            
            LpopObj         = population.objs;
            LpopDec         = population.decs;
            [Lidx, Lcenter] = kmeans(LpopDec, K, "MaxIter", T_clu);
            Lrd             = zeros(K,D);
            
            for i = 1 : K
                ch       = Lidx==i;
                Lrd(i,:) = mean(abs(Lcenter(i,:)-LpopDec(ch,:)),1);
            end
            GDV = zeros(N,D);
            cnt = 0;
            
            while cnt <= t
                cnt        = cnt + 100;
                popDec     = GDVTSF_Utils.operator_GDV_TSO(Problem, population, gBest, GDV, p1, p2, p3, B, G);
                population = Problem.Evaluation(popDec);
                popObj     = population.objs;
                [Lcenter, Lidx, Lrd, GDV] = GDVTSF_Utils.cal_GDV(N, D, K, T_clu, popDec, popObj, Lrd, LpopObj, Lcenter, Lidx);
                LpopObj                   = popObj;
            end
        end
    end
end
```

### `WOF_Utils.m`
```matlab
classdef WOF_Utils
% WOF_Utils - Static class for all WOF utility and auxiliary functions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    methods(Static)
        %% 1. Create Global Dummy
        function GlobalDummy = create_global_dummy(gamma, xPrime, G, Global, populationSize, psi, optimiser)
            GlobalDummy       = {};
            GlobalDummy.lower = zeros(1,gamma);
            GlobalDummy.upper = ones(1,gamma).*2.0;
            if or(optimiser == 2, optimiser == 4)
                [uniW,GlobalDummy.N] = UniformPoint(populationSize,Global.M);
                GlobalDummy.uniW     = uniW;
            else
                GlobalDummy.N = populationSize;
            end
            GlobalDummy.xPrime      = xPrime;
            GlobalDummy.G           = G;
            GlobalDummy.psi         = psi;
            GlobalDummy.xPrimelower = Global.lower;
            GlobalDummy.xPrimeupper = Global.upper;
            GlobalDummy.isDummy     = true;
            GlobalDummy.Global      = Global;
        end
        
        %% 2. Eliminate Duplicates
        function Population = eliminate_duplicates(input)
            [~,ia,~]   = unique(input.objs,'rows');
            Population = input(ia);
        end
        
        %% 3. Fill Population
        function Population = fill_population(input, Problem)
            Population               = input;
            theCurrentPopulationSize = size(input,2);
            if theCurrentPopulationSize < Problem.N
                amountToFill = Problem.N-theCurrentPopulationSize;
                FrontNo      = NDSort(input.objs,inf);
                CrowdDis     = CrowdingDistance(input.objs,FrontNo);
                MatingPool   = TournamentSelection(2,amountToFill+1,FrontNo,-CrowdDis);
                Offspring    = OperatorGA(Problem,input(MatingPool));
                Population   = [Population,Offspring(1:amountToFill)];
            end
        end
        
        %% 4. Create Initial Weight Population
        function WeightPopulation = create_initial_weight_population(N, gamma, GlobalDummy)
            decs             = rand(N,gamma).*2.0;
            WeightPopulation = [];
            for i = 1:N
                solution         = WOF_WeightIndividual(decs(i,:),GlobalDummy);
                WeightPopulation = [WeightPopulation, solution];
            end
        end
        
        %% 5. Extract Population
        function W = extract_population(WeightPopulation, Problem, Population, G, psi, xPrime, q, methodToSelectxPrimeSolutions)
            % Step 1
            weightIndividualList = WOF_Utils.select_xprimes(WeightPopulation, q, methodToSelectxPrimeSolutions);
            calc                 = size(weightIndividualList,2)*size(Population,2);
            PopDec1              = ones(calc,Problem.D);
            count                = 1;
            for wi = 1 : size(weightIndividualList,2)
                weightIndividual = weightIndividualList(wi);
                weightVars       = weightIndividual.dec;
                for i = 1 : size(Population,2)
                    individualVars   = Population(i).dec;
                    x                = WOF_Utils.transformation_function_matrix_form(individualVars,weightVars(G),Problem.upper,Problem.lower, psi);
                    PopDec1(count,:) = x;
                    count            = count + 1;
                end
            end
            W1 = Problem.Evaluation(PopDec1);
            
            % Step 2
            PopDec2 = [];
            for wi = 1 : size(WeightPopulation,2)
                weightIndividual = WeightPopulation(wi);
                weightVars       = weightIndividual.dec;
                individualVars   = xPrime.dec;
                x                = 1:Problem.D;
                for j = 1 : Problem.D
                    x(j) = WOF_Utils.transformation_function(individualVars(j),weightVars(G(j)),Problem.upper(j),Problem.lower(j), psi);   
                end
                PopDec2 = [PopDec2;x]; 
            end
            W2 = Problem.Evaluation(PopDec2);
            W  = [W1,W2];
        end
        
        %% 6. Environmental Selection
        function [Population,FrontNo,CrowdDis] = environmental_selection(Population,N)
            [FrontNo,MaxFNo]     = NDSort(Population.objs,N);
            Next                 = false(1,length(FrontNo));
            Next(FrontNo<MaxFNo) = true;
            
            CrowdDis = CrowdingDistance(Population.objs,FrontNo);
            Last     = find(FrontNo==MaxFNo);
            [~,Rank] = sort(CrowdDis(Last),'descend');
            Popsize  = min(N,size(Population,2));
            Next(Last(Rank(1:Popsize-sum(Next)))) = true;
            
            Population = Population(Next);
            FrontNo    = FrontNo(Next);
            CrowdDis   = CrowdDis(Next);
        end
        
        %% 7. GA Half
        function Offspring = GA_half(Global,Parent)
            [proC,disC,proM,disM] = deal(1,20,1,20);
            Parent  = Parent.decs;
            Parent1 = Parent(1:floor(end/2),:);
            Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
            [N,D]   = size(Parent1);
            
            beta          = zeros(N,D);
            mu            = rand(N,D);
            beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
            beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
            beta          = beta.*(-1).^randi([0,1],N,D);
            beta(rand(N,D)<0.5)              = 1;
            beta(repmat(rand(N,1)>proC,1,D)) = 1;
            Offspring     = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
            
            Lower = repmat(Global.lower,N,1);
            Upper = repmat(Global.upper,N,1);
            Site  = rand(N,D) < proM/D;
            mu    = rand(N,D);
            temp  = Site & mu<=0.5;
            
            Offspring       = min(max(Offspring,Lower),Upper);
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                              (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
            temp            = Site & mu>0.5; 
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                              (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
        end
        
        %% 8. Create Groups
        function [outIndexList,numberOfGroups] = create_groups(Problem,numberOfGroups,xPrime,method)
            switch method
                case 1 %linear grouping
                    varsPerGroup = floor(Problem.D/numberOfGroups);
                    outIndexList = [];
                    for i = 1 : numberOfGroups-1
                       outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
                    end
                    outIndexList = [outIndexList, ones(1,Problem.D-size(outIndexList,2)).*numberOfGroups];
                case 2 %orderByValueGrouping
                    varsPerGroup = floor(Problem.D/numberOfGroups);
                    vars         = xPrime.dec;
                    [~,I]        = sort(vars);
                    outIndexList = ones(1,Problem.D);
                    for i = 1 : numberOfGroups-1
                       outIndexList(I(((i-1)*varsPerGroup)+1:i*varsPerGroup)) = i;
                    end
                    outIndexList(I(((numberOfGroups-1)*varsPerGroup)+1:end)) = numberOfGroups;
                case 3 %random Grouping
                    varsPerGroup = floor(Problem.D/numberOfGroups);
                    outIndexList = [];
                    for i = 1 : numberOfGroups-1
                       outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
                    end
                    outIndexList = [outIndexList, ones(1,Problem.D-size(outIndexList,2)).*numberOfGroups];
                    outIndexList = outIndexList(randperm(length(outIndexList)));
                case 4 %up or down groups
                    outIndexList = ones(1,Problem.D);
                    xPrimeVars   = xPrime.decs;
                    xPrimeObjs   = xPrime.objs;
                    for i = 1 : Problem.D
                        newSolVars    = xPrime.decs;
                        newSolVars(i) = xPrimeVars(i)*1.05;
                        newSol        = Problem.Evaluation(newSolVars);
                        newSolObjs    = newSol.objs;
                        if newSolObjs(1) < xPrimeObjs(1)
                            outIndexList(i) = 2;
                        end
                    end
                    numberOfGroups = 2;
            end
        end
        
        %% 9. Optimise By MOEAD
        function Population = optimise_by_MOEAD(GlobalDummy,Population,W,evaluations,isDummy)
            maximum = WOF_Utils.current_evaluations(GlobalDummy, isDummy) + evaluations;
            T       = max(ceil(GlobalDummy.N/10),2);
            B       = pdist2(W,W);
            [~,B]   = sort(B,2);
            B       = B(:,1:T);
            
            Z = min(Population.objs,[],1);
            g = zeros(GlobalDummy.N);
            for i = 1 : GlobalDummy.N
                g(i,:) = max(repmat(abs(Population(i).obj-Z),GlobalDummy.N,1)./W,[],2)';
            end
            [~,rank]  = sort(g,2);
            associate = zeros(1,GlobalDummy.N);
            for i = 1 : GlobalDummy.N
                x = find(~associate(rank(i,:)),1);
                associate(rank(i,x)) = i;
            end
            Population = Population(associate);
            
            while WOF_Utils.current_evaluations(GlobalDummy, isDummy) < maximum
                for i = 1 : GlobalDummy.N
                    P = B(i,randperm(size(B,2)));
                    if isDummy == true
                        NewDec    = WOF_Utils.GA_half(GlobalDummy, Population(P(1:2)));
                        Offspring = WOF_WeightIndividual(NewDec,GlobalDummy);
                    else
                        Offspring = OperatorGAhalf(GlobalDummy,Population(P(1:2)));
                    end
                    
                    Z    = min(Z,Offspring.obj);
                    type = 1;
                    switch type
                        case 1
                            normW   = sqrt(sum(W(P,:).^2,2));
                            normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                            normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                            CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                            CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                            g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                            g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                        case 2
                            g_old = max(abs(Population(P).objs-repmat(Z,T,1)).*W(P,:),[],2);
                            g_new = max(repmat(abs(Offspring.obj-Z),T,1).*W(P,:),[],2);
                        case 3
                            Zmax  = max(Population.objs,[],1);
                            g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                            g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                        case 4
                            g_old = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2);
                            g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2);
                    end
                    Population(P(g_old>=g_new)) = Offspring;
                end
            end
        end
        
        %% 9.1 Helper for MOEAD
        function e = current_evaluations(GlobalDummy, isDummy)
            if isDummy == true  
                e = GlobalDummy.Global.FE;
            else
                e = GlobalDummy.FE;
            end
        end
        
        %% 10. Select xPrimes
        function weightIndList = select_xprimes(input,amount, method)
            inputSize = size(input,2);
            switch method 
                case 1 %largest Crowding Distance from first front
                    inFrontNo     = NDSort(input.objs,inf);
                    weightIndList = [];
                    i             = 1;
                    if inputSize < amount
                        weightIndList = input;
                    else
                        while size(weightIndList,2) < amount 
                            left          = amount - size(weightIndList,2);
                            theFront      = inFrontNo == i;
                            newPop        = input(theFront);
                            FrontNo       = NDSort(newPop.objs,inf);
                            CrowdDis      = CrowdingDistance(newPop.objs,FrontNo);
                            [~,I]         = sort(CrowdDis,'descend');
                            until         = min(left,size(newPop,2));
                            weightIndList = [weightIndList,newPop(I(1:until))];
                            i             = i + 1;
                        end
                    end
                case 2 %tournament selection by front and CD
                    FrontNo       = NDSort(input.objs,inf);
                    CrowdDis      = CrowdingDistance(input.objs,FrontNo);
                    weightIndList = input(TournamentSelection(2,amount,FrontNo,-CrowdDis));
                case 3 % first m+1 by reference lines + fill with random
                    objValues     = input.objs;
                    m             = size(objValues,2);
                    weightIndList = [];
                    for i = 1 : m
                        vec           = zeros(1,m);
                        vec(1,i)      = 1;
                        angles        = pdist2(vec,real(objValues),'cosine');
                        [~,minIndex]  = min(angles);
                        weightIndList = [weightIndList,input(minIndex)];
                    end
                    if size(weightIndList,2) < amount
                        vec           = ones(1,m);
                        angles        = pdist2(vec,real(objValues),'cosine');
                        [~,minIndex]  = min(angles);
                        weightIndList = [weightIndList,input(minIndex)];
                    end
                    while size(weightIndList,2) < amount
                        ind           = input(randi([1 inputSize],1,1));
                        weightIndList = [weightIndList,ind];
                    end
            end
        end
        
        %% 11. Transformation Function
        function value = transformation_function(xPrime,weight,maxVal,minVal,method)
            value = xPrime;
            switch method
                case 1 %multiplication
                    value = xPrime*weight;
                case 2 %p-value
                    pWert = 0.2;
                    value = xPrime+pWert*(weight-1.0)*(maxVal-minVal);
                case 3 %interval
                    if weight > 1.0
                        weight   = weight - 1.0;
                        interval = maxVal - xPrime;
                        value    = xPrime + weight * interval;
                    else
                        interval = xPrime - minVal;
                        value    = minVal + weight * interval;
                    end           
            end
            if value < minVal
               value = minVal;
            elseif value > maxVal
               value = maxVal;
            end
        end
        
        %% 12. Transformation Function Matrix Form
        function value = transformation_function_matrix_form(xPrime,weight,maxVal,minVal,method)
            value = xPrime;
            switch method
                case 1 %multiplication
                    value = xPrime*weight;
                case 2 %p-value
                    pWert = 0.2;
                    value = xPrime+pWert*(weight-1.0)*(maxVal-minVal);
                case 3 %interval
                    interval = xPrime - minVal;
                    value    = minVal + weight .* interval;
                    interval = maxVal - xPrime;
                    value(weight > 1.0) = xPrime(weight > 1.0) + (weight(weight > 1.0)-1.0) .* interval(weight > 1.0); 
            end
            if value < minVal
               value = minVal;
            elseif value > maxVal
               value = maxVal;
            end
        end
    end
end
```

### `WOF_WeightIndividual.m`
```matlab
classdef WOF_WeightIndividual < handle
% WOF_WeightIndividual - The class of an individual used in WOF to store
% weight variables. 

% ----------------------------------------------------------------------- 
%  Copyright (C) 2020 Heiner Zille
%
%  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
%  International License. (CC BY-NC-SA 4.0). To view a copy of this license, 
%  visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or see the 
%  pdf-file "License-CC-BY-NC-SA-4.0.pdf" that came with this code. 
%
%  You are free to: 
%  * Share ? copy and redistribute the material in any medium or format
%  * Adapt ? remix, transform, and build upon the material 
%  Under the following terms:
%  * Attribution ? You must give appropriate credit, provide a link to the 
%     license, and indicate if changes were made. You may do so in any reasonable 
%     manner, but not in any way that suggests the licensor endorses you or your use.
%  * NonCommercial ? You may not use the material for commercial purposes.
%  * ShareAlike ? If you remix, transform, or build upon the material, you must 
%    distribute your contributions under the same license as the original.
%  * No additional restrictions ? You may not apply legal terms or technological 
%    measures that legally restrict others from doing anything the license permits.
% 
%  Author of this Code: 
%   Heiner Zille <heiner.zille@ovgu.de> or <heiner.zille@gmail.com>
%
%  This code is based on the following publications:
% 
%  1) Heiner Zille 
%     "Large-scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art"  
%     PhD Thesis, Otto von Guericke University Magdeburg, 2019 
%     http://dx.doi.org/10.25673/32063 
% 
%  2) Heiner Zille and Sanaz Mostaghim
%     "Comparison Study of Large-scale Optimisation Techniques on the LSMOP Benchmark Functions"  
%     IEEE Symposium Series on Computational Intelligence (SSCI), IEEE, Honolulu, Hawaii, November 2017
%     https://ieeexplore.ieee.org/document/8280974 
% 
%  3) Heiner Zille, Hisao Ishibuchi, Sanaz Mostaghim and Yusuke Nojima
%     "A Framework for Large-scale Multi-objective Optimization based on Problem Transformation"
%     IEEE Transactions on Evolutionary Computation, Vol. 22, Issue 2, pp. 260-275, April 2018.
%     http://ieeexplore.ieee.org/document/7929324
%  
%  4) Heiner Zille, Hisao Ishibuchi, Sanaz Mostaghim and Yusuke Nojima
%     "Weighted Optimization Framework for Large-scale Mullti-objective Optimization"
%     Genetic and Evolutionary Computation Conference (GECCO), ACM, Denver, USA, July 2016
%     http://dl.acm.org/citation.cfm?id=2908979
%
%  This file is intended to work with the PlatEMO framework version 2.5. 
%  Date of publication of this code: 06.04.2020 
%  Last Update of this code: 06.04.2020
%  A newer version of this algorithm may be available. Please contact the author 
%  or see http://www.ci.ovgu.de/Research/Codes.html. 
%
% The files may have been modified in Feb 2021 by the authors of the Platemo framework to work with the Platemo 3.0 release. 
% ----------------------------------------------------------------------- 
% This file is derived from its original version containied in the PlatEMO 
% framework. 
% -----------------------------------------------------------------------
    
    properties(SetAccess = private)
        dec;        % Decision variables of the individual
        obj;        % Objective values of the individual
        con;        % Constraint values of the individual
        add;        % Additional properties of the individual
        ind;        % the actual individual to extract later
    end
    methods
        %% Constructor
        function obj = WOF_WeightIndividual(variables, GlobalDummy, addValues)
            
            if nargin > 0
                xPrimeVars = GlobalDummy.xPrime.dec;
                xPrimeSize = size(xPrimeVars,2);

                obj = WOF_WeightIndividual;
                
                % Set the infeasible decision variables to boundary values
                variables  = max(min(variables,GlobalDummy.upper),GlobalDummy.lower);


                
                x = WOF_Utils.transformation_function_matrix_form(xPrimeVars,variables(GlobalDummy.G),GlobalDummy.xPrimeupper,GlobalDummy.xPrimelower, GlobalDummy.psi);


                Problem = WOF_WeightIndividual.Current();
                obj.dec = variables;
                obj.ind = Problem.Evaluation(x);
                obj.obj = obj.ind.obj;
                obj.con = obj.ind.con;
            
            
                if nargin > 2
                    CallStack = dbstack();
                    Field     = CallStack(2).name;
                    obj.add.(Field) = addValues;
                end
            end
            
        end
        %% Get the matrix of decision variables of the population
        function value = decs(obj)
        %decs - Get the matrix of decision variables of the population
        %
        %   A = obj.decs returns the matrix of decision variables of the
        %   population obj, where obj is an array of INDIVIDUAL objects.
        
            value = cat(1,obj.dec);
        end
        %% Get the matrix of objective values of the population
        function value = objs(obj)
        %objs - Get the matrix of objective values of the population
        %
        %   A = obj.objs returns the matrix of objective values of the
        %   population obj, where obj is an array of INDIVIDUAL objects.
        
            value = cat(1,obj.obj);
        end
        %% Get the matrix of constraint values of the population
        function value = cons(obj)
        %cons - Get the matrix of constraint values of the population
        %
        %   A = obj.cons returns the matrix of constraint values of the
        %   population obj, where obj is an array of INDIVIDUAL objects.
        
            value = cat(1,obj.con);
        end
        %% Get the matrix of additional property of the population
        function value = adds(obj,addValue)
        %adds - Get the matrix of additional property values of the population
        %
        %   A = obj.adds(AddProper) returns the matrix of the values of the
        %   additional property of the INDIVIDUAL objects obj. The name of
        %   the additional property is same to the function name of the
        %   caller, that is, the values of one additional property of the
        %   individuals can only be obtained by the function which created
        %   them. If any individual in obj does not contain the specified
        %   additional property, assign it a default value specified in
        %   AddProper.
        
            CallStack = dbstack();
            Field     = CallStack(2).name;
            value     = zeros(length(obj),size(addValue,2));
            for i = 1 : length(obj)
                if ~isfield(obj(i).add,Field)
                    obj(i).add.(Field) = addValue(i,:);
                end
                value(i,:) = obj(i).add.(Field);
            end
        end
    end
    methods(Static, Sealed)
        function obj = Current(obj)
        %Current - Get or set the current PROBLEM object.
        
            persistent Problem;
            if nargin > 0
                Problem = obj;
            end
            if nargout > 0
                obj = Problem;
            end
        end
    end
end
```
