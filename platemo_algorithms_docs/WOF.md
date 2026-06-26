# WOF

**Tags**: <2018> <multi> <real/integer> <large/none>

## Description
Weighted optimization framework

## Reference
H. Zille, H. Ishibuchi, S. Mostaghim, and Y. Nojima. A framework for large-scale multiobjective optimization based on problem transformation. IEEE Transactions on Evolutionary Computation, 2018, 22(2): 260-275.

## Source Code

### `WOF.m`
```matlab
classdef WOF < ALGORITHM
% <2018> <multi> <real/integer> <large/none>
% Weighted optimization framework
% gamma         --- 4    --- Number of groups. Default = 4 
% groups        --- 2    --- Grouping method, 1 = linear, 2 = ordered, 3 = random. Default = ordered 
% psi           --- 3    --- Transformation function, 1 = Multiplication, 2 = P-Value, 3 = Interval. Default = Interval
% t1            --- 1000 --- Number of evaluations for original problem. Default = 1000
% t2            --- 500  --- Number of evaluations for transformed problem. Default = 500
% q             ---      --- The number of chosen solutions to do weight optimisation. If no value is specified, the default value is M+1
% delta         --- 0.5  --- The fraction of function evaluations to use for the alternating weight-optimisation phase. Default = 0.5
% optimiser     --- 1    --- Internal optimisation algorithm. 1 = SMPSO, 2 = MOEAD, 3 = NSGAII, 4 = NSGAIII. Default = SMPSO. Only used if the randomOptimisers parameter is set to 0.
% randomOptimisers --- 1 --- 1 = use random optimisers (SMPSO,MOEAD,NSGAII,NSGAIII) in first phase (defined by delta), and NSGAIII in second phase. 0 = use only specified optimiser. Default = 1 

%------------------------------- Reference --------------------------------
% H. Zille, H. Ishibuchi, S. Mostaghim, and Y. Nojima. A framework for
% large-scale multiobjective optimization based on problem transformation.
% IEEE Transactions on Evolutionary Computation, 2018, 22(2): 260-275.
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

    methods
        function main(Algorithm,Problem)
            %% Set the default parameters
            [gamma,groups,psi,t1,t2,q,delta,optimiser,randomOptimisers] = Algorithm.ParameterSet(4,2,3,1000,500,Problem.M+1,0.5,1,1);
            WOF_WeightIndividual.Current(Problem);
            
            % The size of the population of weight-Individuals
            transformedProblemPopulationSize = 10;

            % There are different methods to select the xPrime solutions. 
            % Three methods can be chosen. The first one uses the largest Crowding
            % Distance values from the first non-dominated front. The second one
            % uses a tournament selection on the population based on
            % Pareto-dominance and Crowding Distance. The third option is 
            % introduced in publication (1), see above, based on
            % reference directions for the first m+1 chosen solutions and selects
            % random solutions afterwards. If method 3 is chosen, q is always at
            % least Global.M 
            methodToSelectxPrimeSolutions = 3; 

            diceForType = false;
            if randomOptimisers == 1
                diceForType = true;
            end

            if optimiser == 2 || optimiser == 4 || diceForType == true
                [uniW,Problem.N] = UniformPoint(Problem.N,Problem.M);
            end

            restart = true;


            %% Generate random population
            Population = Problem.Initialization();
            Algorithm.NotTerminated(Population);

            %% Start the alternating optimisation 
            while Problem.FE < delta*Problem.maxFE

                if (diceForType == true)
                    optimiser = randomizeType(optimiser);
                end

                % In case the population is not full (because the number of non-dominated 
                % solutions returned in the previous iteration is smaller than the 
                % population size), it is filled with new solutions. 
                Population = WOFfillPopulation(Population, Problem);

                % Normal optimisation step for t1 evaluations
                if      optimiser == 4
                    Population = WOF_optimiseByNSGAIII(Problem, Population, uniW, t1, false);
                elseif  optimiser == 3
                    Population = WOF_optimiseByNSGAII(Problem, Population, t1, false);
                elseif  optimiser == 2
                    Population = WOF_optimiseByMOEAD(Problem, Population, uniW, t1, false);
                else
                    Population = WOF_optimiseBySMPSO(Problem, Population, t1, false);
                end
                Algorithm.NotTerminated(Population); 

                % Selection of xPrime solutions 
                xPrimeList = WOF_selectxPrimes(Population, q, methodToSelectxPrimeSolutions); 
                WList   = [];

                % do for each xPrime
                for c = 1:size(xPrimeList,2)
                    xPrime              = xPrimeList(c);

                    % create variable groups 
                    [G,gamma]                   = WOF_createGroups(Problem,gamma,xPrime,groups);

                    if (diceForType == true)
                        optimiser = randomizeType(optimiser);
                    end

                    % a dummy object is needed to simulate the global class. Its
                    % necessary to include this method into the Platemo
                    % framework. 
                    GlobalDummy         = WOFcreateGlobalDummy(gamma, xPrime, G, Problem, transformedProblemPopulationSize, psi,optimiser);

                    % Create initial population for the transformed problem
                    WeightPopulation    = WOFcreateInitialWeightPopulation(GlobalDummy.N, gamma, GlobalDummy);

                    % Optimise the transformed problem 
                    if      optimiser == 4
                        WeightPopulation    = WOF_optimiseByNSGAIII(GlobalDummy, WeightPopulation, GlobalDummy.uniW, t2-transformedProblemPopulationSize, true);
                    elseif  optimiser == 3
                        WeightPopulation    = WOF_optimiseByNSGAII(GlobalDummy, WeightPopulation, t2-transformedProblemPopulationSize, true);
                    elseif  optimiser == 2
                        WeightPopulation    = WOF_optimiseByMOEAD(GlobalDummy, WeightPopulation, GlobalDummy.uniW, t2-transformedProblemPopulationSize, true);
                    else 
                        WeightPopulation    = WOF_optimiseBySMPSO(GlobalDummy, WeightPopulation, t2-transformedProblemPopulationSize, true);
                    end

                    % Extract the population 
                    W                   = WOFextractPopulation(WeightPopulation, Problem, Population, G, psi, xPrime, q, methodToSelectxPrimeSolutions);
                    WList               = [WList,W];  
                end

                % Join populations. Duplicate solution (e.g. found in different
                % optimisation steps with different xPrimes) need to be removed. 
                Population          = WOFeliminateDuplicates([Population,WList]);
                Population          = WOFfillPopulation(Population, Problem);

                % Environmental Selection
                [Population,~,~]    = WOF_EnvironmentalSelection(Population,Problem.N);
                Algorithm.NotTerminated(Population);
            end

            %% Optimise until end for uniformity. 
            remainingEvaluations = Problem.maxFE-Problem.FE;

            if ~restart
                % in this case all remaining evaluations are used directly with the
                % secified algorithm
                if (diceForType == true)
                    optimiser = 4;
                end

                if      optimiser == 4
                    Population = WOF_optimiseByNSGAIII(Problem, Population, uniW, remainingEvaluations, false);
                elseif  optimiser == 3
                    Population = WOF_optimiseByNSGAII(Problem, Population, remainingEvaluations, false);
                elseif  optimiser == 2
                    Population = WOF_optimiseByMOEAD(Problem, Population, uniW, remainingEvaluations, false);
                else
                    Population = WOF_optimiseBySMPSO(Problem, Population, remainingEvaluations, false);
                end
            else
                % in this case the value for t1 is used further to do separate
                % chunks of optimisation. 
                while Algorithm.NotTerminated(Population)
                    Population = WOFfillPopulation(Population, Problem);

                    if (diceForType == true)
                        optimiser = 4;
                    end

                    if      optimiser == 4
                        Population = WOF_optimiseByNSGAIII(Problem, Population, uniW, t1, false);
                    elseif  optimiser == 3
                        Population = WOF_optimiseByNSGAII(Problem, Population, t1, false);
                    elseif  optimiser == 2
                        Population = WOF_optimiseByMOEAD(Problem, Population, uniW, t1, false);
                    else
                        Population = WOF_optimiseBySMPSO(Problem, Population, t1, false);
                    end
                end 
            end
        end
    end
end

function GlobalDummy = WOFcreateGlobalDummy(gamma, xPrime, G, Global, populationSize, psi,optimiser)
    % Creates a dummy object. Needed to simulate the global class. Its
    % necessary to include this method into the Platemo
    % framework. 
    GlobalDummy = {};
    GlobalDummy.lower       = zeros(1,gamma);
    GlobalDummy.upper       = ones(1,gamma).*2.0;
    if or(optimiser == 2,optimiser == 4)
        [uniW,GlobalDummy.N]    = UniformPoint(populationSize,Global.M);
        GlobalDummy.uniW        = uniW;
    else
        GlobalDummy.N           = populationSize;
    end
    GlobalDummy.xPrime      = xPrime;
    GlobalDummy.G           = G;
    GlobalDummy.psi         = psi;
    GlobalDummy.xPrimelower = Global.lower;
    GlobalDummy.xPrimeupper = Global.upper;
    GlobalDummy.isDummy     = true;
    GlobalDummy.Global      = Global;
end

function Population = WOFeliminateDuplicates(input)
    % Eliminates duplicates in the population
    [~,ia,~] = unique(input.objs,'rows');
    Population = input(ia);
end

function Population = WOFfillPopulation(input, Problem)
    % Fills the population with mutations in case its smaller than Global.N
    Population = input;
    theCurrentPopulationSize = size(input,2);
    if theCurrentPopulationSize < Problem.N
        amountToFill    = Problem.N-theCurrentPopulationSize;
        FrontNo         = NDSort(input.objs,inf);
        CrowdDis        = CrowdingDistance(input.objs,FrontNo);
        MatingPool      = TournamentSelection(2,amountToFill+1,FrontNo,-CrowdDis);
        Offspring       = OperatorGA(Problem,input(MatingPool));
        Population      = [Population,Offspring(1:amountToFill)];
    end
end

function WeightPopulation = WOFcreateInitialWeightPopulation(N, gamma, GlobalDummy)
    %creates an initial population for the transformed problem
    decs = rand(N,gamma).*2.0;
    WeightPopulation = [];
    for i = 1:N
        solution = WOF_WeightIndividual(decs(i,:),GlobalDummy);
        WeightPopulation = [WeightPopulation, solution];
    end
end

function W = WOFextractPopulation(WeightPopulation, Problem, Population, G, psi, xPrime, q, methodToSelectxPrimeSolutions)
    % Extracts a population of individuals for the original problem based
    % on the optimised weights. 
    % First a selection of M+1 Weight-Individuals is selected and apllied
    % to the whole population each. 
    % Second all Weight-Individuals are applied to the chosen xPrime
    % solution, since they are optimised for it. 
    
    % Step 1
    weightIndividualList = WOF_selectxPrimes(WeightPopulation, q, methodToSelectxPrimeSolutions);
    calc = size(weightIndividualList,2)*size(Population,2);

    PopDec1 = ones(calc,Problem.D);
    count = 1;
    for wi = 1:size(weightIndividualList,2)
        weightIndividual = weightIndividualList(wi);
        weightVars = weightIndividual.dec;
        
        for i = 1:size(Population,2)
            individualVars = Population(i).dec;
            
            x = WOF_transformationFunctionMatrixForm(individualVars,weightVars(G),Problem.upper,Problem.lower, psi);

            PopDec1(count,:) = x;
            count = count + 1;
        end
        
    end
    
    W1 = Problem.Evaluation(PopDec1);

    % Step 2
    PopDec2 = [];
    for wi = 1:size(WeightPopulation,2)
        weightIndividual = WeightPopulation(wi);
        weightVars = weightIndividual.dec;
        
            individualVars = xPrime.dec;
            x = 1:Problem.D;
            for j = 1:Problem.D
                x(j) = WOF_transformationFunction(individualVars(j),weightVars(G(j)),Problem.upper(j),Problem.lower(j), psi);   
            end
            PopDec2 = [PopDec2;x]; 
    end
    W2 = Problem.Evaluation(PopDec2);
    
    W = [W1,W2];
end

function optimiser = randomizeType(oldOptimiser)
    
    %randomize with all 4 optimisation algorithms
    optimiser = randi(4);
end
```

### `WOF_EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = WOF_EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II that is used inside WOF.
% This function is mostly identical to the original
% EnvironmentalSelection function from the NSGA-II algorithm. 

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
    
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Popsize = min(N,size(Population,2));
    Next(Last(Rank(1:Popsize-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `WOF_GA.m`
```matlab
function Offspring = WOF_GA(Global,Parent)
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

%GA for WOF

    %% Parameter setting
    
    [proC,disC,proM,disM] = deal(1,20,1,20);


    Parent = Parent.decs;

    
    Parent1 = Parent(1:floor(end/2),:);    
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);
    

    
    %% Genetic operators for real encoding
    % Simulated binary crossover
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];

    % Polynomial mutation
    Lower = repmat(Global.lower,2*N,1);
    Upper = repmat(Global.upper,2*N,1);
    Site  = rand(2*N,D) < proM/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    
end
```

### `WOF_GAhalf.m`
```matlab
function Offspring = WOF_GAhalf(Global,Parent)
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

%GAhalf for WOF

    %% Parameter setting
    
    [proC,disC,proM,disM] = deal(1,20,1,20);

    
    Parent = Parent.decs;
    
    
    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);

 
    
    %% Genetic operators for real encoding
    % Simulated binary crossover
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;

    % Polynomial mutation
    Lower = repmat(Global.lower,N,1);
    Upper = repmat(Global.upper,N,1);
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

### `WOF_NSGAIIEnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = WOF_NSGAIIEnvironmentalSelection(Population,N)
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
% framework. The original copyright disclaimer can be found below. 
% -----------------------------------------------------------------------

% The environmental selection of NSGA-II

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

### `WOF_NSGAIIIEnvironmentalSelection.m`
```matlab
function Population = WOF_NSGAIIIEnvironmentalSelection(Population,N,Z,Zmin)
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
% framework. The original copyright disclaimer can be found below. 
% -----------------------------------------------------------------------

% The environmental selection of NSGA-III

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,N-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Choose = LastSelection(PopObj1,PopObj2,K,Z,Zmin)
% Select part of the solutions in the last front

    PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
    [N,M]  = size(PopObj);
    N1     = size(PopObj1,1);
    N2     = size(PopObj2,1);
    NZ     = size(Z,1);

    %% Normalization
    % Detect the extreme points
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));
    end
    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes
    Hyperplane = PopObj(Extreme,:)\ones(M,1);
    a = 1./Hyperplane;
    if any(isnan(a))
        a = max(PopObj,[],1)';
    end
    % Normalization
    PopObj = PopObj./repmat(a',N,1);
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    % Associate each solution with its nearest reference point
    [d,pi] = min(Distance',[],1);

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = hist(pi(1:N1),1:NZ);
    
    %% Environmental selection
    Choose  = false(1,N2);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);
        Jmin = find(rho(Temp)==min(rho(Temp)));
        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose==0 & pi(N1+1:end)==j);
        % Then select one solution associated with this reference point
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(d(N1+I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end
```

### `WOF_SMPSO_operator.m`
```matlab
function NewParticles = WOF_SMPSO_operator(Problem,Particles,isDummy)
% Particle swarm optimization operator in WOF-SMPSO 

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
    
    Particles      = Particles([1:end,1:ceil(end/3)*3-end]);
    
    ParticlesDec   = Particles.decs;
    [N,D]          = size(ParticlesDec);
    ParticlesSpeed = Particles.adds(zeros(N,D));

    %% PSO
    ParticleDec   = ParticlesDec(1:N/3,:);
    ParticleSpeed = ParticlesSpeed(1:N/3,:);
    PBestDec      = ParticlesDec(N/3+1:N/3*2,:);
    GBestDec      = ParticlesDec(N/3*2+1:end,:);
    W  = repmat(unifrnd(0.1,0.5,N/3,1),1,D);
    r1 = repmat(rand(N/3,1),1,D);
    r2 = repmat(rand(N/3,1),1,D);
    C1 = repmat(unifrnd(1.5,2.5,N/3,1),1,D);
    C2 = repmat(unifrnd(1.5,2.5,N/3,1),1,D);
    NewSpeed = W.*ParticleSpeed + C1.*r1.*(PBestDec-ParticleDec) + C2.*r2.*(GBestDec-ParticleDec);
    phi      = max(4,C1+C2);
    NewSpeed = NewSpeed.*2./abs(2-phi-sqrt(phi.^2-4*phi));
    delta    = repmat((Problem.upper-Problem.lower)/2,N/3,1);
    NewSpeed = max(min(NewSpeed,delta),-delta);
    NewDec   = ParticleDec + NewSpeed;
    
    %% Deterministic back
    Lower  = repmat(Problem.lower,N/3,1);
    Upper  = repmat(Problem.upper,N/3,1);
    repair = NewDec < Lower | NewDec > Upper;
    NewSpeed(repair) = 0.001*NewSpeed(repair);
    NewDec = max(min(NewDec,Upper),Lower);
    
    %% Polynomial mutation
    disM  = 20;
    Site1 = repmat(rand(N/3,1)<0.15,1,D);
    Site2 = rand(N/3,D) < 1/D;
    mu    = rand(N/3,D);
    temp  = Site1 & Site2 & mu<=0.5;
    NewDec(temp) = NewDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(NewDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site1 & Site2 & mu>0.5; 
    NewDec(temp) = NewDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-NewDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

    if isDummy == true
        NewParticles = [];
        for i = 1:N/3
            NewParticles = [NewParticles, WOF_WeightIndividual(NewDec(i,:),Problem,NewSpeed(i,:))];
        end
    else
        NewParticles = Problem.Evaluation(NewDec,NewSpeed);
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


                
                x = WOF_transformationFunctionMatrixForm(xPrimeVars,variables(GlobalDummy.G),GlobalDummy.xPrimeupper,GlobalDummy.xPrimelower, GlobalDummy.psi);


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

### `WOF_createGroups.m`
```matlab
function [outIndexList,numberOfGroups] = WOF_createGroups(Problem,numberOfGroups,xPrime,method)
% Creates groups of the varibales. Three diffeent methods can be
% chosen. The first one uses linear groups, the second orders variables
% by absolute values, the third is a random grouping. For more
% information about these mechanisms see publications (1) and (3), see below. 
    
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
    
    switch method
        case 1 %linear grouping
            varsPerGroup = floor(Problem.D/numberOfGroups);
            outIndexList = [];
            for i = 1:numberOfGroups-1
               outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
            end
            outIndexList = [outIndexList, ones(1,Problem.D-size(outIndexList,2)).*numberOfGroups];
        case 2 %orderByValueGrouping
            varsPerGroup = floor(Problem.D/numberOfGroups);
            vars = xPrime.dec;
            [~,I] = sort(vars);
            outIndexList = ones(1,Problem.D);
            for i = 1:numberOfGroups-1
               outIndexList(I(((i-1)*varsPerGroup)+1:i*varsPerGroup)) = i;
            end
            outIndexList(I(((numberOfGroups-1)*varsPerGroup)+1:end)) = numberOfGroups;
        case 3 %random Grouping
            varsPerGroup = floor(Problem.D/numberOfGroups);
            outIndexList = [];
            for i = 1:numberOfGroups-1
               outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
            end
            outIndexList = [outIndexList, ones(1,Problem.D-size(outIndexList,2)).*numberOfGroups];
            outIndexList = outIndexList(randperm(length(outIndexList)));
        case 4 %up or down groups
            outIndexList = ones(1,Problem.D);
            xPrimeVars = xPrime.decs;
            xPrimeObjs = xPrime.objs;
            for i = 1 : Problem.D
                newSolVars = xPrime.decs;
                newSolVars(i) = xPrimeVars(i)*1.05;
                newSol = Problem.Evaluation(newSolVars);
                newSolObjs = newSol.objs;
                if newSolObjs(1) < xPrimeObjs(1)
                    outIndexList(i) = 2;
                end
            end
            numberOfGroups = 2;
    end
end
```

### `WOF_optimiseByMOEAD.m`
```matlab
function Population = WOF_optimiseByMOEAD(GlobalDummy,Population,W,evaluations,isDummy)
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

% MOEA/D

    maximum = currentEvaluations(GlobalDummy, isDummy) + evaluations;
    
    T = max(ceil(GlobalDummy.N/10),2);

    %% Detect the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);
    
    
    %% Associate each subproblem with one solution
    % The ideal point
    Z = min(Population.objs,[],1);
    % The value of each solution on each subproblem (modified Tchebycheff approach)
    g = zeros(GlobalDummy.N);
    for i = 1 : GlobalDummy.N
        g(i,:) = max(repmat(abs(Population(i).obj-Z),GlobalDummy.N,1)./W,[],2)';
    end
    [~,rank] = sort(g,2);
    % The index of solution which each subproblem associated with
    associate = zeros(1,GlobalDummy.N);
    for i = 1 : GlobalDummy.N
        x = find(~associate(rank(i,:)),1);
        associate(rank(i,x)) = i;
    end
    Population = Population(associate);
    
    %% Optimization
    while currentEvaluations(GlobalDummy, isDummy) < maximum
        % For each solution
        for i = 1 : GlobalDummy.N
            % Choose the parents
            P = B(i,randperm(size(B,2)));

            % Generate an offspring
            if isDummy == true
                NewDec = WOF_GAhalf(GlobalDummy, Population(P(1:2)));
                Offspring = WOF_WeightIndividual(NewDec,GlobalDummy);
        
            else
                Offspring = OperatorGAhalf(GlobalDummy,Population(P(1:2)));
            end
            
            % Update the ideal point
            Z = min(Z,Offspring.obj);

            type = 1;
            % Update the neighbours
            switch type
                case 1
                    % PBI approach
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
                    g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                    g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                case 2
                    % Tchebycheff approach
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1)).*W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z),T,1).*W(P,:),[],2);
                case 3
                    % Tchebycheff approach with normalization
                    Zmax  = max(Population.objs,[],1);
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z)./(Zmax-Z),T,1).*W(P,:),[],2);
                case 4
                    % Modified Tchebycheff approach
                    g_old = max(abs(Population(P).objs-repmat(Z,T,1))./W(P,:),[],2);
                    g_new = max(repmat(abs(Offspring.obj-Z),T,1)./W(P,:),[],2);
            end
            Population(P(g_old>=g_new)) = Offspring;
        end
    end
end

function e = currentEvaluations(GlobalDummy, isDummy)
    if isDummy == true  
        e = GlobalDummy.Global.FE;
    else
        e = GlobalDummy.FE;
    end
end
```

### `WOF_optimiseByNSGAII.m`
```matlab
function Population = WOF_optimiseByNSGAII(GlobalDummy, Population, evaluations, isDummy)
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


    maximum = currentEvaluations(GlobalDummy, isDummy) + evaluations;

    [~,FrontNo,CrowdDis] = WOF_NSGAIIEnvironmentalSelection(Population,GlobalDummy.N);

    %% Optimization
    while currentEvaluations(GlobalDummy, isDummy) < maximum
        MatingPool = TournamentSelection(2,GlobalDummy.N,FrontNo,-CrowdDis);     
        if isDummy == true
            NewDec      = WOF_GA(GlobalDummy, Population(MatingPool));
            numberOfNewSolutions = size(NewDec,1);
            Offspring = [];
            for j = 1:numberOfNewSolutions
                Offspring   = [Offspring, WOF_WeightIndividual(NewDec,GlobalDummy)];
            end
        else
            Offspring   = OperatorGA(GlobalDummy, Population(MatingPool));
        end
        [Population,FrontNo,CrowdDis] = WOF_NSGAIIEnvironmentalSelection([Population,Offspring],GlobalDummy.N);

    end
end

function e = currentEvaluations(GlobalDummy, isDummy)
    if isDummy == true  
        e = GlobalDummy.Global.FE;
    else
        e = GlobalDummy.FE;
    end
end
```

### `WOF_optimiseByNSGAIII.m`
```matlab
function Population = WOF_optimiseByNSGAIII(GlobalDummy,Population,Z,evaluations,isDummy)
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


    maximum = currentEvaluations(GlobalDummy, isDummy) + evaluations;

    Zmin         = min(Population(all(Population.cons<=0,2)).objs,[],1);    

    %% Optimization
    while currentEvaluations(GlobalDummy, isDummy) < maximum
        MatingPool = TournamentSelection(2,GlobalDummy.N,sum(max(0,Population.cons),2));
        if isDummy == true
            NewDec      = WOF_GA(GlobalDummy, Population(MatingPool));
            numberOfNewSolutions = size(NewDec,1);
            Offspring = [];
            for j = 1:numberOfNewSolutions
                Offspring   = [Offspring, WOF_WeightIndividual(NewDec,GlobalDummy)];
            end
        else
            Offspring   = OperatorGA(GlobalDummy,Population(MatingPool));
        end
        Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
        Population = WOF_NSGAIIIEnvironmentalSelection([Population,Offspring],GlobalDummy.N,Z,Zmin);
    end
end

function e = currentEvaluations(GlobalDummy, isDummy)
    if isDummy == true  
        e = GlobalDummy.Global.FE;
    else
        e = GlobalDummy.FE;
    end
end
```

### `WOF_optimiseBySMPSO.m`
```matlab
function Gbest = WOF_optimiseBySMPSO(GlobalDummy, inputPopulation, evaluations, isDummy)
% This function performs the optimisation by using the SMPSO
% methodology. 
% Additional functionality has been added and modified to make it
% applicable for normal Individuals as well as the transformed
% weight-variables. 

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

    %% Generate random population
    Population       = inputPopulation;
    Pbest            = Population;
    [Gbest,CrowdDis] = WOF_UpdateGbest(Population,GlobalDummy.N);
    
    maximum = currentEvaluations(GlobalDummy, isDummy) + evaluations;

    %% Optimization
    while currentEvaluations(GlobalDummy, isDummy) < maximum
        Population       = WOF_SMPSO_operator(GlobalDummy, [Population,Pbest,Gbest(TournamentSelection(2,GlobalDummy.N,-CrowdDis))], isDummy);
        [Gbest,CrowdDis] = WOF_UpdateGbest([Gbest,Population],GlobalDummy.N);
        Pbest            = WOF_UpdatePbest(Pbest,Population);
    end

end

function e = currentEvaluations(GlobalDummy, isDummy)
    if isDummy == true  
        e = GlobalDummy.Global.FE;
    else
        e = GlobalDummy.FE;
    end
end

function Pbest = WOF_UpdatePbest(Pbest,Population)
    % Update the local best position of each particle
    replace        = ~all(Population.objs>=Pbest.objs,2);
    Pbest(replace) = Population(replace);
end

function [Gbest,CrowdDis] = WOF_UpdateGbest(Gbest,N)
    % Update the global best set
    Gbest    = Gbest(NDSort(Gbest.objs,1)==1);
    CrowdDis = CrowdingDistance(Gbest.objs);
    [~,rank] = sort(CrowdDis,'descend');
    Gbest    = Gbest(rank(1:min(N,length(Gbest))));
    CrowdDis = CrowdDis(rank(1:min(N,length(Gbest))));
end
```

### `WOF_selectxPrimes.m`
```matlab
function weightIndList = WOF_selectxPrimes(input,amount, method)
% Implements the selection of the x' solutions in WOF-SMPSO. 
% Three methods can be chosen. The first one uses the largest Crowding
% Distance values from the first non-dominated front. The second one
% uses a tournament selection on the population based on
% Pareto-dominance and Crowding Distance. The third option is 
% introduced in publication (2), see above, based on
% reference directions for the first m+1 chosen solutions and selects
% random solutions afterwards.    

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

    
    inputSize = size(input,2);
    switch method 
        case 1 %largest Crowding Distance from first front
            inFrontNo    = NDSort(input.objs,inf);
            weightIndList = [];
            i = 1;
            if inputSize < amount
                weightIndList = input;
            else
                while size(weightIndList,2) < amount 
                    left = amount - size(weightIndList,2);
                    theFront = inFrontNo == i;
                    newPop = input(theFront);
                    FrontNo    = NDSort(newPop.objs,inf);
                    CrowdDis   = WOF_CrowdingDistance(newPop.objs,FrontNo);
                    [~,I] = sort(CrowdDis,'descend');
                    until=min(left,size(newPop,2));
                    weightIndList = [weightIndList,newPop(I(1:until))];
                    i=i+1;
                end
            end
        case 2 %tournament selection by front and CD
            FrontNo    = NDSort(input.objs,inf);
            CrowdDis   = CrowdingDistance(input.objs,FrontNo);
            weightIndList = input(TournamentSelection(2,amount,FrontNo,-CrowdDis));
        case 3 % first m+1 by reference lines + fill with random
            objValues = input.objs;
            m = size(objValues,2);
            weightIndList = [];
            for i = 1:m
                vec = zeros(1,m);
                vec(1,i) = 1;
                angles = pdist2(vec,real(objValues),'cosine');
                [minAngle,minIndex] = min(angles);
                weightIndList = [weightIndList,input(minIndex)];
            end
            if size(weightIndList,2) < amount
                vec = ones(1,m);
                angles = pdist2(vec,real(objValues),'cosine');
                [minAngle,minIndex] = min(angles);
                weightIndList = [weightIndList,input(minIndex)];
            end
            while size(weightIndList,2) < amount
                ind = input(randi([1 inputSize],1,1));
                weightIndList = [weightIndList,ind];
            end
    end
end
```

### `WOF_transformationFunction.m`
```matlab
function value = WOF_transformationFunction(xPrime,weight,maxVal,minVal,method)
% Implements the transformation functions used in WOF-SMPSO. Three 
% methods can be chosen. The p-Value and Multiplication-transformations
% have been introduced in publication (3), see above. 
% The Intervall-transformation (parameter free) has been introduced in 
% publication (2), see above. The interval-intersection method from (3)
% is currently not implemented. 

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


    value = xPrime;
    switch method
        case 1 %multiplication
            value = xPrime*weight;
        case 2 %p-value
            pWert = 0.2;
            value = xPrime+pWert*(weight-1.0)*(maxVal-minVal);
        case 3 %interval
            if weight > 1.0
                weight = weight - 1.0;
                interval = maxVal - xPrime;
                value = xPrime + weight * interval;
            else
                interval = xPrime - minVal;
                value = minVal + weight * interval;
            end           
    end
    
    %do repair
    if value < minVal
       value = minVal;
    elseif value > maxVal
       value = maxVal;
    end
    
end
```

### `WOF_transformationFunctionMatrixForm.m`
```matlab
function value = WOF_transformationFunctionMatrixForm(xPrime,weight,maxVal,minVal,method)
% Implements the transformation functions used in WOF. Three 
% methods can be chosen. The p-Value and Multiplication-transformations
% have been introduced in publication (3), see the WOF.m main file. 
% The Intervall-transformation (parameter free) has been introduced in 
% publication (2), see above. The interval-intersection method from (3)
% is currently not implemented. 

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



    value = xPrime;
    switch method
        case 1 %multiplication
            value = xPrime*weight;
        case 2 %p-value
            pWert = 0.2;
            value = xPrime+pWert*(weight-1.0)*(maxVal-minVal);
        case 3 %interval
            interval = xPrime - minVal;
            value = minVal + weight .* interval;
            interval = maxVal - xPrime;
            value(weight > 1.0) = xPrime(weight > 1.0) + (weight(weight > 1.0)-1.0) .* interval(weight > 1.0); 
    end
    
    %do repair
    if value < minVal
       value = minVal;
    elseif value > maxVal
       value = maxVal;
    end
    
end
```
