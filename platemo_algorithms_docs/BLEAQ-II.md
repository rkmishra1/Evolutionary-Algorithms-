# BLEAQ-II

**Tags**: <2020> <multi> <real> <constrained/none> <bilevel>

## Description
Bilevel evolutionary algorithm based on quadratic approximations II

## Reference
A. Sinha, Z. Lu, K. Deb, and P. Malo. Bilevel optimization based on iterative approximation of mappings. Journal of Heuristics, 2020, 26: 151-185.

## Source Code

### `BLEAQII.m`
```matlab
classdef BLEAQII < ALGORITHM
% <2020> <multi> <real> <constrained/none> <bilevel>
% Bilevel evolutionary algorithm based on quadratic approximations II

%------------------------------- Reference --------------------------------
% A. Sinha, Z. Lu, K. Deb, and P. Malo. Bilevel optimization based on
% iterative approximation of mappings. Journal of Heuristics, 2020, 26:
% 151-185.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            ulPopDec = unifrnd(repmat(Problem.lower(1:Problem.DU),Problem.N,1),repmat(Problem.upper(1:Problem.DU),Problem.N,1));
            % Doing lower level search for population member
            for i = 1 : size(ulPopDec,1)
               [llPopDec(i,:),tag.ulPop(i)] = llSearch(Problem,ulPopDec(i,:),[],[],[]);
            end
            llLocalSearchPopSize = max([4 Problem.N/10]);
            [~,centroids] = kmeans(llPopDec,llLocalSearchPopSize,'EmptyAction','singleton');
            for i = 1 : size(ulPopDec,1)
               [llPopDec(i,:),tag.ulPop(i)] = llSearch(Problem,ulPopDec(i,:),[],[], centroids);
            end
            Population           = Problem.Evaluation([ulPopDec,llPopDec]);
            archive              = struct('tag1',[],'tag0',[]);
            archive.tag1 = archiveUpdate(Problem, archive.tag1, Population);
            alphaStoppingInitial = sum(var([ulPopDec(tag.ulPop==1,:), llPopDec(tag.ulPop==1,:)]))/Problem.D;
            maxError             = 1e-4;
            StoppingCriteria     = 0;
            gen = 0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                gen = gen+1;
                % Upper level optimization
                MatingPool = TournamentSelection(2,3,CalFitness(Problem.C,Population));
                ParentDec  = Population(MatingPool).decs;
                ulOffDec   = OperatorPCX(ParentDec(:,1:Problem.DU),Problem.lower(1:Problem.DU),Problem.upper(1:Problem.DU));
                
                % Lower level optimization
                for i = 1 : size(ulOffDec,1)
                    if (sum(tag.ulPop==1) < Problem.N/2) || (length(archive.tag1) < (Problem.DU+1)*(Problem.DU+2)/2+3*(Problem.DU))
                        % Optimization
                        [~,closest]   = min(pdist2(ulOffDec,ParentDec(:,1:Problem.DU)),[],2);
                        [llOffDec(i,:), tag.Offsprings(i)] = llSearch(Problem,ulOffDec(i,:),ParentDec(closest(i),Problem.DU+1:end),[],[]);  
                    else
                        % Approximation
                        [psiMapping,phiMapping,lies]    = getMappings(Problem,ulOffDec(i,:),archive.tag1);
                        [llOffDec(i,:),sumMSE,validMSE] = getLowerLevelVariableFromMapping(ulOffDec(i,:),psiMapping,phiMapping,Problem,archive);
                        if lies==1 && validMSE<maxError
                            tag.Offsprings(i) = 1;
                        else
                            tag.Offsprings(i) = 0;
                        end
                    end
                end
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);
                
                % Environment selection for upper population
                Population       = EnvironmentalSelection(Problem,Population,Offspring);
                
                PopDec           = Population.decs;
                llmemberVariance = var(PopDec(:,Problem.DU+1:end));
                if sum(tag.Offsprings == 1) > 0 
                    archive.tag1 = archiveUpdate(Problem, archive.tag1, Offspring(tag.Offsprings==1));
                end
                if sum(tag.Offsprings == 0) > 0
                    archive.tag0 = archiveUpdate(Problem, archive.tag0, Offspring(tag.Offsprings==0));
                end
                
                % Local search in EliteIndiv
                Fitness    = CalFitness(Problem.C,Population);
                if sum(tag.ulPop==1)==0
                    [~,Index] = min(Fitness);
                else
                    [~,index]  = min(Fitness(tag.ulPop==1));
                    I          = find(tag.ulPop==1);
                    Index      = I(index);
                end
                initialIndivLS = Population(Index);
                PopDec        = Population.decs;
                llPopDec      = PopDec(:,Problem.DU+1:end); 
                alphaStopping = sum(var([ulPopDec(tag.ulPop==1,:),llPopDec(tag.ulPop==1,:)]))/Problem.D;
                alphaStopping = alphaStopping/alphaStoppingInitial;
                if alphaStopping < 1e-4
                    StoppingCriteria = 1;
                end
                [eliteIndivLS,llEliteIndivLS] = DoLocalSearch(Problem,initialIndivLS,archive,StoppingCriteria,gen);
                [llEliteIndivLS,           ~] = llSearch(Problem,eliteIndivLS,llEliteIndivLS,llmemberVariance,[]);
                eliteLS = Problem.Evaluation([eliteIndivLS,llEliteIndivLS]);
                % EnvironmentalSelection
                if CalFitness(Problem.C,eliteLS) > CalFitness(Problem.C,initialIndivLS)
                    eliteLS = initialIndivLS;
                end
                Population = EnvironmentalSelection(Problem,Population,eliteLS);
            end
        end
    end
end

function archive = archiveUpdate(Problem, archive, newData)

    archiveSize = ((Problem.DU+1)*(Problem.DU+2)/2+3*(Problem.DU))*10;
    archive = [archive, newData];
    
    if length(archive) > archiveSize
        archive(1) = [];
    end 
end
```

### `CalFitness.m`
```matlab
function Fitness = CalFitness(C,Population)
% Calculate the fitness of each solution in terms of a single level

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = Population.objs;
    PopCon = Population.cons;
    if any(isnan(PopObj(:,1)))  % Lower level
        PopObj = PopObj(:,2);
        PopCon = PopCon(:,C+1:end);
    else                        % Upper level
        PopObj = PopObj(:,1);
        PopCon = PopCon(:,1:C);
    end
    if isempty(PopCon)
        PopCon = zeros(size(PopObj,1),1);
    else
        PopCon = sum(max(0,PopCon),2);
    end
    Feasible = PopCon <= 0;
    Fitness  = Feasible.*PopObj + ~Feasible.*(PopCon+1e10);
end
```

### `DoLocalSearch.m`
```matlab
function [eliteIndivLS,llEliteIndivLS,localSearch] = DoLocalSearch(Problem,initialIndivLS,archive,StoppingCriteria,gen)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Flag.localSearch = 0; 
    Flag.recheck     = 0; 
    Flag.update      = 0; 
    Flag.replace     = 0;
    Flag.dontDoPhi   = 0;
    
    numberLowerLevelPointsTrain = (Problem.DU+1)*(Problem.DU+2)/2+2*(Problem.DU);
    numberLowerLevelPointsEval  = Problem.DU;
    minSizePsiLS = numberLowerLevelPointsTrain + numberLowerLevelPointsEval;
    minSizePhiLS = (Problem.DU+Problem.DL+1)*(Problem.DU+Problem.DL+2)/2 + 2*(Problem.DU) + Problem.DU;

    terminationFlag   = (Problem.FE > Problem.maxFE) || (StoppingCriteria == 1);
    frequency         = Problem.N;
    frequencyOffsetLS = frequency - 1;
    frequencyOffsetRC = floor(frequency/2 - 1);
    
    if length(archive.tag1) >= minSizePsiLS
        if terminationFlag == 1
            Flag.localSearch = 1; % Exact local search
        elseif mod(gen+frequencyOffsetLS,frequency) == 0
            Flag.localSearch = 2; % Default local search
        end
    end

    IndivDec     = initialIndivLS.decs;
    eliteIndiv   = IndivDec(:,1:Problem.DU);
    llEliteIndiv = IndivDec(:,Problem.DU+1:end);
    ulDimMax     = Problem.upper(:,1:Problem.DU);
    ulDimMin     = Problem.lower(:,1:Problem.DU);
    llDimMax     = Problem.upper(:,Problem.DU+1:end);
    llDimMin     = Problem.lower(:,Problem.DU+1:end);
    range        = 0.05;
    
    doExactLSFlag = Flag.localSearch;
    if length([archive.tag1,archive.tag0]) < minSizePhiLS
        Flag.dontDoPhi = 1;
    end
    
    % Obtain psi and phi mappings around eliteIndiv
    [psiMapping,phiMapping,~]   = getMappings(Problem,eliteIndiv,archive.tag1);
    functionLowerLevelVariables = psiMapping.function;
    functionLowerLevelObjective = phiMapping.function{:};
      
    localSearch.psiMSE = psiMapping.validMSE; localSearch.phiMSE = phiMapping.validMSE;
    % Adaptive selection Which mapping to pick during execution
    if (psiMapping.validMSE <= phiMapping.validMSE) || Flag.dontDoPhi == 1 || (rand <=0.5)% psi-mapping
        if doExactLSFlag == 1       % Run an exact local search
            localSearchMethod = 1;	% psi w/ EXACT obj. func.
        else% Run an approximate local search (default)
            localSearchMethod = 2;	% psi w/ Approx. obj. func.
        end
    else % phi-mapping
        if doExactLSFlag == 1
            localSearchMethod = 3;	% phi w/ EXACT obj. func.
        else
            localSearchMethod = 4;	% phi w/ Approx. obj. func.
        end       
    end
    
    % Data preparation
    archiveLS    = [archive.tag1,archive.tag0];
    archiveLSDec = archiveLS.decs;
    upper        = archiveLSDec(:,1:Problem.DU);
    for j = 1 : size(upper,1)
        distances(j) = sum((eliteIndiv - upper(j,:)).^2);
    end
    [~,I] = sort(distances);
    
    if localSearchMethod == 2
        archiveConsidered = (Problem.DU+1)*(Problem.DU+2)/2+2*(Problem.DU)+Problem.DU;
        % If run psi w/ approx. obj. func. 
        % need to approximate F(x) and G(x) with Tag 1 member
        archiveLSDec = archiveLS(I(1:archiveConsidered)).decs;
        archiveLSObj = archiveLS(I(1:archiveConsidered)).objs;
        archiveLSCon = archiveLS(I(1:archiveConsidered)).cons;
        archivePsi.upper = upper(I(1:archiveConsidered),:);
        archivePsi.lower = archiveLSDec(:,Problem.DU+1:end);
        archivePsi.functionValue        = archiveLSObj(:,1);
        archivePsi.equalityConstrVals   = [];
        archivePsi.inequalityConstrVals = archiveLSCon(:,1:Problem.C);
        approxPsi.function              = QuadApprox(archivePsi.functionValue, [archivePsi.upper]);
        if size(archivePsi.equalityConstrVals,2) ~= 0
	        for i = 1 : size(archivePsi.equalityConstrVals,2)  
                approxPsi.equalityConstr{i} = QuadApprox(archivePsi.equalityConstrVals(:,i), [archivePsi.upper]);
	        end
	    else
	        approxPsi.equalityConstr = [];
        end
	    if size(archivePsi.inequalityConstrVals,2) ~= 0
	        for i = 1 : size(archivePsi.inequalityConstrVals,2)
                approxPsi.inequalityConstr{i} = QuadApprox(archivePsi.inequalityConstrVals(:,i), [archivePsi.upper]);
	        end
	    else
	        approxPsi.inequalityConstr = [];
        end
    end
    
    if localSearchMethod == 4
        % If run phi w/ approx. obj. func. 
        % need to approximate F(x,y), G(x,y), f(x,y), g(x,y) w/ random
        % member
        archiveConsidered = (Problem.DU+Problem.DL+1)*(Problem.DU+Problem.DL+2)/2 + 2*(Problem.DU) + Problem.DU;
        archiveLSDec      = archiveLS(I(1:archiveConsidered)).decs;
        archiveLSObj      = archiveLS(I(1:archiveConsidered)).objs;
        archiveLSCon      = archiveLS(I(1:archiveConsidered)).cons;
        archivePhi.upper  = upper(I(1:archiveConsidered),:);
        archivePhi.lower  = archiveLSDec(:,Problem.DU+1:end);
        archivePhi.functionValue          = archiveLSObj(:,1);
        archivePhi.equalityConstrVals     = [];
        archivePhi.inequalityConstrVals   = archiveLSCon(:,1:Problem.C);
        archivePhi.llFunctionValue        = archiveLSObj(:,2);
        archivePhi.llEqualityConstrVals   = [];
        archivePhi.llInequalityConstrVals = archiveLSCon(:,Problem.C+1:end);
        approxPhi.function = QuadApprox(archivePhi.functionValue, [archivePhi.upper, archivePhi.lower]);        
		if ~isempty(archivePhi.equalityConstrVals)
		    for i = 1 : size(archivePhi.equalityConstrVals,2)		            
                approxPhi.equalityConstr{i} = QuadApprox(archivePhi.equalityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);
		    end
		else
		    approxPhi.equalityConstr = [];
		end

		if ~isempty(archivePhi.inequalityConstrVals)
		    for i = 1 : size(archivePhi.inequalityConstrVals,2)
               approxPhi.inequalityConstr{i} = QuadApprox(archivePhi.inequalityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);            
		    end
		else
		    approxPhi.inequalityConstr = [];
		end

		% Phi-function with Approximated F(x,y) and f(x,y)
        % Needs to approximate f(x,y), g(x,y) & h(x,y)
        approxPhi.llFunction = QuadApprox(archivePhi.llFunctionValue, [archivePhi.upper archivePhi.lower]);
 
        if ~isempty(archivePhi.llEqualityConstrVals)
            for i = 1 : size(archivePhi.llEqualityConstrVals,2)
                approxPhi.llEqualityConstr{i} = QuadApprox(archivePhi.llEqualityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);
            end
        else
            approxPhi.llEqualityConstr = [];
        end

        if ~isempty(archivePhi.llInequalityConstrVals)
            for i = 1 : size(archivePhi.llInequalityConstrVals,2)
                approxPhi.llInequalityConstr{i} = QuadApprox(archivePhi.llInequalityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);
            end
        else
            approxPhi.llInequalityConstr = [];
        end
    end

    if localSearchMethod == 1
        % Exact Psi-function approximation based local search 
        options = optimset('Algorithm','sqp','Display','off');
        [lb,ub] = createLocalSearchBound([eliteIndiv],[ulDimMin],[ulDimMax],range);
        [eliteIndivLS,~,EXITFLAG,OUTPUT] = fmincon(@(x) -approximatedFunctionPsi(x,llDimMin,llDimMax,functionLowerLevelVariables,Problem),[eliteIndiv],[],[],[],[],lb,ub,@(x) approximatedConstraintsPsi(x,llDimMin,llDimMax,functionLowerLevelVariables,Problem),options);
        llEliteIndivLS                   = llEliteIndiv;
        localSearch.method               = 'Psi';
        localSearch.termination          = EXITFLAG;
        localSearch.functionEvaluation   = OUTPUT.funcCount;
        return;
    end
    
    if localSearchMethod == 2
         options = optimset('Algorithm','sqp','Display','off');      
        % psi-mapping based local search w/ approximated obj. func. 
        lb = ulDimMin;
        ub = ulDimMax;
        [eliteIndivLS,~,EXITFLAG,OUTPUT] = fmincon(@(x) -approximatedFunction(x,approxPsi.function),[eliteIndiv],[],[],[],[],lb,ub,@(x) approximatedConstraints(x,approxPsi.equalityConstr,approxPsi.inequalityConstr),options);
        llEliteIndivLS                   = llEliteIndiv;
        localSearch.method               = 'Approx';
        localSearch.termination          = EXITFLAG;
        localSearch.functionEvaluation   = OUTPUT.funcCount;
        return;
    end
        
    if localSearchMethod == 3
    	% Exact Phi-function approximation based local search 
        options = optimset('Algorithm','sqp','Display','off');
        [lb,ub] = createLocalSearchBound([eliteIndiv llEliteIndiv],[ulDimMin llDimMin],[ulDimMax llDimMax],range);
        [eliteIndivFull,~,EXITFLAG,OUTPUT] = fmincon(@(x) -approximatedFunctionPhi(x,Problem),[eliteIndiv llEliteIndiv],[],[],[],[],lb,ub,@(x) approximatedConstraintsPhi(x,functionLowerLevelObjective,Problem),options);
        eliteIndivLS                   = eliteIndivFull(1:Problem.DU);
        llEliteIndivLS                 = eliteIndivFull(Problem.DU+1:end);
        localSearch.method             = 'Phi';
        localSearch.termination        = EXITFLAG;
        localSearch.functionEvaluation = OUTPUT.funcCount;
        return;
    end
    
    if localSearchMethod == 4
        % phi-mapping based local search w/ approximated obj. func.
        options = optimset('Algorithm','sqp','Display','off');
        lb      = [ulDimMin llDimMin];
        ub      = [ulDimMax llDimMax];
        [eliteIndivFull,~,EXITFLAG,OUTPUT] = fmincon(@(x) -approximatedFunction(x,approxPhi.function),[eliteIndiv llEliteIndiv],[],[],[],[],lb,ub,@(x) approximatedConstraintsPhi2(x,approxPhi.equalityConstr,approxPhi.inequalityConstr, approxPhi.llFunction, functionLowerLevelObjective, approxPhi.llEqualityConstr, approxPhi.llInequalityConstr, Problem.DU, Problem.DL),options);
        eliteIndivLS                       = eliteIndivFull(1:Problem.DU);
        llEliteIndivLS                     = eliteIndivFull(Problem.DU+1:end);
        localSearch.method                 = 'ApproxPhi';
        localSearch.termination            = EXITFLAG;
        localSearch.functionEvaluation     = OUTPUT.funcCount;    
        return;
    end
end

function functionValue = approximatedFunctionPsi(xu,llDimMin,llDimMax,psiFunction,Problem)
    for j = 1 : size(psiFunction,2)
        xl(j) = psiFunction{j}.constant + xu*psiFunction{j}.linear + xu*psiFunction{j}.sqmatrix*xu';
    end
    % check if predicted lower level optimal solution is outside the bound
    xl            = checkLimits(xl,llDimMin,llDimMax);
	Population    = Problem.Evaluation([xu,xl]);
    Obj           = Population.objs;
    functionValue = Obj(:,1);
end

function functionValue = approximatedFunctionPhi(pop,Problem)
    xu            = pop(:,1:Problem.DU);
    xl            = pop(:,Problem.DU+1:end);
	Population    = Problem.Evaluation([xu,xl]);
    Obj           = Population.objs;
    functionValue = Obj(:,1);
end
        
function approxFunctionValue = approximatedFunction(pop,parameters)
    approxFunctionValue = parameters.constant + pop*parameters.linear + pop*parameters.sqmatrix*pop';
end

function [c,ceq] = approximatedConstraintsPsi(xu,llDimMin,llDimMax,psiFunction,Problem)
    for j = 1 : size(psiFunction,2)
        xl(j) = psiFunction{j}.constant + xu*psiFunction{j}.linear + xu*psiFunction{j}.sqmatrix*xu';
    end
    % Check if predicted lower level optimal solution is outside the bound
    xl  = checkLimits(xl,llDimMin,llDimMax);
    Population = Problem.Evaluation([xu,xl]);
    Con = Population.cons;
    c   = Con(:,1:Problem.C);
    ceq = [];
end
    
function [c,ceq] = approximatedConstraintsPhi(pop,parametersPhiFunction,Problem)
    ulPop = pop(:,1:Problem.DU);
    llPop = pop(:,Problem.DU+1:end);
    Population = Problem.Evaluation(pop);
    c      = Population.cons;
    Obj    = Population.objs;
    c1     = Obj(:,2);
    ceq    = [];  
    n      = length(c);
    c2     = (parametersPhiFunction.constant + ulPop*parametersPhiFunction.linear + ulPop*parametersPhiFunction.sqmatrix*ulPop');
    c(n+1) = c2 - c1;
end

function [c,ceq] = approximatedConstraints(pop,parametersEqualityConstr,parametersInequalityConstr)
    if ~isempty(parametersEqualityConstr)
        for i = 1 : length(parametersEqualityConstr)
            ceq(i) = parametersEqualityConstr{i}.constant + pop*parametersEqualityConstr{i}.linear + pop*parametersEqualityConstr{i}.sqmatrix*pop';
        end
    else
        ceq = [];
    end
    if ~isempty(parametersInequalityConstr)
        for i = 1 : length(parametersInequalityConstr)
            c(i) = parametersInequalityConstr{i}.constant + pop*parametersInequalityConstr{i}.linear + pop*parametersInequalityConstr{i}.sqmatrix*pop';
        end
    else
        c = [];
    end
end

function [c,ceq] = approximatedConstraintsPhi2(pop,parametersEqualityConstr,parametersInequalityConstr,parametersLowerLevelFunction,parametersPhiFunction,parametersLLEqualityConstr,parametersLLInequalityConstr,dimULPop,dimLLPop)
    ulPop = pop(:,1:dimULPop);
    llPop = pop(:,dimULPop+1:dimULPop+dimLLPop);
    ceq   = [];
    c     = [];
    if ~isempty(parametersEqualityConstr)
        for i = 1 : length(parametersEqualityConstr)
            ceq(i) = parametersEqualityConstr{i}.constant + pop*parametersEqualityConstr{i}.linear + pop*parametersEqualityConstr{i}.sqmatrix*pop';
        end
    end
    if ~isempty(parametersLLEqualityConstr)
        for i = 1 : length(parametersLLEqualityConstr)
            ceq(end+1) = parametersLLEqualityConstr{i}.constant + pop*parametersLLEqualityConstr{i}.linear + pop*parametersLLEqualityConstr{i}.sqmatrix*pop';
        end
    end
    if ~isempty(parametersInequalityConstr)
        for i = 1 : length(parametersInequalityConstr)
            c(i) = parametersInequalityConstr{i}.constant + pop*parametersInequalityConstr{i}.linear + pop*parametersInequalityConstr{i}.sqmatrix*pop';
        end
    end
    if ~isempty(parametersLLInequalityConstr)
        for i = 1 : length(parametersLLInequalityConstr)
            c(end+1) = parametersLLInequalityConstr{i}.constant + pop*parametersLLInequalityConstr{i}.linear + pop*parametersLLInequalityConstr{i}.sqmatrix*pop';
        end
    end
    c1 = (parametersLowerLevelFunction.constant + pop*parametersLowerLevelFunction.linear + pop*parametersLowerLevelFunction.sqmatrix*pop');
    c2 = (parametersPhiFunction.constant + ulPop*parametersPhiFunction.linear + ulPop*parametersPhiFunction.sqmatrix*ulPop');
    c(end+1) = c2 - c1;
end
    
function offsprings = checkLimits(offsprings,DimMin,DimMax)
    numOffsprings = size(offsprings,1);
    dimMinMatrix  = DimMin(ones(1,numOffsprings),:);
    offsprings(offsprings<dimMinMatrix) = dimMinMatrix(offsprings<dimMinMatrix);
    dimMaxMatrix  = DimMax(ones(1,numOffsprings),:);
    offsprings(offsprings>dimMaxMatrix) = dimMaxMatrix(offsprings>dimMaxMatrix);
end

function [LB,UB] = createLocalSearchBound(Indv,DimMin,DimMax,range)
	diff = range.*(DimMax - DimMin);
	LB   = Indv - diff;
	LB   = checkLimits(LB,DimMin,DimMax);
	UB   = Indv + diff;
	UB   = checkLimits(UB,DimMin,DimMax);
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Problem,Population,Offspring)
% The environmental selection of NBLEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    selected = randperm(length(Population),2);
    Pool     = [Population(selected),Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool));
    Population(selected) = Pool(rank(1:length(selected)));
end
```

### `OperatorPCX.m`
```matlab
function Offspring = OperatorPCX(Parent,Lower,Upper)
% Offspring generation based on PCX and polynomial mutation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [proC,proM,disM] = deal(0.9,1,20);
    [N,D] = size(Parent);

    %% PCX
    W         = mean(Parent,1);
    p1        = [2:N,1];
    p2        = [3:N,1,2];
    Offspring = Parent + 0.1*(Parent-W) + mean(abs(Parent-W),2).*(Parent(p2,:)-Parent(p1,:))/2;
    Site      = repmat(rand(N,1)>proC,1,D);
    Offspring(Site) = Parent(Site);
    
    %% Polynomial mutation
    Lower = repmat(Lower,N,1);
    Upper = repmat(Upper,N,1);
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

### `QuadApprox.m`
```matlab
function approxmodel = QuadApprox(y,X)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    warning off all;
    dimensions  = size(X,2);
    datasetSize = size(X,1);
    if (datasetSize<dimensions)
        error('Cannot compute model as the datasetSize is smaller than dimensions.');
    end

    modelConsidered = {'linear','purequadratic','quadratic'};
    numModel = length(modelConsidered);
    XX       = cell(1,numModel);
    output   = cell(1,numModel);
    mse      = zeros(1,numModel);
    bic      = zeros(1,numModel);

    for i = 1 : numModel
        model = modelConsidered{i};
        XX{i} = x2fx(X, model);
        output{i}.beta = XX{i}\y; 
        mse(i) = real(sum((y-XX{i}*output{i}.beta).^2)/length(y));
        bic(i) = size(XX{i},2)*log(size(XX{i},1))/size(XX{i},1) + 2*log(mse(i)); 
    end

    [~,index] = min(bic);

    modelSelected = modelConsidered(index);
    constant      = output{index}.beta(1);
    linear        = output{index}.beta(2:2+dimensions-1);
    sqmatrix      = zeros(dimensions,dimensions);

    if strcmp(modelSelected,'purequadratic')
        diagonal = output{i}.beta(end-dimensions+1:end);
        for i = 1 : dimensions
            sqmatrix(i,i) = diagonal(i);
        end
    end

    if strcmp(modelSelected,'quadratic')
        cross    = output{i}.beta(2+dimensions:end-dimensions);
        diagonal = output{i}.beta(end-dimensions+1:end);
        k        = 0;
        for i = 1 : dimensions
            sqmatrix(i,i) = diagonal(i);
            for j = i+1 : dimensions
                k = k + 1;
                sqmatrix(i,j) = cross(k)/2;
                sqmatrix(j,i) = cross(k)/2;
            end
        end
    end

    stdXX           = std(XX{index});
    stdy            = std(y);
    stdXX(stdXX==0) = -realmin;
    stdy(stdy==0)   = -realmin;
    XXNorm          = (XX{index}-ones(size(XX{index},1),1)*mean(XX{index}))./(ones(size(XX{index},1),1)*stdXX);
    yNorm           = (y-mean(y))./stdy;
    betaNorm        = XXNorm\yNorm;
    mseNorm         = sum((yNorm-XXNorm*betaNorm).^2)/length(yNorm);

    approxmodel = struct('model',modelSelected,'constant',real(constant),...
                    'linear',real(linear),'sqmatrix',real(sqmatrix),...
                    'mse',mse(index),'mseNorm',mseNorm,'bic',bic(index));

    if isnan(approxmodel.mse)
        error('The code has ended up with NAN values. One of the reasons could be duplicate rows in the input matrix that makes the system under-defined and a solution cannot be found using least-squares.')
    end
end
```

### `getLowerLevelVariableFromMapping.m`
```matlab
function [offspringsLowerLevelVariables,sumMSE,validMSE] = getLowerLevelVariableFromMapping(offsprings,psiMapping,phiMapping,Problem,archive)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % minimum size required for phi-mapping based lower level optimal variable retrieval
    minPhiSize = (Problem.DU+Problem.DL+1)*(Problem.DU+Problem.DL+2)/2 + 2*(Problem.DU) + Problem.DU;
    
    %% Determine which mapping to select in the descendant update 
    runPhi = 0;
    if (length(archive.tag1)+length(archive.tag0)) >= minPhiSize
        runPhi = ((psiMapping.sumMSE>=phiMapping.sumMSE) && (psiMapping.validMSE>=phiMapping.validMSE)) || (rand<0.25);
    end
    
	offspringsLowerLevelVariables = getLowerLevelVariableFromPsi(offsprings,psiMapping.function);
    sumMSE   = psiMapping.sumMSE;
    validMSE = psiMapping.validMSE;
        
	if runPhi == 1
		offspringsLowerLevelVariables = getLowerLevelVariableFromPhi(Problem,offsprings,offspringsLowerLevelVariables,phiMapping.function{:},[archive.tag1,archive.tag0],minPhiSize);
        sumMSE   = phiMapping.sumMSE;
        validMSE = phiMapping.validMSE;
    end
end	

function indvLowerLevelVariables = getLowerLevelVariableFromPsi(indv,psiFunction)
	DL = length(psiFunction);
	indvLowerLevelVariables = zeros(1,DL);
	for j = 1 : DL
		indvLowerLevelVariables(j) = psiFunction{j}.constant + indv*psiFunction{j}.linear + indv*psiFunction{j}.sqmatrix*indv';
	end	
end

function indvLowerLevelVariables = getLowerLevelVariableFromPhi(Problem,indv,indvLowerLevelVariables,phiFunction,archive,archiveSize)
    archiveDec       = archive.decs;
	archivePhi.upper = archiveDec(:,1:Problem.DU);
    for j = 1 : size(archivePhi.upper,1)
        distances(j) = sum((indv - archivePhi.upper(j,:)).^2);
    end
    [~,I]      = sort(distances);
    I          = I(1:archiveSize);
    archive    = archive(I);
    archiveDec = archive.decs;
    archiveObj = archive.objs;
    archiveCon = archive.cons;
    archivePhi.upper = archivePhi.upper(I,:);
    archivePhi.lower = archiveDec(:,Problem.DU+1:end);
	archivePhi.functionValue          = archiveObj(:,1);
	archivePhi.equalityConstrVals     = [];
	archivePhi.inequalityConstrVals   = archiveCon(:,1:Problem.C);	
	archivePhi.llFunctionValue        = archiveObj(:,2);
    archivePhi.llEqualityConstrVals   = [];
    archivePhi.llInequalityConstrVals = archiveCon(:,Problem.C+1:end);

    approxPhi.function = QuadApprox(archivePhi.functionValue, [archivePhi.upper archivePhi.lower]);
    if ~isempty(archivePhi.equalityConstrVals)
        for i = 1 : size(archivePhi.equalityConstrVals,2)		            
            approxPhi.equalityConstr{i} = QuadApprox(archivePhi.equalityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);
        end
    else
        approxPhi.equalityConstr = [];
    end
    if ~isempty(archivePhi.inequalityConstrVals)
        for i = 1 : size(archivePhi.inequalityConstrVals,2)
           approxPhi.inequalityConstr{i} = QuadApprox(archivePhi.inequalityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);            
        end
    else
        approxPhi.inequalityConstr = [];
    end

    approxPhi.llFunction = QuadApprox(archivePhi.llFunctionValue, [archivePhi.upper archivePhi.lower]);

    if ~isempty(archivePhi.llEqualityConstrVals)
        for i = 1 : size(archivePhi.llEqualityConstrVals,2)
            approxPhi.llEqualityConstr{i} = QuadApprox(archivePhi.llEqualityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);
        end
    else
        approxPhi.llEqualityConstr = [];
    end

    if ~isempty(archivePhi.llInequalityConstrVals)
        for i = 1 : size(archivePhi.llInequalityConstrVals,2)
            approxPhi.llInequalityConstr{i} = QuadApprox(archivePhi.llInequalityConstrVals(:,i), [archivePhi.upper archivePhi.lower]);
        end
    else
        approxPhi.llInequalityConstr = [];
    end
    lb      = min(archivePhi.lower);
    ub      = max(archivePhi.lower);
    options = optimset('Algorithm','sqp','Display','off');  
    [indvLowerLevelVariables,FVAL,EXITFLAG,OUTPUT] = fmincon(@(xl) -approximatedFunction(xl,indv,approxPhi.function),...
                    indvLowerLevelVariables,[],[],[],[],lb,ub,@(xl) approximatedConstraints(xl,indv,...
                    approxPhi.equalityConstr,approxPhi.inequalityConstr, approxPhi.llFunction,...
                    phiFunction, approxPhi.llEqualityConstr, approxPhi.llInequalityConstr),options);  
end

function approxFunctionValue = approximatedFunction(xl,xu,parameters)
    pop = [xu xl];
    approxFunctionValue = parameters.constant + pop*parameters.linear + pop*parameters.sqmatrix*pop'; 
end

function [c,ceq] = approximatedConstraints(llPop,ulPop,parametersEqualityConstr,parametersInequalityConstr,parametersLowerLevelFunction,parametersPhiFunction,parametersLLEqualityConstr,parametersLLInequalityConstr)
    pop = [ulPop,llPop];
    ceq = [];
    c   = [];
    if ~isempty(parametersEqualityConstr)
        for i = 1 : length(parametersEqualityConstr)
            ceq(i) = parametersEqualityConstr{i}.constant + pop*parametersEqualityConstr{i}.linear + pop*parametersEqualityConstr{i}.sqmatrix*pop';
        end
    end
    if ~isempty(parametersLLEqualityConstr)
        for i = 1 : length(parametersLLEqualityConstr)
            ceq(end+1) = parametersLLEqualityConstr{i}.constant + pop*parametersLLEqualityConstr{i}.linear + pop*parametersLLEqualityConstr{i}.sqmatrix*pop';
        end
    end
    if ~isempty(parametersInequalityConstr)
        for i = 1 : length(parametersInequalityConstr)
            c(i) = parametersInequalityConstr{i}.constant + pop*parametersInequalityConstr{i}.linear + pop*parametersInequalityConstr{i}.sqmatrix*pop';
        end
    end
    if ~isempty(parametersLLInequalityConstr)
        for i = 1 : length(parametersLLInequalityConstr)
            c(end+1) = parametersLLInequalityConstr{i}.constant + pop*parametersLLInequalityConstr{i}.linear + pop*parametersLLInequalityConstr{i}.sqmatrix*pop';
        end
    end
    c1       = (parametersLowerLevelFunction.constant + pop*parametersLowerLevelFunction.linear + pop*parametersLowerLevelFunction.sqmatrix*pop');
    c2       = (parametersPhiFunction.constant + ulPop*parametersPhiFunction.linear + ulPop*parametersPhiFunction.sqmatrix*ulPop');
    c(end+1) = c2 - c1;
end
```

### `getMappings.m`
```matlab
function [psiMapping,phiMapping,lies] = getMappings(Problem,indv,archive,option)
% Construct Psi-mapping or Phi-mapping or both

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    psiApprox    = []; 
    phiApprox    = [];
    constructPsi = 0; 
    constructPhi = 0;
    
    if nargin < 4
        option = 3;
    end
    if option == 1
        constructPsi = 1;
    elseif option == 2
        constructPhi = 1;
    elseif option == 3
        constructPsi = 1;
        constructPhi = 1;
    end
    
    % Minimum number of Lower level points required for quadratic approximation of both mappings
    numberLowerLevelPointsTrain = (Problem.DU+1)*(Problem.DU+2)/2+2*(Problem.DU);
    numberLowerLevelPointsEval  = Problem.DU;
    minNumberLowerLevelPoints   = numberLowerLevelPointsTrain + numberLowerLevelPointsEval;

    PopDec = archive.decs;
    quadArchive.upper = PopDec(:,1:Problem.DU);
    quadArchive.lower = PopDec(:,Problem.DU+1:end);
    
    % Only the members close to the offspring are used for quadratic approximation
    distances = zeros(size(quadArchive.upper,1));
    for k = 1 : length(quadArchive)
        distances(k) = sum((indv - quadArchive.upper(k,:)).^2);
    end
    [~,I]  = sort(distances);
    I      = I(1:minNumberLowerLevelPoints);           
    sizeI  = length(I);
    permut = randperm(sizeI);
    J      = permut(1:sizeI-numberLowerLevelPointsEval);
    quadApproxMembers = I(J);
    % I stores the members being considered from the archive
    % quadApproxMembers stores the members from I used for a quadratic
    setDiff = setdiff(I,quadApproxMembers);
    if constructPsi == 1
        psiApprox = cell(1,Problem.DL); sumMSE = 0;
        for j = 1 : Problem.DL 
            psiApprox{j} = QuadApprox(quadArchive.lower(quadApproxMembers,j),quadArchive.upper(quadApproxMembers,:));
            %Calculating sum of MSE for all lower level variable approximations
            sumMSE = sumMSE + psiApprox{j}.mseNorm;
        end
        predictedLowerLevelVariables = zeros(length(setDiff),Problem.DL);
        for k = 1 : length(setDiff)
            for j = 1 : Problem.DL
                predictedLowerLevelVariables(k,j) = psiApprox{j}.constant + quadArchive.upper(setDiff(k),:)*psiApprox{j}.linear + quadArchive.upper(setDiff(k),:)*psiApprox{j}.sqmatrix*quadArchive.upper(setDiff(k),:)';
            end
        end
        psiMapping.function = psiApprox;
        psiMapping.sumMSE   = sumMSE;
        psiMapping.validMSE = mean(mean((predictedLowerLevelVariables - quadArchive.lower(setDiff,:)).^2));
    end
    
    if constructPhi == 1 
        PopObj        = archive.objs;
        quadArchive.llFunctionValue = PopObj(:,2);
        llFunctionDim = size(quadArchive.llFunctionValue,2);
        sumMSE        = 0;
        phiApprox     = cell(1,llFunctionDim);
        for j = 1 : llFunctionDim
            phiApprox{j} = QuadApprox(quadArchive.llFunctionValue(quadApproxMembers,j), quadArchive.upper(quadApproxMembers,:));
            sumMSE       = sumMSE+phiApprox{j}.mseNorm;
        end
        predictedLowerLevelObjective = zeros(length(setDiff),llFunctionDim);  
        for k = 1 : length(setDiff)
            for j = 1 : llFunctionDim
                predictedLowerLevelObjective(k,j) = phiApprox{j}.constant + quadArchive.upper(setDiff(k),:)*phiApprox{j}.linear + quadArchive.upper(setDiff(k),:)*phiApprox{j}.sqmatrix*quadArchive.upper(setDiff(k),:)';
            end
        end
        phiMapping.function = phiApprox;
        phiMapping.sumMSE   = sumMSE;
        phiMapping.validMSE = mean(mean((predictedLowerLevelObjective - quadArchive.llFunctionValue(setDiff,:)).^2));
    end
    
    %Checks if the offspring lies in between the training points and not outside
    lies = 1;
    for j = 1 : size(quadArchive.upper,2)
        if indv(j)>=max(quadArchive.upper(quadApproxMembers,j)) || indv(j)<=min(quadArchive.upper(quadApproxMembers,j))
            lies = 0;
        end
    end
end
```

### `llSearch.m`
```matlab
function [eliteIndiv,tag] = llSearch(Problem,ulPopDec,llPopmember,llmemberVariance, llPop)
% Obtain the upper member corresponds to the best lower member

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% SQP
    epsilonZero = 1e-6;
    tag         = 0;
    % Initialize the lower population
    llPopSize = (Problem.DL+1)*(Problem.DL+2)/2+Problem.DL;
    if ~isempty(llPopmember)
        for i = 1 : llPopSize - 1
            llPopDec(i,:) = Problem.lower(Problem.DU+1:end) + rand(1, Problem.DL).*(Problem.upper(Problem.DU+1:end)-Problem.lower(Problem.DU+1:end));
        end
        llPopDec(llPopSize,:) = llPopmember;
    else
        for i = 1 : llPopSize
            llPopDec(i,:) = Problem.lower(Problem.DU+1:end) + rand(1, Problem.DL).*(Problem.upper(Problem.DU+1:end)-Problem.lower(Problem.DU+1:end));
        end
        llPopmember = llPopDec(llPopSize,:);
    end
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec,llPopSize,1),llPopDec]);
    llPopCon     = llPopulation.cons;
    llPopObj     = llPopulation.objs;
    if sum(sum(isnan(llPopCon)))>0 || sum(sum(isnan(llPopObj)))>0
        llPopObj = llPopObj(~isnan(llPopObj));
        llPopCon = llPopCon(~isnan(llPopCon));
        llPopCon = reshape(llPopCon,[llPopSize,length(llPopCon)/llPopSize]);
    end
    % Construct the Lower quadratic approximations of objective function and linear approximations of constraints
    approx.function       = QuadApprox(llPopObj, llPopDec);
    approx.equalityConstr = [];
    if size(llPopCon,2) ~= 0
        for i = 1 : size(llPopCon,2)
            approx.inequalityConstr{i} = QuadApprox(llPopCon(:,i), llPopDec);
        end
    else
        approx.inequalityConstr = [];
    end
    
    % Quadratic functions of linear constraints are optimized using sequence quadratic programming
    options = optimset('Display','off','TolX',1e-10,'TolFun',1e-10);
    llUpper = Problem.upper(Problem.DU+1:end);
    llLower = Problem.lower(Problem.DU+1:end);
    [eliteIndivllDec,~,~,~,~] = fmincon(@(x) approximatedFunction(x,approx.function),llPopmember,[],[],[],[],llLower,llUpper,@(x) approximatedConstraints(x,approx.equalityConstr,approx.inequalityConstr),options);
    eliteIndiv   = Problem.EvaluationLower([ulPopDec, eliteIndivllDec]);
    llPopulation = [llPopulation, eliteIndiv];
    
    eliteIndivDec               = eliteIndiv.decs;
    eliteFunctionValue          = eliteIndiv.objs;
    elitellFunctionValue        = eliteFunctionValue(~isnan(eliteFunctionValue));
    eliteInequalityConstrVals   = eliteIndiv.cons;
    elitellInequalityConstrVals = eliteInequalityConstrVals(~isnan(eliteInequalityConstrVals));
    elitellEqualityConstrVals   = [];
    
    f       = approximatedFunction(eliteIndivDec(:, Problem.DU+1:end),approx.function);
    [c,ceq] = approximatedConstraints(eliteIndivllDec,approx.equalityConstr,approx.inequalityConstr);
    d       = sqrt((f-elitellFunctionValue)^2+sum((c-elitellInequalityConstrVals).^2)+sum((ceq-elitellEqualityConstrVals).^2));
    if d < epsilonZero
        tag = 1;
        eliteIndiv = eliteIndiv.dec(Problem.DU+1:end);
        return;
    end
    if tag == 0
        llPopDec = zeros(Problem.N, Problem.DL);
        if ~isempty(llPop)
            if size(llPop,1) > Problem.N
                r = randperm(size(llPop,1));
                r = r(1:Problem.N);
                llPopDec = llPop(r,:);
            else
                for i = 1 : Problem.N
                    llPopDec(i,:) = Problem.lower(Problem.DU+1:end) + rand(1, Problem.DL).*(Problem.upper(Problem.DU+1:end)-Problem.lower(Problem.DU+1:end));
                end
                llPopDec(1:size(llPop,1),:) = llPop;
            end
        else
            for i = 1 : Problem.N
                llPopDec(i,:) = Problem.lower(Problem.DU+1:end) + rand(1, Problem.DL).*(Problem.upper(Problem.DU+1:end)-Problem.lower(Problem.DU+1:end));
            end
            llPopDec(Problem.N+1,:) = eliteIndivllDec;
        end
    end
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec, size(llPopDec,1),1), llPopDec]);
    FElower      = 0;
    alpha_init   = sum(var(llPopDec));
    
    %% Optimization
    while FElower < Problem.maxFElower
        % Select parents and generate offspring
        MatingPool  = TournamentSelection(2,3,CalFitness(Problem.C,llPopulation));
        ParentDec   = llPopulation(MatingPool).decs;
        llOffDec    = OperatorPCX(ParentDec(:,Problem.DU+1:end),Problem.lower(Problem.DU+1:end),Problem.upper(Problem.DU+1:end));
        llOffspring = Problem.EvaluationLower([repmat(ulPopDec,size(llOffDec,1),1),llOffDec]);
        FElower     = FElower + length(llOffspring);
        % Select r members with better adaptability
        llPopulation = EnvironmentalSelection(Problem,llPopulation,llOffspring);
        llPopDec = llPopulation.decs;
        alpha = sum(var(llPopDec(:,Problem.DU+1:end)))/alpha_init;
        if alpha < 1e-4
            tag = 1;
            [~,best]   = min(CalFitness(Problem.C,llPopulation));
            eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
            return;
        end
    end
    [~,best]   = min(CalFitness(Problem.C,llPopulation));
    eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
    
end

function approxFunctionValue = approximatedFunction(pop, parameters)
    approxFunctionValue = parameters.constant + pop*parameters.linear + pop*parameters.sqmatrix*pop';
end

function [c, ceq] = approximatedConstraints(pop, parametersEqualityConstr, parametersInequalityConstr)
    if ~isempty(parametersEqualityConstr)
        for i = 1 : length(parametersEqualityConstr)
            ceq(i) = parametersEqualityConstr{i}.constant + pop*parametersEqualityConstr{i}.linear + pop*parametersEqualityConstr{i}.sqmatrix*pop';
        end
    else
        ceq = [];
    end

    if ~isempty(parametersInequalityConstr)
        for i = 1 : length(parametersInequalityConstr)
            c(i) = parametersInequalityConstr{i}.constant + pop*parametersInequalityConstr{i}.linear + pop*parametersInequalityConstr{i}.sqmatrix*pop';
        end
    else
        c = [];
    end
end
```
