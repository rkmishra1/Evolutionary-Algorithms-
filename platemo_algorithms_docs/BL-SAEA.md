# BL-SAEA

**Tags**: <2024> <multi> <real> <constrained/none> <bilevel>

## Description
Bi-level surrogate modelling based evolutionary algorithm

## Reference
H. Jiang, K. Qiu, Y. Tian, X. Zhang, and Y. Jin. Efficient surrogate modeling method for evolutionary algorithm to solve bilevel optimization problems. IEEE Transactions on Cybernetics, 2024, 54(7): 4335-4347

## Source Code

### `BLSAEA.m`
```matlab
classdef BLSAEA < ALGORITHM
% <2024> <multi> <real> <constrained/none> <bilevel>
% Bi-level surrogate modelling based evolutionary algorithm

%------------------------------- Reference --------------------------------
% H. Jiang, K. Qiu, Y. Tian, X. Zhang, and Y. Jin. Efficient surrogate
% modeling method for evolutionary algorithm to solve bilevel optimization
% problems. IEEE Transactions on Cybernetics, 2024, 54(7): 4335-4347
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

            ulPopSize=floor(Problem.N/2);                   % Size of UL population
            ulDim=Problem.DU;                               % Number of UL dimensions
            llPopSize=Problem.N-ulPopSize;                  % Size of LL population
            llMaxGens=2000;                                 % Maximum number of generations allowed at LL
            llDim=Problem.DL;                               % Number of LL dimensions

            ulDimMin=Problem.lower(1:Problem.DU);           % Minimum bound accross UL dimensions
            ulDimMax=Problem.upper(1:Problem.DU);           % Maximum bound accross UL dimensions

            llDimMin=Problem.lower(Problem.DU+1:end);    	% Minimum bound accross LL dimensions
            llDimMax=Problem.upper(Problem.DU+1:end);    	% Maximum bound accross LL dimensions

            llStoppingCriteria = 1e-5;

            %% Ulsearch
            proC=1;disC=20;proM=1;disM=20;

            ulFunctionEvaluation = 0;
            llFunctionEvaluation = 0;

            %Upper population initialization (LHS sampling)
            Pu = UniformPoint(ulPopSize,ulDim,'Latin');
            ulPop = repmat(ulDimMax-ulDimMin,ulPopSize,1).*Pu+repmat(ulDimMin,ulPopSize,1);

            %For each xu, find the optimal lower-level solution
            for i=1:ulPopSize
                [llPop(i,:), llFunctionValue(i,:),l_con(i,:),llFunctionEvaluation,ulFunctionEvaluation]=llSearch([],0,[],[],Problem,llDim, llDimMin, llDimMax, ulPop(i,:),llPopSize,llStoppingCriteria,proC,disC,proM,disM,llFunctionEvaluation,llMaxGens,ulFunctionEvaluation);
            end

            %ulevaluate
            [ul_obj, ~, ul_con]=ulevaluate(ulPop, llPop, Problem);
            ulFunctionEvaluation = ulFunctionEvaluation+ulPopSize;
            if isempty(ul_con)
                [n,~]=size(ul_obj);
                ul_con=zeros(n,1);
            end

            %Fitness
            if isempty(ul_con) && all(l_con==0)
                ul_fitness=ul_obj;
            else
                PopCon   = sum(max(0,ul_con),2)+sum(max(0,l_con),2);
                Feasible = PopCon <= 0;
                ul_fitness  = Feasible.*ul_obj + ~Feasible.*(PopCon+1e10);
            end

            %Archive
            Archive_xu=ulPop; Archive_xl=llPop; Archive_F=ul_obj; Archive_f=llFunctionValue;

            GEN=0;

            Population  = Problem.Evaluation([ulPop,llPop]);
            %Iterate
            while Algorithm.NotTerminated(Population)
                %Selection
                MatingPool1 = TournamentSelection(2,ulPopSize,ul_fitness);
                MatingPool2 = TournamentSelection(2,ulPopSize,ul_fitness);

                %Produce offspring
                Offspring1  = BLSAEAGA(ulPop(MatingPool1',:),{proC,disC,proM,disM},ulDimMin, ulDimMax);
                Offspring2  = BLSAEAGA(ulPop(MatingPool2',:),{proC,disC,proM,disM},ulDimMin, ulDimMax);

                Offspring  = [Offspring1 ; Offspring2];

                X=Archive_xu;
                Y=Archive_F;

                % srgtsKRGSetOptions
                options = srgtsKRGSetOptions(X,Y);

                % srgtsKRGFit
                switch func2str(options.FIT_Fn)
                    case 'dace_fit'
                        state.FIT_Fn    = options.FIT_Fn;
                        state.FIT_FnVal = NaN;
                        if isempty(options.KRG_LowerBound) % no optimization for theta
                            [surrogate.KRG_DACEModel, state.KRG_DACEPerf, state.FIT_FnVal] = dace_fit(...
                                options.P, ...
                                options.T, ...
                                options.KRG_RegressionModel, ...
                                options.KRG_CorrelationModel, ...
                                options.KRG_Theta0);
                        else
                            [surrogate.KRG_DACEModel, state.KRG_DACEPerf, state.FIT_FnVal] = dace_fit(...
                                options.P, ...
                                options.T, ...
                                options.KRG_RegressionModel, ...
                                options.KRG_CorrelationModel, ...
                                options.KRG_Theta0, ...
                                options.KRG_LowerBound, ...
                                options.KRG_UpperBound);
                        end

                    case 'srgtsXVFit'
                        [surrogate state] = srgtsXVFit(options);
                end

                [pre_off_ul_obj,PredVar] = dace_predictor(Offspring, surrogate.KRG_DACEModel);

                objs=[pre_off_ul_obj PredVar];
                [FrontNo,~] = NDSort(objs,ulPopSize*2);

                %Select according to non-dominated level, first frontier
                Next = FrontNo ==1;
                Num = find(Next==1);
                non_n=length(Num);

                %For non-dominated descendants, find the optimal lower-level solution
                [Archive_size,~]=size(Archive_F);

                for j=1:non_n

                    %Calculate the cosine of the included angle and find the closest point K
                    for e=1:Archive_size
                        c_o_s(e,1)=1-pdist([Offspring(Num(j),:);Archive_xu(e,:)],'cosine');
                    end
                    c_o_s=c_o_s(1:Archive_size);

                    %Establish the mapping relationship of xu-xl(i), local model
                    models = struct('surrogate', {}, 'state', {});
                    xl0=[];
                    for i = 1:llDim
                        % srgtsKRGSetOptions
                        options = srgtsKRGSetOptions(X,Y);

                        % srgtsKRGFit
                        switch func2str(options.FIT_Fn)
                            case 'dace_fit'
                                models(i).state.FIT_Fn    = options.FIT_Fn;
                                models(i).state.FIT_FnVal = NaN;
                                if isempty(options.KRG_LowerBound) % no optimization for theta
                                    [models(i).surrogate.KRG_DACEModel, models(i).state.KRG_DACEPerf, models(i).state.FIT_FnVal] = dace_fit(...
                                        options.P, ...
                                        options.T, ...
                                        options.KRG_RegressionModel, ...
                                        options.KRG_CorrelationModel, ...
                                        options.KRG_Theta0);
                                else
                                    [models(i).surrogate.KRG_DACEModel, models(i).state.KRG_DACEPerf, models(i).state.FIT_FnVal] = dace_fit(...
                                        options.P, ...
                                        options.T, ...
                                        options.KRG_RegressionModel, ...
                                        options.KRG_CorrelationModel, ...
                                        options.KRG_Theta0, ...
                                        options.KRG_LowerBound, ...
                                        options.KRG_UpperBound);
                                end

                            case 'srgtsXVFit'
                                [models(i).surrogate models(i).state] = srgtsXVFit(options);
                        end
                        % srgtsKRGPredictor
                        [xl(i),~] = dace_predictor(Offspring(Num(j),:), models(i).surrogate.KRG_DACEModel);
                        xl0=[xl0 xl(i)];

                    end
                    %As the initial solution for lower-level optimization
                    [ND_off_llPop(j,:), ND_off_llFunctionValue(j,:),ND_l_con(j,:),llFunctionEvaluation,ulFunctionEvaluation]=llSearch(xl0,1,ulPop,llPop,Problem,llDim, llDimMin, llDimMax, Offspring(Num(j),:),llPopSize,llStoppingCriteria,proC,disC,proM,disM,llFunctionEvaluation,llMaxGens,ulFunctionEvaluation);

                end

                %ulevaluate
                [ND_off_ul_obj, ~, ND_off_ul_con]=ulevaluate(Offspring(Num',:), ND_off_llPop, Problem);
                ulFunctionEvaluation = ulFunctionEvaluation+non_n;
                if isempty(ND_off_ul_con)
                    [n,~]=size(ND_off_ul_obj);
                    ND_off_ul_con=zeros(n,1);
                end

                %Fitness
                if isempty(ND_off_ul_con) && all(ND_l_con(1:non_n,:)==0)
                    ND_off_ul_fitness=ND_off_ul_obj;
                else
                    PopCon   = sum(max(0,ND_off_ul_con),2)+sum(max(0,ND_l_con(1:non_n,:)),2);
                    Feasible = PopCon <= 0;
                    ND_off_ul_fitness  = Feasible.*ND_off_ul_obj + ~Feasible.*(PopCon+1e10);
                end

                %Newly generated descendant archive
                Archive_xu=[Archive_xu;Offspring(Num',:)]; Archive_xl=[Archive_xl;ND_off_llPop(1:non_n,:)]; Archive_F=[Archive_F;ND_off_ul_obj(1:non_n,:)]; Archive_f=[Archive_f;ND_off_llFunctionValue(1:non_n,:)];

                %Update archive
                A=round(Archive_xu, -9);
                [~,ia,~] = unique(A,'rows');

                Archive_xu = Archive_xu(ia,:);
                Archive_xl = Archive_xl(ia,:);
                Archive_F = Archive_F(ia,:);
                Archive_f = Archive_f(ia,:);
                [A_size,~]= size(Archive_F);
                if A_size>900
                    [~,F_rank]=sort(Archive_F);
                    Archive_xu=Archive_xu(F_rank(1:900),:);
                    Archive_xl=Archive_xl(F_rank(1:900),:);
                    Archive_F=Archive_F(F_rank(1:900),:);
                    Archive_f=Archive_f(F_rank(1:900),:);
                end

                %Merge parent and spring populations
                ulPop = [ulPop;Offspring(Num',:)];
                llPop = [llPop;ND_off_llPop(1:non_n,:)];
                ul_obj=[ul_obj;ND_off_ul_obj(1:non_n,:)];
                ul_fitness=[ul_fitness;ND_off_ul_fitness(1:non_n,:)];
                llFunctionValue=[llFunctionValue;
                    ND_off_llFunctionValue(1:non_n,:)];
                [~,rank]   = sort(ul_fitness);
                ulPop = ulPop(rank(1:ulPopSize),:);
                llPop = llPop(rank(1:ulPopSize),:);
                ul_obj = ul_obj(rank(1:ulPopSize),:);
                ul_fitness = ul_fitness(rank(1:ulPopSize),:);
                llFunctionValue = llFunctionValue(rank(1:ulPopSize),:);


                %Current optimal solution gradient
                current_best_xu=ulPop(1,:);

                [Archive_size,~]=size(Archive_F);

                %Calculate the cosine of the included angle and find the closest point K
                for e=1:Archive_size
                    c_o_s(e,1)=1-pdist([current_best_xu;Archive_xu(e,:)],'cosine');
                end
                c_o_s=c_o_s(1:Archive_size);

                %Establish the mapping relationship of xu-xl(i), local model
                models = struct('surrogate', {}, 'state', {});
                n_xl0=[];
                for i = 1:llDim
                    % srgtsKRGSetOptions
                    options = srgtsKRGSetOptions(X,Y);

                    % srgtsKRGFit
                    switch func2str(options.FIT_Fn)
                        case 'dace_fit'
                            models(i).state.FIT_Fn    = options.FIT_Fn;
                            models(i).state.FIT_FnVal = NaN;
                            if isempty(options.KRG_LowerBound) % no optimization for theta
                                [models(i).surrogate.KRG_DACEModel, models(i).state.KRG_DACEPerf, models(i).state.FIT_FnVal] = dace_fit(...
                                    options.P, ...
                                    options.T, ...
                                    options.KRG_RegressionModel, ...
                                    options.KRG_CorrelationModel, ...
                                    options.KRG_Theta0);
                            else
                                [models(i).surrogate.KRG_DACEModel, models(i).state.KRG_DACEPerf, models(i).state.FIT_FnVal] = dace_fit(...
                                    options.P, ...
                                    options.T, ...
                                    options.KRG_RegressionModel, ...
                                    options.KRG_CorrelationModel, ...
                                    options.KRG_Theta0, ...
                                    options.KRG_LowerBound, ...
                                    options.KRG_UpperBound);
                            end

                        case 'srgtsXVFit'
                            [models(i).surrogate models(i).state] = srgtsXVFit(options);
                    end
                    % srgtsKRGPredictor
                    [xl(i),~] = dace_predictor(Offspring(Num(j),:), models(i).surrogate.KRG_DACEModel);

                    n_xl0=[n_xl0 xl(i)];

                end

                %nested_LocalSearch
                lb = Problem.lower(1:Problem.DU);
                ub = Problem.upper(1:Problem.DU);

                options = optimoptions('fmincon','Algorithm','sqp','Display','none');
                Population = Problem.Evaluation([ulPop,llPop]);
                % Optimize the up-level objective using
                llPopulation = Problem.EvaluationLower([ulPop,llPop]);
                llPopCon = llPopulation.cons;
                llPopObj = llPopulation.objs;

                if sum(sum(isnan(llPopCon)))>0 || sum(sum(isnan(llPopObj)))>0
                    % llPopCon
                    nanColumnsCon = any(isnan(llPopCon), 1);
                    llPopCon(:, nanColumnsCon) = [];
                end
                l=size(llPopCon, 2);

                PopCon = Population.cons;
                PopObj = Population.objs;
                ulPopObj=PopObj(:,1);
                ulPopCon = PopCon(:,1:l);

                % Construct the Lower quadratic approximations of objective function and linear approximations of constraints
                approx.function       = QuadApprox(ulPopObj, ulPop);
                approx.equalityConstr = [];
                if size(ulPopCon,2) ~= 0
                    for i = 1 : size(ulPopCon,2)
                        approx.inequalityConstr{i} = QuadApprox(ulPopCon(:,i), ulPop);
                    end
                else
                    approx.inequalityConstr = [];
                end
                [x,~,~,output] = fmincon(@(x)nestedproblem(x, Problem,n_xl0,ulPop,llPop),current_best_xu,[],[],[],[],lb,ub,@(x) approximatedConstraints(x,approx.equalityConstr,approx.inequalityConstr),options);

                L_current_best_xu=x;
                ulFunctionEvaluation=ulFunctionEvaluation+output.funcCount;
                [L_xl, L_f,L_l_con,llFunctionEvaluation,ulFunctionEvaluation]=llSearch([],0,[],[],Problem,llDim, llDimMin, llDimMax, L_current_best_xu,llPopSize,llStoppingCriteria,proC,disC,proM,disM,llFunctionEvaluation,llMaxGens,ulFunctionEvaluation);
                [L_ul_obj, ~, L_ul_con,]=ulevaluate(L_current_best_xu, L_xl, Problem);
                % Fitness
                if isempty(L_ul_con) && all(L_l_con==0)
                    L_ul_fitness=L_ul_obj;
                else
                    PopCon   = sum(max(0,L_ul_con),2)+sum(max(0,L_l_con),2);
                    Feasible = PopCon <= 0;
                    L_ul_fitness  = Feasible.*L_ul_obj + ~Feasible.*(PopCon+1e10);
                end

                %Replacement population worst solution
                ulPop(ulPopSize,:)=L_current_best_xu;
                ul_obj(ulPopSize,:)=L_ul_obj;
                llPop(ulPopSize,:)=L_xl;
                llFunctionValue(ulPopSize,:)=L_f;
                ul_fitness(ulPopSize,:)=L_ul_fitness;

                GEN=GEN+1;

            end
        end
    end
end

function [ f,llFunctionEvaluations,xl] = nestedproblem(xa,Problem,n_xl0,ulPop,llPop)
    xu1 = xa(1:Problem.p);
    xu2 = xa(Problem.p+1:Problem.p+Problem.r);
    %SQP
    ub = Problem.upper(Problem.DU+1:end);
    lb = Problem.lower(Problem.DU+1:end); 

    options = optimoptions('fmincon','Algorithm','sqp','Display','none');

    llPopulation = Problem.EvaluationLower([ulPop,llPop]);
    llPopCon = llPopulation.cons;
    llPopObj = llPopulation.objs;
    if sum(sum(isnan(llPopCon)))>0 || sum(sum(isnan(llPopObj)))>0
        % Deal with llPopObj
        nanColumnsObj = any(isnan(llPopObj), 1);
        llPopObj(:, nanColumnsObj) = [];
        
        % Deal with llPopCon
        nanColumnsCon = any(isnan(llPopCon), 1);
        llPopCon(:, nanColumnsCon) = [];

    end
   % Construct the Lower quadratic approximations of objective function and linear approximations of constraints
    approx.function       = QuadApprox(llPopObj, llPop);
    approx.equalityConstr = [];
    if size(llPopCon,2) ~= 0
        for i = 1 : size(llPopCon,2)
            approx.inequalityConstr{i} = QuadApprox(llPopCon(:,i), llPop);
        end
    else
        approx.inequalityConstr = [];
    end
    [x,~,~,output] = fmincon(@(x) approximatedFunction(x,approx.function),n_xl0,[],[],[],[],lb,ub,@(x) approximatedConstraints(x,approx.equalityConstr,approx.inequalityConstr),options);
    xl=x;
    llFunctionEvaluations=output.funcCount;

    try
       xl1 = xl(1:Problem.q+Problem.s);
       xl2 = xl(Problem.q+Problem.s+1:Problem.q+Problem.s+Problem.r);
    catch
       xl1 = xl(1:Problem.q);
       xl2 = xl(Problem.q+1:Problem.q+Problem.r);
    end

    PopObj = Problem.CalObj([xu1 xu2 xl1 xl2]);
    f=PopObj(:,1);
end

function approxFunctionValue = approximatedFunction(pop, parameters)
    approxFunctionValue = parameters.constant + pop*parameters.linear + pop*parameters.sqmatrix*pop';
end

%approximatedConstraints
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

%OperatorGA - Crossover and mutation operators of genetic algorithm.
function Offspring = BLSAEAGA(Parent,Parameter,llDimMin, llDimMax)
    % Parameter setting
    if nargin > 1
        [proC,disC,proM,disM] = deal(Parameter{:});
    else
        [proC,disC,proM,disM] = deal(1,20,1,20);
    end

    Parent1 = Parent(1:floor(end/2),:);% get the two step reason
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);

    % Genetic operators for real encoding  % Simulated binary crossover
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
    Lower = repmat(llDimMin,2*N,1);
    Upper = repmat(llDimMax,2*N,1);
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

### `llSearch.m`
```matlab
function [ best_xl, best_f,best_con,llFunctionEvaluations,ulFunctionEvaluations] = llSearch(xl0,flag,ulPop,P_llPop,Problem,llDim, llDimMin, llDimMax, xu,llPopSize,llStoppingCriteria,proC,disC,proM,disM,llFunctionEvaluations,llMaxGens,ulFunctionEvaluations )

    if flag==0

        %Lower population initialization (LHS sampling)
        Pl = UniformPoint(llPopSize,llDim,'Latin');
        llPop    = repmat(llDimMax-llDimMin,llPopSize,1).*Pl+repmat(llDimMin,llPopSize,1);

    else
        %random initialization
        if(llDim<=3)
            P = UniformPoint(13,llDim,'Latin');
            llPop    = repmat(llDimMax-llDimMin,13,1).*P+repmat(llDimMin,13,1);
        else
            Pl = UniformPoint(23,llDim,'Latin');
            llPop    = repmat(llDimMax-llDimMin,23,1).*Pl+repmat(llDimMin,23,1);
        end

        %Find the closest point in the upper population and the corresponding optimal solution in the lower level
        [ulPopSize,~]=size(ulPop);
        for e=1:ulPopSize
            c_o_s(e,1)=1-pdist([xu;ulPop(e,:)],'cosine');
        end
        [maxv,maxi] = max(c_o_s);
        Y=P_llPop(maxi,:);

        %disturbance
        sigma = 0.2*(llDimMax-llDimMin);
        for i=1:1
            mu       = rand(1,llDim) < 0.5;
            Ydec     = Y;
            Ydec(mu) = Ydec(mu) + sigma(mu).*randn(1,sum(mu));
            %Out of bounds processing
            for j=1:llDim
                if Ydec(j)<llDimMin(j)
                    Ydec(j)=llDimMin(j);

                elseif Ydec(j)>llDimMax(j)
                    Ydec(j)=llDimMax(j);
                end
            end

            llPop=[llPop;Ydec];
        end
        llPop=[llPop;Y];

        %disturbance
        if(llDim<=3)
            for i=1:14
                mu       = rand(1,llDim) < 0.5;
                Ydec     = xl0;
                Ydec(mu) = Ydec(mu) + sigma(mu).*randn(1,sum(mu));
                %Out of bounds processing
                for j=1:llDim
                    if Ydec(j)<llDimMin(j)
                        Ydec(j)=llDimMin(j);

                    elseif Ydec(j)>llDimMax(j)
                        Ydec(j)=llDimMax(j);
                    end
                end

                llPop=[llPop;Ydec];
            end
        else
            for i=1:24
                mu       = rand(1,llDim) < 0.5;
                Ydec     = xl0;
                Ydec(mu) = Ydec(mu) + sigma(mu).*randn(1,sum(mu));
                %Out of bounds processing
                for j=1:llDim
                    if Ydec(j)<llDimMin(j)
                        Ydec(j)=llDimMin(j);

                    elseif Ydec(j)>llDimMax(j)
                        Ydec(j)=llDimMax(j);
                    end
                end
                llPop=[llPop;Ydec];
            end
        end
        %Handling xl0 out of bounds
        for j=1:llDim
            if xl0(j)<llDimMin(j)
                xl0(j)=llDimMin(j);
            elseif xl0(j)>llDimMax(j)
                xl0(j)=llDimMax(j);
            end
        end

        llPop=[llPop;xl0];

    end

    %llevaluate
    [ll_obj, ~, ll_con]=llevaluate(llPop, Problem, xu);
    llFunctionEvaluations = llFunctionEvaluations+llPopSize;

    %L_fitness
    if isempty(ll_con)
        ll_fitness=ll_obj;

    else
        PopCon   = sum(max(0,ll_con),2);
        Feasible = PopCon <= 0;
        ll_fitness  = Feasible.*ll_obj + ~Feasible.*(PopCon+1e10);
    end

    %Record the optimal value trajectory of the population
    [~,index]=min(ll_fitness);
    current_best_f=ll_obj(index,:);
    history_best_f=current_best_f;

    v0 = var(llPop);
    alpha = 1;
    GEN=0;

    %%Iterate
    while alpha>llStoppingCriteria && GEN<=llMaxGens

        %TournamentSelection
        MatingPool = TournamentSelection(2,llPopSize,ll_fitness);
        %produce offspring
        Offspring  = BLSAEAGA(llPop(MatingPool',:),{proC,disC,proM,disM},llDimMin, llDimMax);

        %llevaluate
        [off_ll_obj, ~, off_ll_con]=llevaluate(Offspring, Problem, xu);
        llFunctionEvaluations = llFunctionEvaluations+llPopSize;
        %L_fitness
        if isempty(off_ll_con)
            off_ll_fitness=off_ll_obj;

        else
            PopCon   = sum(max(0,off_ll_con),2);
            Feasible = PopCon <= 0;
            off_ll_fitness  = Feasible.*off_ll_obj + ~Feasible.*(PopCon+1e10);
        end

        %Merge parent and child populations
        llPop = [llPop;Offspring];ll_fitness=[ll_fitness;off_ll_fitness];  ll_obj=[ll_obj;off_ll_obj];
        [~,rank]   = sort(ll_fitness);
        llPop = llPop(rank(1:llPopSize),:);  ll_obj = ll_obj(rank(1:llPopSize),:);  ll_fitness = ll_fitness(rank(1:llPopSize),:);

        current_best_xl=llPop(1,:);

        %SQP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ub = Problem.upper(Problem.DU+1:end);
        lb = Problem.lower(Problem.DU+1:end);

        options = optimoptions('fmincon','Algorithm','sqp','Display','none');
        % Construct the Lower quadratic approximations of objective function and linear approximations of constraints
        approx.function       = QuadApprox(ll_obj, llPop);
        approx.equalityConstr = [];
        if size(off_ll_con,2) ~= 0
            for i = 1 : size(off_ll_con,2)
                approx.inequalityConstr{i} = QuadApprox(off_ll_con(:,i), llPop);
            end
        else
            approx.inequalityConstr = [];
        end
        [x,~,~,output] = fmincon(@(x) approximatedFunction(x,approx.function),current_best_xl,[],[],[],[],lb,ub,@(x) approximatedConstraints(x,approx.equalityConstr,approx.inequalityConstr),options);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        L_current_best_xl=x;
        llFunctionEvaluations1=output.funcCount;

        llFunctionEvaluations=llFunctionEvaluations+llFunctionEvaluations1;

        [L_off_ll_obj, ~, L_off_ll_con]=llevaluate(L_current_best_xl, Problem, xu);
        if isempty(L_off_ll_con)
            L_off_ll_fitness=L_off_ll_obj;

        else
            PopCon   = sum(max(0,L_off_ll_con),2);
            Feasible = PopCon <= 0;
            L_off_ll_fitness  = Feasible.*L_off_ll_obj + ~Feasible.*(PopCon+1e10);
        end

        if ll_fitness(llPopSize,:)>L_off_ll_fitness
            llPop(llPopSize,:)=L_current_best_xl;
            ll_obj(llPopSize,:)=L_off_ll_obj;
            ll_fitness(llPopSize,:)=L_off_ll_fitness;
        end

        alpha = sum(var(llPop)./v0);
        if alpha>1
            alpha = 1;
        end

        GEN=GEN+1;

        [~,index]=min(ll_fitness);
        current_best_f=ll_obj(index,:);
        history_best_f=[history_best_f current_best_f];


        if GEN>9
            if  abs(history_best_f(end)-history_best_f(end-5))<1e-5
                break;
            end

        end

    end

    val=min(ll_fitness);
    value=abs(ll_fitness-val);
    k=find(value<1e-9);
    llPop=llPop(k,:);    ll_obj=ll_obj(k,:);
    [n,~]=size(llPop);
    A=[];
    for i=1:n
        [ul_obj, ~, ul_con]=ulevaluate(xu, llPop(i,:), Problem);
        A=[A ul_obj];
    end

    [~,index]=min(A);
    best_xl=llPop(index,:);
    best_f=ll_obj(index,:);

    [~, ~, best_con]=llevaluate(best_xl, Problem, xu);

    if isempty(best_con)
        best_con=0;
    end
end

function approxFunctionValue = approximatedFunction(pop, parameters)
    approxFunctionValue = parameters.constant + pop*parameters.linear + pop*parameters.sqmatrix*pop';
end

%approximatedConstraints
function [c, ceq] = approximatedConstraints(pop, parametersEqualityConstr, parametersInequalityConstr)
    if ~isempty(parametersEqualityConstr)%等式约束
        for i = 1 : length(parametersEqualityConstr)
            ceq(i) = parametersEqualityConstr{i}.constant + pop*parametersEqualityConstr{i}.linear + pop*parametersEqualityConstr{i}.sqmatrix*pop';
        end
    else
        ceq = [];
    end

    if ~isempty(parametersInequalityConstr)%不等式约束
        for i = 1 : length(parametersInequalityConstr)
            c(i) = parametersInequalityConstr{i}.constant + pop*parametersInequalityConstr{i}.linear + pop*parametersInequalityConstr{i}.sqmatrix*pop';
        end
    else
        c = [];
    end
end

%OperatorGA - Crossover and mutation operators of genetic algorithm.
function Offspring = BLSAEAGA(Parent,Parameter,llDimMin, llDimMax)
    % Parameter setting
    if nargin > 1
        [proC,disC,proM,disM] = deal(Parameter{:});
    else
        [proC,disC,proM,disM] = deal(1,20,1,20);
    end

    Parent1 = Parent(1:floor(end/2),:);% get the two step reason
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);
  
   
    % Genetic operators for real encoding  % Simulated binary crossover
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
    Lower = repmat(llDimMin,2*N,1);
    Upper = repmat(llDimMax,2*N,1);
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

### `llevaluate.m`
```matlab
function [functionValue,equalityConstrVals,inequalityConstrVals]=llevaluate(llPop, Problem, ulMember)

    %This function evaluates the lower level objective values and constraints
    %for a set of lower level members corresponding to a given upper level member.
    noOfMembers = size(llPop,1);
    %Function call here

    equalityConstrVals = [];
    inequalityConstrVals = [];  
    llPopSize = (Problem.DL+1)*(Problem.DL+2)/2+Problem.DL;
    for i=1:noOfMembers
        if size(llPop,1) == size(ulMember,1) 
            llPopulation = Problem.EvaluationLower([ulMember(i,:), llPop(i,:)]);
            llPopCon = llPopulation.cons;
            llPopObj = llPopulation.objs;
            if sum(sum(isnan(llPopCon)))>0 || sum(sum(isnan(llPopObj)))>0
                llPopObj = llPopObj(~isnan(llPopObj));
                llPopCon = llPopCon(~isnan(llPopCon));
            end
            if(isempty(llPopCon))
                equalityConstrValsTemp = [];
                inequalityConstrValsTemp = [];
            else
                equalityConstrValsTemp = [];
                inequalityConstrValsTemp = llPopCon;
            end
            functionValue(i,:) = llPopObj;

        elseif size(ulMember,1) == 1 
            llPopulation = Problem.EvaluationLower([ulMember, llPop(i,:)]);
            llPopObj = llPopulation.objs;
            llPopCon = llPopulation.cons;
            if sum(sum(isnan(llPopCon)))>0 || sum(sum(isnan(llPopObj)))>0
                llPopObj = llPopObj(~isnan(llPopObj));
                llPopCon = llPopCon(~isnan(llPopCon));
            end
            if(isempty(llPopCon))
                equalityConstrValsTemp = [];
                inequalityConstrValsTemp = [];
            else
                equalityConstrValsTemp = [];
                inequalityConstrValsTemp = llPopCon;
            end
            functionValue(i,:) = llPopObj;
            
        else
            disp('Error in llTestProblem, size of llPop and ulMember mismatch');
        end
        
        if ~isempty(equalityConstrValsTemp)
            equalityConstrVals(i,:) = equalityConstrValsTemp;
        end
        if ~isempty(inequalityConstrValsTemp)
            inequalityConstrVals(i,:) = inequalityConstrValsTemp;
        end
    end
end
```

### `srgtsKRGSetOptions.m`
```matlab
function srgtOPT = srgtsKRGSetOptions(P, T, FIT_Fn, ...
   KRG_RegressionModel, KRG_CorrelationModel, KRG_Theta0, KRG_LowerBound, KRG_UpperBound)
%Function srgtsKRGSetOptions creates the SURROGATES Toolbox option
%structure for kriging models. This structure contains the following fiels:
%
%* GENERAL PARAMETERS
%
%   SRGT   - Identifier of the surrogate technique: 'KRG'.
%   P      - NPOINTS-by-NDV matrix, where NPOINTS is the number of points
%            of the sample and NDV is the number of design variables.
%            Default: Empty matrix.
%   T      - NPOINTS-by-1 vector of responses on the P matrix points.
%            Default: Empty vector.
%   FIT_Fn - Function handle of the fitting function (which is used to
%            optimize KRG_Theta). [@dace_fit | @srgtsXVFit].
%            Default: @dace_fit.
%
%* KRIGING PARAMETERS
%
%   KRG_RegressionModel  - Function handle to a regression model. [
%                          function_handle | @dace_regpoly0 |
%                          @dace_regpoly1 | @dace_regpoly2]. Default:
%                          @dace_regpoly0.
%   KRG_CorrelationModel - Function handle to a correlation model. [
%                          function_handle | @dace_corrcubic |
%                          @dace_correxp | @dace_correxpg |
%                          @dace_corrgauss | @dace_corrlin |
%                          @dace_corrspherical | @dace_corrspline ].
%                          Default: @dace_corrgauss.
%   KRG_Theta0           - Initial guess for theta (correlation function
%                          parameters). Default:
%                          (NPOINTS^(-1/NDV))*ones(1, NDV).
%   KRG_LowerBound       - Lower bound for theta. Default: Empty vector.
%   KRG_UpperBound       - Upper bound for theta. Default: Empty vector.
%
%The SURROGATES Toolbox uses the DACE toolbox of Lophaven et al. (2002) to
%execute the kriging algorithm. As in DACE, when KRG_LowerBound and
%KRG_UpperBound are empty ([]) there will be NO optimization on theta
%(correlation function parameters).
%
%This is how you can use srgtsKRGSetOptions:
%
%     OPTIONS = srgtsKRGSetOptions: creates a structure with the empty
%     parameters.
%
%     OPTIONS = srgtsKRGSetOptions(P, T): Given the sampled data P (input
%     variables) and T (output variables), it creates a structure with
%     default parameters used for all not specified fields.
%
%     OPTIONS = srgtsKRGSetOptions(P, T, ..
%     KRG_UpperBound, FIT_Fn, FIT_LossFn KRG_RegressionModel, ...
%     KRG_CorrelationModel, KRG_Theta0, KRG_LowerBound):
%     it creates a structure with each of the specified fields.
%
%Example:
%     % basic information about the problem
%     myFN = @cos;  % this could be any user-defined function
%     designspace = [0;     % lower bound
%                    2*pi]; % upper bound
%
%     % create DOE
%     npoints = 5;
%     X = linspace(designspace(1), designspace(2), npoints)';
%
%     % evaluate analysis function at X points
%     Y = feval(myFN, X);
%
%     % fit surrogate models
%     options = srgtsKRGSetOptions(X, Y)
%
%     options =
%
%                     SRGT: 'KRG'
%                        P: [5x1 double]
%                        T: [5x1 double]
%      KRG_RegressionModel: @dace_regpoly0
%     KRG_CorrelationModel: @dace_corrgauss
%               KRG_Theta0: 0.2000
%           KRG_LowerBound: []
%           KRG_UpperBound: []
%
%REFERENCES:
%
%Lophaven SN, Nielsen HB, and Sndergaard J, DACE - A MATLAB Kriging
%Toolbox, Technical Report IMM-TR-2002-12, Informatics and Mathematical
%Modelling, Technical University of Denmark, 2002.
%Available at: http://www2.imm.dtu.dk/~hbn/dace/.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Felipe A. C. Viana
% felipeacviana@gmail.com
% http://sites.google.com/site/felipeacviana
%
% This program is free software; you can redistribute it and/or
% modify it. This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data
srgtOPT.SRGT = 'KRG';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options

    switch nargin
        case 0
            srgtOPT.P = [];
            srgtOPT.T = [];

            srgtOPT.FIT_Fn = [];

            srgtOPT.KRG_RegressionModel  = [];
            srgtOPT.KRG_CorrelationModel = [];
            srgtOPT.KRG_Theta0           = [];
            srgtOPT.KRG_LowerBound       = [];
            srgtOPT.KRG_UpperBound       = [];

        case 2
            [npoints nvariables] = size(P);

            srgtOPT.P = P;
            srgtOPT.T = T;

            srgtOPT.FIT_Fn = @dace_fit;

            srgtOPT.KRG_RegressionModel  = @dace_regpoly0;
            srgtOPT.KRG_CorrelationModel = @dace_corrgauss;
            srgtOPT.KRG_Theta0           = (npoints^(-1/nvariables))*ones(1, nvariables);
            srgtOPT.KRG_LowerBound       = [];
            srgtOPT.KRG_UpperBound       = [];

        otherwise
            srgtOPT.P = P;
            srgtOPT.T = T;

            srgtOPT.FIT_Fn = FIT_Fn;

            srgtOPT.KRG_RegressionModel  = KRG_RegressionModel;
            srgtOPT.KRG_CorrelationModel = KRG_CorrelationModel;
            srgtOPT.KRG_Theta0           = KRG_Theta0;
            srgtOPT.KRG_LowerBound       = KRG_LowerBound;
            srgtOPT.KRG_UpperBound       = KRG_UpperBound;
    end
end
```

### `ulevaluate.m`
```matlab
function [functionValue equalityConstrVals inequalityConstrVals]=ulevaluate(ulPop, llPop, Problem)

    %This function evaluates the upper level objective values and constraints
    %for a set of upper level members and their corresponding lower level members.
    %Function call here
    noOfMembers = size(ulPop,1);
    
    equalityConstrVals = [];
    inequalityConstrVals = [];
    l=0;
    for i=1:noOfMembers
        Population = Problem.Evaluation([ulPop(i, :), llPop(i, :)]);
        
        llPopulation = Problem.EvaluationLower([ulPop(i, :), llPop(i, :)]);
        llPopCon = llPopulation.cons;
        if sum(sum(isnan(llPopCon)))>0 || sum(sum(isnan(llPop)))>0
            llPopCon = llPopCon(~isnan(llPopCon));
        end
        l=length(llPopCon);

        PopCon = Population.cons;
        PopObj = Population.objs; 
        functionValue(i,:)=PopObj(:,1);
        inequalityConstrValsTemp = PopCon(:,1:l);
        equalityConstrValsTemp = [];

        if ~isempty(equalityConstrValsTemp)
            equalityConstrVals(i,:) = equalityConstrValsTemp;
        end
        if ~isempty(inequalityConstrValsTemp)
            inequalityConstrVals(i,:) = inequalityConstrValsTemp;
        end
    end
end
```
