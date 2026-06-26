# NNDREA-MO

**Tags**: <2025> <multi> <binary> <large/none> <constrained/none> <sparse/none>

## Description
Evolutionary algorithm with neural network-based dimensionality reduction

## Reference
Y. Tian, L. Wang, S. Yang, J. Ding, Y. Jin, and X. Zhang. Neural network-based dimensionality reduction for large-scale binary optimization with millions of variables. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2328-2342.

## Source Code

### `CalHV.m`
```matlab
function F = CalHV(points,bounds,k,nSample)
% Calculate the hypervolume-based fitness value of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/

    [N,M] = size(points);
    if M > 2
        % Use the estimated method for three or more objectives
        alpha = zeros(1,N); 
        for i = 1 : k 
            alpha(i) = prod((k-[1:i-1])./(N-[1:i-1]))./i; 
        end
        Fmin = min(points,[],1);
        S    = unifrnd(repmat(Fmin,nSample,1),repmat(bounds,nSample,1));
        PdS  = false(N,nSample);
        dS   = zeros(1,nSample);
        for i = 1 : N
            x        = sum(repmat(points(i,:),nSample,1)-S<=0,2) == M;
            PdS(i,x) = true;
            dS(x)    = dS(x) + 1;
        end
        F = zeros(1,N);
        for i = 1 : N
            F(i) = sum(alpha(dS(PdS(i,:))));
        end
        F = F.*prod(bounds-Fmin)/nSample;
    else
        % Use the accurate method for two objectives
        pvec  = 1:size(points,1);
        alpha = zeros(1,k);
        for i = 1 : k 
            j = 1 : i-1; 
            alpha(i) = prod((k-j)./(N-j))./i;
        end
        F = hypesub(N,points,M,bounds,pvec,alpha,k);
    end
end

function h = hypesub(l,A,M,bounds,pvec,alpha,k)
% The recursive function for the accurate method

    h     = zeros(1,l); 
    [S,i] = sortrows(A,M); 
    pvec  = pvec(i); 
    for i = 1 : size(S,1) 
        if i < size(S,1) 
            extrusion = S(i+1,M) - S(i,M); 
        else
            extrusion = bounds(M) - S(i,M);
        end
        if M == 1
            if i > k
                break; 
            end
            if alpha >= 0
                h(pvec(1:i)) = h(pvec(1:i)) + extrusion*alpha(i); 
            end
        elseif extrusion > 0
            h = h + extrusion*hypesub(l,S(1:i,:),M-1,bounds,pvec(1:i),alpha,k); 
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Decs,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Decs,N)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
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
    Decs       = Decs(Next, :);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
```

### `FCNForward.m`
```matlab
 function Output = FCNForward(Input, instance, s_list)
% Fully connected neural network forward propagation
% The function input includes a set of neural network weights, 
% problem information, and a list of neural network structures
    
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    % Initialize output
    Output = zeros(size(Input, 1), size(instance, 1));
    for n = 1 : size(Input, 1)
        output = instance;
        % Initialize pointer
        pointer = 1;
        for i = 1 : size(s_list, 1)
            % Traverse the list of neural network structures
            if s_list(i, 2) ~= -1
                % Check if there is only bias, if not then
                weight  = Input(n, pointer:pointer + s_list(i, 1)*s_list(i, 2)-1);
                pointer = pointer + s_list(i, 1)*s_list(i, 2);
                output  = output * reshape(weight, [s_list(i, 1), s_list(i, 2)]);
            else
                % only bias then
                bias    = Input(n, pointer:pointer + s_list(i, 1)-1);
                pointer = pointer + s_list(i, 1);
                output  = output + bias;
                % Check if it is the last layer of the neural network
                if i == size(s_list, 1)
                    % If it is the last layer, output after activation
                    output       = step(output);
                    Output(n, :) = output;
                else
                    output = leaky_relu(output);
                end
            end
        end
    end
end

function y = leaky_relu(x)
    y = x .* (x>0) + 0.01 * x .* (x<=0);
end

function y = step(x)
    y = x > 0;
end
```

### `NNDREAMO.m`
```matlab
classdef NNDREAMO < ALGORITHM
% <2025> <multi> <binary> <large/none> <constrained/none> <sparse/none>
% Evolutionary algorithm with neural network-based dimensionality reduction
% lower ---  -1 --- Lower bound of network weights
% upper ---   1 --- Upper bound of network weights
% delta --- 0.5 --- Proportion of the first stage

%------------------------------- Reference --------------------------------
% Y. Tian, L. Wang, S. Yang, J. Ding, Y. Jin, and X. Zhang. Neural
% network-based dimensionality reduction for large-scale binary
% optimization with millions of variables. IEEE Transactions on
% Evolutionary Computation, 2025, 29(6): 2328-2342.
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
            %% Parameter setting
            [lower, upper, delta] = Algorithm.ParameterSet(-1, 1, 0.5);
            % Extracting information from problems
            switch class(Problem)
                case {'MOKP','Sparse_KP'}
                    Type     = 1;
                    Instance = [Problem.P; Problem.W].';
                case 'Sparse_IS'
                    Type     = 1;
                    Instance = sparse(Problem.Data);
                case 'Sparse_CD'
                    Type    = 2;
                    AdjMat  = Problem.Adj;
                case 'Sparse_CN'
                    Type    = 2;
                    AdjMat  = Problem.A;
                otherwise
                    error('The %s problem cannot be solved by this algorithm',class(Problem));
            end
            % Feature extraction
            if Type == 1       % Random perturbation for non-graph problems
                Instance = normrnd(Instance,0.1);
            elseif Type == 2   % Feature extraction for graph problems
                AdjMat   = sparse(AdjMat);
                D_mat    = zeros(size(AdjMat));
                D_mat(logical(eye(size(D_mat)))) = sum(AdjMat,2);
                D_mat    = D_mat ^ -0.5;
                D_mat    = sparse(D_mat);
                [V,DM]   = eig(full(D_mat*AdjMat*D_mat));
                [~,ind]  = sort(diag(DM));
                Instance = V(:,ind(1:10));
            end
            % Set the neural network structure
            structure = [size(Instance, 2), 4, 1];
            s_list    = ones((size(structure, 2)-1)*2, 2) * -1;
            % Get list of neural network structures and dimension after
            % neural network weight flattening
            Dim = 0;
            for i = 1 : size(structure, 2)-1
                s_list(2*i-1,1) = structure(i);
                s_list(2*i-1,2) = structure(i+1);
                s_list(2*i,1)   = structure(i+1);
                Dim = Dim + structure(i) * structure(i+1) + structure(i+1);
            end
            % Get lower and upper bounds of the search space
            lower = repmat(lower,1,Dim);
            upper = repmat(upper,1,Dim);

            %% Generate random population
            % Initialize population weights
            PopWeight = unifrnd(repmat(lower,Problem.N,1),repmat(upper,Problem.N,1));
            % Fully connected neural network for forward propagation
            Output = FCNForward(PopWeight, Instance, s_list);
            % Evaluate the population
            Population = Problem.Evaluation(Output);
            [~,~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,PopWeight,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                if Problem.FE <= Problem.maxFE * delta
                    % First stage
                    MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                    OffWeight  = OperatorReal(PopWeight(MatingPool, :),lower, upper);
                    Output     = FCNForward(OffWeight, Instance, s_list);
                    Offspring  = Problem.Evaluation(Output);
                    [Population,PopWeight,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],[PopWeight;OffWeight], Problem.N);
                else
                    % Second stage
                    for i = 1 : Problem.N
                        drawnow('limitrate');
                        Offspring = OperatorGAhalf(Problem,Population(randperm(end,2)));
                        if all(Offspring.con<=0)
                            [Population,FrontNo] = Reduce([Population,Offspring],FrontNo);
                        end
                    end
                end
            end
        end
    end
end
```

### `OperatorReal.m`
```matlab
function OffspringDec = OperatorReal(Parent,lower,upper)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Parent1      = Parent(1:floor(end/2),:);
    Parent2      = Parent(floor(end/2)+1:floor(end/2)*2,:);
    OffspringDec = GAreal(Parent1,Parent2,lower,upper,1,20,1,20);
end

function Offspring = GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM)
% Genetic operators for real variables

    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
             
    %% Polynomial mutation
    Lower = repmat(lower,2*N,1);
    Upper = repmat(upper,2*N,1);
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

### `OperatorRealHalf.m`
```matlab
function OffspringDec = OperatorRealHalf(Parent,lower,upper)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Parent1      = Parent(1:floor(end/2),:);
    Parent2      = Parent(floor(end/2)+1:floor(end/2)*2,:);
    OffspringDec = GArealHalf(Parent1,Parent2,lower,upper,1,20,1,20);
end

function Offspring = GArealHalf(Parent1,Parent2,lower,upper,proC,disC,proM,disM)
% Genetic operators for real and integer variables

    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
             
    %% Polynomial mutation
    Lower = repmat(lower,N,1);
    Upper = repmat(upper,N,1);
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

### `Reduce.m`
```matlab
function [Population,FrontNo] = Reduce(Population,FrontNo)
% Delete one solution from the population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Identify the solutions in the last front
    FrontNo   = UpdateFront(Population.objs,FrontNo);
    LastFront = find(FrontNo==max(FrontNo));
    PopObj    = Population(LastFront).objs;
    [N,M]     = size(PopObj);
    
    %% Calculate the contribution of hypervolume of each solution
    deltaS = inf(1,N);
    if M == 2
        [~,rank] = sortrows(PopObj);
        for i = 2 : N-1
            deltaS(rank(i)) = (PopObj(rank(i+1),1)-PopObj(rank(i),1)).*(PopObj(rank(i-1),2)-PopObj(rank(i),2));
        end
    elseif N > 1
        deltaS = CalHV(PopObj,max(PopObj,[],1)*1.1,1,10000);
    end
    
    %% Delete the worst solution from the last front
    [~,worst] = min(deltaS);
    FrontNo   = UpdateFront(Population.objs,FrontNo,LastFront(worst));
    Population(LastFront(worst)) = [];
end
```

### `UpdateFront.m`
```matlab
function FrontNo = UpdateFront(PopObj,FrontNo,x)
% Update the front No. of each solution when a solution is added or deleted

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    if nargin < 3
        %% Add a new solution (has been stored in the last of PopObj)
        FrontNo  = [FrontNo,0];
        Move     = false(1,N);
        Move(N)  = true;
        CurrentF = 1;
        % Locate the front No. of the new solution
        while true
            Dominated = false;
            for i = 1 : N-1
                if FrontNo(i) == CurrentF
                    m = 1;
                    while m <= M && PopObj(i,m) <= PopObj(end,m)
                        m = m + 1;
                    end
                    Dominated = m > M;
                    if Dominated
                        break;
                    end
                end
            end
            if ~Dominated
                break;
            else
                CurrentF = CurrentF + 1;
            end
        end
        % Move down the dominated solutions front by front
        while any(Move)
            NextMove = false(1,N);
            for i = 1 : N
                if FrontNo(i) == CurrentF
                    Dominated = false;
                    for j = 1 : N
                        if Move(j)
                            m = 1;
                            while m <= M && PopObj(j,m) <= PopObj(i,m)
                                m = m + 1;
                            end
                            Dominated = m > M;
                            if Dominated
                                break;
                            end
                        end
                    end
                    NextMove(i) = Dominated;
                end
            end
            FrontNo(Move) = CurrentF;
            CurrentF      = CurrentF + 1;
            Move          = NextMove;
        end
    else
        %% Delete the x-th solution
        Move     = false(1,N);
        Move(x)  = true;
        CurrentF = FrontNo(x) + 1;
        while any(Move)
            NextMove = false(1,N);
            for i = 1 : N
                if FrontNo(i) == CurrentF
                    Dominated = false;
                    for j = 1 : N
                        if Move(j)
                            m = 1;
                            while m <= M && PopObj(j,m) <= PopObj(i,m)
                                m = m + 1;
                            end
                            Dominated = m > M;
                            if Dominated
                                break;
                            end
                        end
                    end
                    NextMove(i) = Dominated;
                end
            end
            for i = 1 : N
                if NextMove(i)
                    Dominated = false;
                    for j = 1 : N
                        if FrontNo(j) == CurrentF-1 && ~Move(j)
                            m = 1;
                            while m <= M && PopObj(j,m) <= PopObj(i,m)
                                m = m + 1;
                            end
                            Dominated = m > M;
                            if Dominated
                                break;
                            end
                        end
                    end
                    NextMove(i) = ~Dominated;
                end
            end
            FrontNo(Move) = CurrentF - 2;
            CurrentF      = CurrentF + 1;
            Move          = NextMove;
        end
        FrontNo(x) = [];
    end
end
```
