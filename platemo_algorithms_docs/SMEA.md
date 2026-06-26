# SMEA

**Tags**: <2016> <multi> <real/integer>

## Description
Self-organizing multiobjective evolutionary algorithm

## Reference
H. Zhang, A. Zhou, S. Song, Q. Zhang, X. Gao, and J. Zhang. A self- organizing multiobjective evolutionary algorithm. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 792-806.

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
            j = 1:i-1; 
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

### `SMEA.m`
```matlab
classdef SMEA < ALGORITHM
% <2016> <multi> <real/integer>
% Self-organizing multiobjective evolutionary algorithm
% D    ---     --- Number of neurons in each dimension of the latent space
% tau0 --- 0.7 --- Initial learning rate
% H    ---   5 --- Size of neighborhood mating pools

%------------------------------- Reference --------------------------------
% H. Zhang, A. Zhou, S. Song, Q. Zhang, X. Gao, and J. Zhang. A self-
% organizing multiobjective evolutionary algorithm. IEEE Transactions on
% Evolutionary Computation, 2016, 20(5): 792-806.
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
            [D,tau0,H] = Algorithm.ParameterSet(repmat(ceil(Problem.N.^(1/(Problem.M-1))),1,Problem.M-1),0.7,5);
            Problem.N  = prod(D);
            sigma0     = sqrt(sum(D.^2)/(Problem.M-1))/2;

            %% Generate random population
            Population = Problem.Initialization();
            FrontNo    = NDSort(Population.objs,inf);

            %% Initialize the SOM
            % Training set
            S = Population.decs;
            % Weight vector of each neuron
            W = S;
            % Position of each neuron
            D = arrayfun(@(S)1:S,D,'UniformOutput',false);
            eval(sprintf('[%s]=ndgrid(D{:});',sprintf('c%d,',1:length(D))))
            eval(sprintf('Z=[%s];',sprintf('c%d(:),',1:length(D))))
            % Distance between each two neurons in latent space
            LDis = pdist2(Z,Z);
            % H nearest neurons of each neuron in latent space
            [~,B] = sort(LDis,2);
            B     = B(:,2:min(H+1,end));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Update SOM
                for s = 1 : size(S,1)
                    sigma  = sigma0*(1-(Problem.FE+s)/Problem.maxFE);
                    tau    = tau0*(1-(Problem.FE+s)/Problem.maxFE);
                    [~,u1] = min(pdist2(S(s,:),W));
                    U      = LDis(u1,:) < sigma;
                    W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)).*(repmat(S(s,:),sum(U),1)-W(U,:));
                end

                % Associate each solution with a neuron
                A  = 1 : Problem.N;
                U  = 1 : Problem.N;
                XU = zeros(1,Problem.N);
                for i = 1 : Problem.N
                    x        = randi(length(A));
                    [~,u]    = min(pdist2(Population(A(x)).dec,W(U,:)));
                    XU(U(u)) = A(x);
                    A(x)     = [];
                    U(u)     = [];
                end

                % Evolution
                A = Population.decs;
                for u = 1 : Problem.N
                    drawnow('limitrate');
                    if rand < 0.9
                        Q = XU(B(u,:));
                    else
                        Q = 1 : Problem.N;
                    end
                    Q = Q(randperm(end,2));
                    y = OperatorDE(Problem,Population(u),Population(Q(1)),Population(Q(2)));
                    [Population,FrontNo] = Select(Population,FrontNo,y);
                end

                % Update the training set
                S = setdiff(Population.decs,A,'rows');
            end
        end
    end
end
```

### `Select.m`
```matlab
function [Population,FrontNo] = Select(Population,FrontNo,y)
% And one offspring to the population and delete the worst one

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Identify the solutions in the last front
    Population = [Population,y];
    PopObj     = Population.objs;
    [N,M]      = size(PopObj);
    FrontNo    = UpdateFront(PopObj,FrontNo);
    
    %% Delete the worst solution
    if max(FrontNo) > 1
        Distance  = pdist2(PopObj,PopObj);
        Distance(logical(eye(N))) = inf;
        LastFront = find(FrontNo==max(FrontNo));
        [~,worst] = min(min(Distance(LastFront,:),[],2));
        worst     = LastFront(worst);
    else
        deltaS = inf(1,N);
        if M == 2
            [~,rank] = sortrows(PopObj);
            for i = 2 : N-1
                deltaS(rank(i)) = (PopObj(rank(i+1),1)-PopObj(rank(i),1)).*(PopObj(rank(i-1),2)-PopObj(rank(i),2));
            end
        elseif N > 1
            deltaS = CalHV(PopObj,max(PopObj,[],1)*1.1,1,10000);
        end
        [~,worst] = min(deltaS);
    end
    FrontNo = UpdateFront(PopObj,FrontNo,worst);
    Population(worst) = [];
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
