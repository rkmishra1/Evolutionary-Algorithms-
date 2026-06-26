# MOEA-D-CMA

**Tags**: <2017> <multi/many> <real/integer>

## Description
MOEA/D with covariance matrix adaptation evolution strategy

## Reference
H. Li, Q. Zhang, and J. Deng. Biased multiobjective optimization and decomposition algorithm. IEEE Transactions on Cybernetics, 2017, 47(1): 52-66.

## Source Code

### `MOEADCMA.m`
```matlab
classdef MOEADCMA < ALGORITHM
% <2017> <multi/many> <real/integer>
% MOEA/D with covariance matrix adaptation evolution strategy
% K --- 5 --- Number of groups

%------------------------------- Reference --------------------------------
% H. Li, Q. Zhang, and J. Deng. Biased multiobjective optimization and
% decomposition algorithm. IEEE Transactions on Cybernetics, 2017, 47(1):
% 52-66.
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
            K = Algorithm.ParameterSet(5);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);
            % Transformation on W
            W = 1./W./repmat(sum(1./W,2),1,size(W,2));
            % Cluster the subproblems
            G = kmeans(W,K);
            G = arrayfun(@(S)find(G==S),1:K,'UniformOutput',false);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% Initial the CMA model
            sk    = cellfun(@(S)S(randi(length(S))),G);
            xk    = Population(sk).decs;
            Sigma = struct('s',num2cell(sk),'x',num2cell(xk,2)','sigma',0.5,'C',eye(Problem.D),'pc',0,'ps',0);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                for s = 1 : Problem.N
                    k = find([Sigma.s]==s);
                    if ~isempty(k)
                        P = B(s,randperm(size(B,2)));
                        % Generate offsprings by CMA
                        Offspring = Problem.Evaluation(mvnrnd(Sigma(k).x,Sigma(k).sigma^2*Sigma(k).C,4+floor(3*log(Problem.D))));
                        % Sort the parent and offsprings
                        Combine   = [Offspring,Population(s)];
                        [~,rank]  = sort(max(abs(Combine.objs-repmat(Z,length(Combine),1)).*repmat(W(s,:),length(Combine),1),[],2));
                        % Update the CMA model
                        Sigma(k)  = UpdateCMA(Combine(rank).decs,Sigma(k),ceil(Problem.FE/Problem.N));
                        if isempty(Sigma(k).s)
                            sk = G{k}(randi(length(G{k})));
                            Sigma(k).s = sk;
                            Sigma(k).x = Population(sk).dec;
                        end
                    else
                        % Generate an offspring by DE
                        if rand < 0.9
                            P = B(s,randperm(size(B,2)));
                        else
                            P = randperm(Problem.N);
                        end
                        Offspring = OperatorDE(Problem,Population(s),Population(P(1)),Population(P(2)));
                    end
                    for x = 1 : length(Offspring)
                        % Update the ideal point
                        Z = min(Z,Offspring(x).obj);
                        % Update the solutions in P by Tchebycheff approach
                        g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                        g_new = max(repmat(abs(Offspring(x).obj-Z),length(P),1).*W(P,:),[],2);
                        Population(P(find(g_old>=g_new,2))) = Offspring(x);
                    end
                end
            end
        end
    end
end
```

### `UpdateCMA.m`
```matlab
function Sigma = UpdateCMA(X,Sigma,gen)
% Update the CMA model

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    n = size(X,2);
    
    %% Calculate the CMA parameters
    mu    = 4 + floor(3*log(n));
    mu1   = floor(mu/2);
    w     = log((mu+1)/2) - log(1:mu1);
    w     = w./sum(w);
    mueff = 1./sum(w.^2);
    cs    = (mueff+2)./(n+mueff+5);
    ds    = 1 + 2*max(0,sqrt((mueff-1)./(n+1))-1) + cs;
    cc    = (4+mueff/n)./(n+4+2*mueff/n);
    c1    = 2./((n+1.3).^2+mueff);
    cmu   = min(1-c1,2*(mueff-2+1/mueff)./((n+2).^2+mueff)); % Modified
    ENI   = sqrt(n)*(1-1/4/n+1/21/n^2);
    
    %% Update the CMA model
    y           = (X(1:mu1,:)-repmat(Sigma.x,mu1,1))/Sigma.sigma;
    yw          = w*y;
    Sigma.x     = Sigma.x + Sigma.sigma*yw;
    Sigma.ps    = (1-cs)*Sigma.ps + sqrt(cs*(2-cs)*mueff)*Sigma.C^(-1/2)*yw';
    hs          = norm(Sigma.ps)./sqrt(1-(1-cs).^(2*(gen+1))) < (1.4+2/(n+1))*ENI;
    deltahs     = 1 - hs; % Modified
    Sigma.pc    = (1-cc)*Sigma.pc + hs*sqrt(cc*(2-cc)*mueff)*yw;
    Sigma.sigma = Sigma.sigma*exp(cs/ds*(norm(Sigma.ps)/ENI-1));
    Sigma.C     = (1-c1-cmu)*Sigma.C + c1*(Sigma.pc'*Sigma.pc+deltahs*Sigma.C) + cmu*y'*diag(w)*y;
    Sigma.C     = triu(Sigma.C) + triu(Sigma.C,1)'; % Enforce symmetry
    
    %% Reset the CMA model if possible
    [B,D] = eig(Sigma.C);
    diagD = diag(D);
    diagC = diag(Sigma.C);
    ConditionCov  = max(diagD) > 1e14*min(diagD);
    NoEffectCoord = any(Sigma.x==Sigma.x+0.2*Sigma.sigma*sqrt(diagC)');
    NoEffectAxis  = all(Sigma.x==Sigma.x+0.1*Sigma.sigma*sqrt(diagD(mod(gen,n)+1))*B(:,mod(gen,n)+1)');
    TolXUp        = any(Sigma.sigma*sqrt(diagC)>1e4);
    if ConditionCov || NoEffectCoord || NoEffectAxis || TolXUp
        Sigma = struct('s',[],'x',[],'sigma',0.5,'C',eye(n),'pc',0,'ps',0);
    end
end
```
