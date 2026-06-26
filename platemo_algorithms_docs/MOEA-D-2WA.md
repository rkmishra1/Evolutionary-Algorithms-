# MOEA-D-2WA

**Tags**: <2021> <multi/many> <real/integer/label/binary/permutation> <constrained>

## Description
MOEA/D with two-type weight vector adjustments

## Reference
R. Jiao, S. Zeng, C. Li, and Y. S. Ong. Two-type weight adjustments in MOEA/D for highly constrained many-objective optimization. Information Sciences, 2021, 578: 592-614.

## Source Code

### `MOEAD2WA.m`
```matlab
classdef MOEAD2WA < ALGORITHM
% <2021> <multi/many> <real/integer/label/binary/permutation> <constrained>
% MOEA/D with two-type weight vector adjustments

%------------------------------- Reference --------------------------------
% R. Jiao, S. Zeng, C. Li, and Y. S. Ong. Two-type weight adjustments in 
% MOEA/D for highly constrained many-objective optimization. Information
% Sciences, 2021, 578: 592-614.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    methods
        function main(Algorithm, Problem)
            %% Generate the weight vectors
            OrN            = Problem.N;
            [W, Problem.N] = WeightGeneration(Problem.N, Problem.N, Problem.M, 1.0);

            %% Detect the neighbours of each solution
            T      = 20;
            nr     = 2;
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1:T);
            
            %% Generate random population
            Population    = Problem.Initialization();
            Z             = min(Population.objs, [], 1);
            [initialE, ~] = max(max(0, Population.cons), [], 1);
            initialE(initialE < 1) = 1;
            nCon          = size(Population.cons, 2);
            CV            = min(sum(max(0, Population.cons)./initialE, 2)./nCon, [], 1);
            Z             = [Z, CV]; 
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Reduce the dynamic constraint boundry
                epsn = ReduceBoundary(initialE, Problem.FE, Problem.maxFE - OrN);
                
                % Update weights
                a    = (W(:, Problem.M + 1) - 1e-6) > epsn(:, 1)./initialE(:, 1);
                W(a, :) = [];
                
                % Update population
                if Problem.N > size(W, 1)
                    Problem.N  = size(W, 1);
                    B          = pdist2(W, W);
                    [~, B]     = sort(B, 2);
                    B          = B(:, 1:T);
                    Population = PopulationUpdate(Population, Problem.N, initialE, epsn, Z(:, 1:Problem.M));
                end
                % For each solution
                for i = 1 : Problem.N     
                    % Choose the parents
                    if rand < 0.9
                        P = B(i,randperm(size(B, 2)));
                    else
                        P = randperm(Problem.N);
                    end

                    % Generate an offspring
                    Offspring = OperatorGAhalf(Problem,Population(P(1:2)));

                    % Update the ideal point
                    Z(:, 1:Problem.M)   = min(Z(:, 1:Problem.M), Offspring.obj);
                    CV                  = min(sum(max(0, Offspring.cons)./initialE, 2)./nCon, [], 1);
                    Z(:, Problem.M + 1) = min(Z(:, Problem.M + 1), CV);

                    % Calculate the constraint violation of offspring and P
                    cvo = max(0, Offspring.con);
                    cvp = max(0, Population(P).cons); 
                    cvO = sum(max(0, Offspring.cons)./initialE, 2)./nCon;
                    cvP = sum(max(0, Population(P).cons)./initialE, 2)./nCon;

                    % Update the solutions in P by PBI approach
                    PObj    = [Population(P).objs, cvP];
                    OObj    = [Offspring.obj, cvO];
                    normW   = sqrt(sum(W(P, :).^2, 2));
                    normP   = sqrt(sum((PObj - repmat(Z, length(P), 1)).^2, 2));
                    normO   = sqrt(sum((OObj - Z).^2, 2));
                    CosineP = sum((PObj - repmat(Z, length(P), 1)).*W(P, :),2)./normW./normP;
                    CosineO = sum(repmat(OObj - Z,length(P), 1).*W(P, :),2)./normW./normO;
                    g_old   = normP.*CosineP + 5*normP.*sqrt(1 - CosineP.^2);
                    g_new   = normO.*CosineO + 5*normO.*sqrt(1 - CosineO.^2);
                    % Neighbor solution replacement
                    index   = find((sum(cvo<=epsn, 2)==nCon&sum(cvp<=epsn, 2)==nCon&g_old>=g_new)|(sum(cvo<=epsn, 2)==nCon&sum(cvp<=epsn, 2)<nCon)|(sum(cvo<=epsn,2)<nCon&sum(cvp<=epsn, 2)<nCon&sum(max(0, Population(P).cons), 2)>sum(max(0, Offspring.con))), nr);
                    Population(P(index)) = Offspring;
                end   
            end  
        end
    end
end


function epsn = ReduceBoundary(eF, k, MaxK)
% The shrink of the dynamic constraint boundary  
    cp       = 4;
    z        = 1e-8;
    Nearzero = 1e-15;
    B        = MaxK./power(log((eF + z)./z), 1.0./cp);
    B(B==0)  = B(B==0) + Nearzero;
    f        = eF.* exp( -(k./B).^cp );
    tmp      = find(abs(f-z) < Nearzero);
    f(tmp)   = f(tmp).*0 + z;
    epsn     = f - z;
    epsn(epsn<=0) = 0;
end
```

### `PopulationUpdate.m`
```matlab
function Population = PopulationUpdate(Population, N, initialE, epsn, Zmin)
% Update the population

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    nCon   = size(Population.cons, 2);
    ConVio = max(0,Population.cons);
    if sum(sum(ConVio<=epsn, 2)==nCon) >= N
        %% Selection among epsilon-feasible solutions
        tmp        = sum(ConVio<=epsn, 2)==nCon;
        Population = Population(1:end, tmp);
        PopCv      = sum(max(0, Population.cons)./initialE, 2)./nCon;
        
        %% calculate angle between each two solutions objective values
        Next(1:size(Population, 2)) = true;
        PopObj                      = Population(Next).objs;
        Delete                      = LastSelection(PopObj, sum(Next) - N, Zmin, PopCv);
        Temp                        = find(Next);
        Next(Temp(Delete))          = false;
        Population                  = Population(Next);
    else
        %% Selection including epsilon-infeasible solutions
        CV         = sum(max(0, Population.cons)./initialE, 2)./nCon;
        [~, rank]  = sort(CV);
        % Population for next generation
        Population = Population(rank(1:N));
    end
end

function Delete = LastSelection(PopObj, K, Zmin, PopCv)
    %% calculate angle between each two solutions based on objective values
    N      = size(PopObj, 1);
    PopObj = (PopObj - repmat(Zmin, N, 1))./(repmat(max(PopObj), N, 1) - repmat(Zmin, N, 1));
    Cosine = 1 - pdist2(PopObj, PopObj, 'cosine');
    Cosine = Cosine.*(1 - eye(size(PopObj, 1)));

    % Environmental selection
    Delete  = false(1, N);
    % Delete K solutions one by one with smallest angle
    while sum(Delete) < K
        [Jmin_row, Jmin_column] = find(Cosine==max(max(Cosine)));
        j      = randi(length(Jmin_row));
        Temp_1 = Jmin_row(j);
        Temp_2 = Jmin_column(j);
        if PopCv(Temp_1) > PopCv(Temp_2)
            Delete(Temp_1)    = true;
            Cosine(:, Temp_1) = 0;
            Cosine(Temp_1, :) = 0;
        else
            Delete(Temp_2)    = true;
            Cosine(:, Temp_2) = 0;
            Cosine(Temp_2, :) = 0;
        end 
    end
end
```

### `WeightGeneration.m`
```matlab
function [W1, N] = WeightGeneration(N, AdN, M, thetaM)
%Generate a set of uniform randomly distributed points on the objective and
%constraint violation space

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Ruwang Jiao

    SampleNum = 5000;
    W1        = UniformPoint(N, M);
    W1        = [W1 zeros(size(W1, 1), 1)];
    W2        = CalWeight1(thetaM, M+1, SampleNum);
    W1        = [W1; [zeros(1, M) 1]];
    while size(W1, 1) < N + AdN
        index = find_index_with_largest_angle (W1, W2);
        W1(size(W1, 1) + 1, :) = W2(index, :);
        W2(index, :) = [];
    end	
    
    W1 = max(W1, 1e-6);
    N  = size(W1, 1);
end

function theta = CalWeight1(thetaM, M, SampleNum)
    theta           = rand(SampleNum, M);
    theta           = theta./repmat(sum(theta, 2), 1, size(theta, 2));
    SumWeiFormer    = 1 - theta(:, M);
    SumWeiAfter     = 1 - theta(:, M).*thetaM;
    theta(:, 1:M-1) = SumWeiAfter./SumWeiFormer.*theta(:, 1:M-1);
    theta(:, M)     = theta(:, M).*thetaM;
end
    
function index = find_index_with_largest_angle (W1, W2)
    Cosine    = pdist2(W2, W1, 'cosine');  
    Temp      = sort(Cosine,2, 'descend');
    [~, Rank] = sortrows(Temp);
    index     = Rank(1);
end
```
