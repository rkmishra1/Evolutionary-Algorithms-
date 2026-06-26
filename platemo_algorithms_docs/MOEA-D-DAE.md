# MOEA-D-DAE

**Tags**: <2020> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
MOEA/D with detect-and-escape strategy

## Reference
Q. Zhu, Q. Zhang, and Q. Lin. A constrained multi-objective evolutionary algorithm with detect-and-escape strategy. IEEE Transactions on Evolutionary Computation, 2020, 24(5): 938-947.

## Source Code

### `ArchiveUpdate.m`
```matlab
function Population = ArchiveUpdate(Population,N)
% Update archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Select feasible solutions
    fIndex     = all(Population.cons <= 0,2);
    Population = Population(fIndex);
    if isempty(Population)
        return
    else
        if size(Population.objs,2)==2
            %% Non-dominated sorting
            [FrontNo,~] = NDSort(Population.objs,1);
            Next = (FrontNo == 1);    
            Population = Population(Next);    
            if sum(Next) > N
                %% Calculate the crowding distance of each solution
                CrowdDis   = CrowdingDistance(Population.objs);
                [~,Rank]   = sort(CrowdDis,'descend');
                Population = Population(Rank(1:N));
            end
        else    
            Population = Population(NDSort(Population.objs,1)==1);
            Population = Population(randperm(length(Population)));
            PCObj = Population.objs;
            nND   = length(Population);
            %% Population maintenance
            if length(Population) > N
                % Normalization
                fmax  = max(PCObj,[],1);
                fmin  = min(PCObj,[],1);
                PCObj = (PCObj-repmat(fmin,nND,1))./repmat(fmax-fmin,nND,1);
                % Determine the radius of the niche
                d  = pdist2(PCObj,PCObj);
                d(logical(eye(length(d)))) = inf;
                sd = sort(d,2);
                r  = mean(sd(:,min(3,size(sd,2))));
                R  = min(d./r,1);
                % Delete solution one by one
                while length(Population) > N
                    [~,worst]  = max(1-prod(R,2));
                    Population(worst)  = [];
                    R(worst,:) = [];
                    R(:,worst) = [];
                end
            end
        end
    end
end
```

### `MOEADDAE.m`
```matlab
classdef MOEADDAE < ALGORITHM
% <2020> <multi> <real/integer/label/binary/permutation> <constrained>
% MOEA/D with detect-and-escape strategy

%------------------------------- Reference --------------------------------
% Q. Zhu, Q. Zhang, and Q. Lin. A constrained multi-objective evolutionary
% algorithm with detect-and-escape strategy. IEEE Transactions on
% Evolutionary Computation, 2020, 24(5): 938-947.
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
            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            % Size of neighborhood
            T = ceil(Problem.N/5);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population  = Problem.Initialization();
            z           = min(Population.objs,[],1);
            Pi          = ones(Problem.N,1);
            CP_old      = sum(sum(max(Population.cons,0),2));
            avg_fit     = sum(max(abs((Population.objs-repmat(z,Problem.N,1)).*W),[],2))/Problem.N;
            epsilon_max = max(sum(max(Population.cons,0),2));
            epsilon     = epsilon_max*size(Population.cons,2);
            sigma_min   = 1-(1/(epsilon+1.0e-10))^(3/ceil(Problem.maxFE/Problem.N));
            current     = 0;
            A     = ArchiveUpdate(Population,Problem.N);
            gen   = 0;    
            CV    = sum(max(Population.cons,0),2);
            fr    = length(find(CV<=0))/Problem.N;
            sigma = max(sigma_min,fr);

            %% Optimization
            while Algorithm.NotTerminated(A)
                Q = [];
                for subgeneration = 1 : 5
                    Bounday = find(sum(W<1e-3,2)==Problem.M-1)';
                    I = [Bounday,TournamentSelection(10,floor(Problem.N/5)-length(Bounday),-Pi)];
                    for j = 1 : length(I)
                        i = I(j);
                        % Choose the parents
                        if rand < 0.9
                            P = B(i,randperm(size(B,2)));
                        else
                            P = randperm(Problem.N);
                        end
                        Offspring = OperatorGAhalf(Problem,Population(P(1:2)));
                        z = min([z;Offspring.objs],[],1);
                        [Population,Pi] = UpdatePop(Population,P,Offspring,epsilon,z,W,sigma,Pi,avg_fit);
                        if current == 1
                            tA = UpdatetA(tA,P,Offspring,z,W,avg_fit);
                        end   
                        if sum(max(Offspring.cons,0),2) == 0
                            Q = [Q Offspring];
                        end
                    end 
                end
                gen   = gen+1;
                CV    = sum(max(Population.cons,0),2);
                fr    = length(find(CV<=0))/Problem.N;
                sigma = max(sigma_min,fr);
                if current==0 || current==2
                    if fr < 0.95
                        epsilon = (1-sigma)*epsilon;
                    else
                        if current == 0
                           current = 1;
                           epsilon = 1e+30;
                           tA      = Population;
                           gen     = 1;
                        elseif current == 2
                           epsilon   = epsilon_max;
                           sigma_min = 1-(1/(epsilon+1.0e-10))^(3/ceil(Problem.maxFE/Problem.N));
                        end 
                    end 
                end
                if mod(gen,10) == 0
                    CP      = sum(sum(max(Population.cons,0)));
                    ROC     = abs(CP-CP_old)/(CP+1e-10);
                    CP_old  = CP;
                    avg_fit = sum(max(abs((Population.objs-repmat(z,Problem.N,1)).*W),[],2))/Problem.N;
                    if current == 0
                        if (ROC>0&&ROC<1e-5) && (CP>0.1*epsilon_max)
                            current = 1;
                            epsilon = 1e+30;
                            gen     = 1;
                            tA      = Population;
                        end 
                    elseif current == 1
                        if ROC>0 && ROC<1e-5
                           current     = 2;
                           Population  = tA;
                           epsilon_max = max(sum(max(Population.cons,0),2));
                           epsilon     = epsilon_max;
                           sigma_min   = 1-(1/(epsilon+1.0e-10))^(3/ceil(Problem.maxFE/Problem.N));
                           z           = min(Population.objs,[],1);
                        end
                    end
                end
                if size(Q,2) > 0
                   A = ArchiveUpdate([A Q],Problem.N);
                end
            end
        end
    end
end
```

### `UpdatePop.m`
```matlab
function [Population,pi] = UpdatePop(Population,P,Offspring,epsilon,z,W,sigma,pi,avg_fit)
% Update population 

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    c  = 0;
    nr = 2; 
    while c~=nr && size(P,2)~=0
        index = randperm(size(P,2));
        i     = P(index(1));
        g_o   = max(repmat(abs(Offspring.objs-z),1,1).*W(i,:),[],2);
        g_pi  = max(repmat(abs(Population(i).objs-z),1,1).*W(i,:),[],2);
        if (sum(max(Population(i).cons,0),2)<=epsilon) && (sum(max(Offspring.cons,0),2)<epsilon)
            fitnesso  = g_o;
            fitnesspi = g_pi;
        else
            fitnesso  = sigma*g_o+(1-sigma)*avg_fit*sum(max(Offspring.cons,0),2);
            fitnesspi = sigma*g_pi+(1-sigma)*avg_fit*sum(max(Population(i).cons,0),2);
        end
        if fitnesso < fitnesspi
            Population(i) = Offspring;
            delta = (fitnesspi-fitnesso)/fitnesspi;
            if delta > 0.001
                pi(i) = 1;
            else
                pi(i) = 0.95+(0.05*delta/0.001)*pi(i); 
            end
            c = c+1;
        end
        P(index(1)) = [];
    end
end
```

### `UpdatetA.m`
```matlab
function tA = UpdatetA(tA,P,Offspring,z,W,avg_fit)
% Update temparory archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    c  = 0;
    nr = 2; 
    while c~=nr && size(P,2)~=0
        index = randperm(size(P,2));
        i     = P(index(1));
        g_o   = max(repmat(abs(Offspring.objs-z),1,1).*W(i,:),[],2);
        g_pi  = max(repmat(abs(tA(i).objs-z),1,1).*W(i,:),[],2);
        e     = 1/size(tA,2);
        if (e*g_o+(1-e)*avg_fit*sum(max(Offspring.cons,0),2))<(e*g_pi+(1-e)*avg_fit*sum(max(tA(i).cons,0),2))
            tA(i) = Offspring;
        end
        c = c + 1;
        P(index(1)) = [];
    end
end
```
