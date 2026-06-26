# MOEA-PSL

**Tags**: <2021> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>

## Description
Multi-objective evolutionary algorithm based on Pareto optimal subspace

## Reference
Y. Tian, C. Lu, X. Zhang, K. C. Tan, and Y. Jin. Solving large-scale multi-objective optimization problems with sparse optimal solutions via unsupervised neural networks. IEEE Transactions on Cybernetics, 2021, 51(6): 3115-3128.

## Source Code

### `DAE.m`
```matlab
classdef DAE < handle
% Feedforward neural network

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(SetAccess = private)
        nVisible  = 0;
        nHidden   = 0;
        Epoch     = 10;
        BatchSize = 1;
        InputZeroMaskedFraction = 0.5;
        Momentum  = 0.5;
        LearnRate = 0.1;
        WA        = [];
        WB        = [];
        lower     = [];
        upper     = [];
    end
    methods
        %% Constructor
        function obj = DAE(nVisible,nHidden,Epoch,BatchSize,InputZeroMaskedFraction,Momentum,LearnRate)
            obj.nVisible  = nVisible;
            obj.nHidden   = nHidden;
            obj.Epoch     = Epoch;
            obj.BatchSize = BatchSize;
            obj.InputZeroMaskedFraction = InputZeroMaskedFraction;
            obj.Momentum  = Momentum;
            obj.LearnRate = LearnRate; 
            obj.WA = (rand(nHidden,nVisible+1)-0.5)*8*sqrt(6/(nHidden+nVisible));   
            obj.WB = (rand(nVisible,nHidden+1)-0.5)*8*sqrt(6/(nVisible+nHidden));  
        end
        %% Train
        function train(obj,X)
            obj.lower = min(X,[],1);
            obj.upper = max(X,[],1);
            X = (X-repmat(obj.lower,size(X,1),1))./repmat(obj.upper-obj.lower,size(X,1),1);
            vW{1} = zeros(size(obj.WA));
            vW{2} = zeros(size(obj.WB));
            if(obj.InputZeroMaskedFraction ~= 0)
                theta = rand(size(X)) > obj.InputZeroMaskedFraction;
            else
                theta = true(size(X));
            end
            X_temp = X.*theta;
            X_temp = [ones(size(X,1),1),X_temp];
            for i = 1 : obj.Epoch
                kk = randperm(size(X,1));
                for batch = 1 : size(X,1)/obj.BatchSize
                    batch_x = X_temp(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);
                    batch_y = X(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);

                    % Feedforward pass
                    poshid1 = 1./(1+exp(-batch_x*obj.WA'));
                    poshid1 = [ones(obj.BatchSize,1),poshid1];
                    poshid2 = 1./(1+exp(-poshid1*obj.WB'));

                    % BP
                    e     = batch_y - poshid2;
                    d{3}  = -e.*(poshid2.*(1-poshid2));
                    d_act = poshid1.*(1-poshid1);
                    d{2}  = d{3}*obj.WB.*d_act;
                    for i = 1 : 2
                        if i+1 == 3
                            dW{i} = (d{i+1}'*poshid1/size(d{3},1));
                        else
                            dW{i} = (d{i+1}(:,2:end)'*batch_x)/size(d{i+1},1);
                        end
                    end
                    for i = 1 : 2
                        dW{i} = obj.LearnRate*dW{i};
                        if obj.Momentum > 0
                            vW{i} = obj.Momentum*vW{i} + dW{i};
                            dW{i} = vW{i};
                        end
                        if i == 1
                            obj.WA = obj.WA - dW{i};
                        else
                            obj.WB = obj.WB - dW{i};
                        end
                    end
                end
            end
        end
        %% Reduce
        function H = reduce(obj,X)
            X = (X-repmat(obj.lower,size(X,1),1))./repmat(obj.upper-obj.lower,size(X,1),1);
            H = 1./(1+exp(-X*obj.WA(:,2:end)'-repmat(obj.WA(:,1)',size(X,1),1)));
        end
        %% Recover
        function X = recover(obj,H)
            X = 1./(1+exp(-H*obj.WB(:,2:end)'-repmat(obj.WB(:,1)',size(H,1),1)));
            X = X.*repmat(obj.upper-obj.lower,size(X,1),1) + repmat(obj.lower,size(X,1),1);
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis,sRatio] = EnvironmentalSelection(Population,Dec,Mask,N,len,num)
% The environmental selection of MOEA/PSL

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions
    success = false(1,length(Population));
    [~,uni] = unique(Population.objs,'rows');
    if length(uni) == 1
        [~,uni] = unique(Population.decs,'rows');
    end
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;    
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Calculate the ratio of successful offsprings
    success(uni(Next)) = true;
    s1     = sum(success(len+1:len+num));
    s2     = sum(success(len+num+1:end));
    sRatio = (s1+1e-6)./(s1+s2+1e-6);
    sRatio = min(max(sRatio,0.1),0.9);
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `MOEAPSL.m`
```matlab
classdef MOEAPSL < ALGORITHM
% <2021> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
% Multi-objective evolutionary algorithm based on Pareto optimal subspace
% learning

%------------------------------- Reference --------------------------------
% Y. Tian, C. Lu, X. Zhang, K. C. Tan, and Y. Jin. Solving large-scale
% multi-objective optimization problems with sparse optimal solutions via
% unsupervised neural networks. IEEE Transactions on Cybernetics, 2021,
% 51(6): 3115-3128.
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
            %% Population initialization
            P   = UniformPoint(Problem.N,Problem.D,'Latin');
            Dec = P.*repmat(Problem.upper-Problem.lower,Problem.N,1) + repmat(Problem.lower,Problem.N,1);
            Dec(:,Problem.encoding==4) = 1;
            Mask       = UniformPoint(Problem.N,Problem.D,'Latin') > 0.5;
            Population = Problem.Evaluation(Dec.*Mask);
            [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,Problem.N,0,0);

            %% Optimization
            rho = 0.5;
            while Algorithm.NotTerminated(Population)
                Site = rho > rand(1,ceil(Problem.N/2));
                if any(Site)
                    [rbm,dae,allZero,allOne] = ModelTraining(Mask,Dec,any(Problem.encoding~=4));
                else
                    [rbm,dae,allZero,allOne] = deal([]);
                end
                MatingPool       = TournamentSelection(2,ceil(Problem.N/2)*2,FrontNo,-CrowdDis);
                [OffDec,OffMask] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),rbm,dae,Site,allZero,allOne);
                Offspring        = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FrontNo,CrowdDis,sRatio] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N,length(Population),2*sum(Site));
                rho = (rho+sRatio)/2;
            end
        end
    end
end
```

### `ModelTraining.m`
```matlab
function [rbm,dae,allZero,allOne] = ModelTraining(Mask,Dec,REAL)
% Training RBM and DAE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Determine the size of hidden layers
    allZero = all(~Mask,1);
    allOne  = all(Mask,1);
    other   = ~allZero & ~allOne;
    K       = sum(mean(abs(Mask(:,other).*Dec(:,other))>1e-6,1)>rand(1,sum(other)));
    K       = min(max(K,1),size(Mask,1));
    
    %% Train RBM and DAE
    rbm = RBM(sum(other),K,10,1,0,0.5,0.1);
    rbm.train(Mask(:,other));
    if REAL
        dae = DAE(size(Dec,2),K,10,size(Dec,1),0.5,0.5,0.1);
        dae.train(Dec);
    else
        dae = [];
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask] = Operator(Problem,ParentDec,ParentMask,rbm,dae,Site,allZero,allOne)
% The operator of MOEA/PSL
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    Parent1Mask = ParentMask(1:end/2,:);
    Parent2Mask = ParentMask(end/2+1:end,:);
    Parent1Dec  = ParentDec(1:end/2,:);
    Parent2Dec  = ParentDec(end/2+1:end,:);
    
    %% Binary variation
    if any(Site)
        other   = ~allZero & ~allOne;
        OffTemp = BinaryCrossover(rbm.reduce(Parent1Mask(Site,other)),rbm.reduce(Parent2Mask(Site,other)));
        OffTemp = rbm.recover(OffTemp);
        OffMask = false(size(OffTemp,1),size(Parent1Mask,2));
        OffMask(:,other)  = OffTemp;
        OffMask(:,allOne) = true;
    else
        OffMask = [];
    end
    OffMask = [OffMask;BinaryCrossover(Parent1Mask(~Site,:),Parent2Mask(~Site,:))];
    OffMask = BinaryMutation(OffMask);
    
    %% Real variation
    if any(Problem.encoding~=4)
        if any(Site)
            OffDec = RealCrossover(dae.reduce(Parent1Dec(Site,:)),dae.reduce(Parent2Dec(Site,:)));
            OffDec = dae.recover(OffDec);
        else
            OffDec = [];
        end
        OffDec = [OffDec;RealCrossover(Parent1Dec(~Site,:),Parent2Dec(~Site,:))];
        OffDec = RealMutation(OffDec,Problem.lower,Problem.upper);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(size(OffMask));
    end
end

function Offspring = BinaryCrossover(Parent1,Parent2)
% Uniform crossover

    k = rand(size(Parent1)) < 0.5;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
end

function Offspring = BinaryMutation(Offspring)
% Bitwise mutation

    Site = rand(size(Offspring)) < 1/size(Offspring,2);
    Offspring(Site) = ~Offspring(Site);
end

function Offspring = RealCrossover(Parent1,Parent2)
% Simulated binary crossover

    disC  = 20;
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
end

function Offspring = RealMutation(Offspring,Lower,Upper)
% Polynomial mutation

    disM  = 20;
    [N,D] = size(Offspring);
    Lower = repmat(Lower,N,1);
    Upper = repmat(Upper,N,1);
    Site  = rand(N,D) < 1/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring       = min(max(Offspring,Lower),Upper);
end
```

### `RBM.m`
```matlab
classdef RBM < handle
% Restricted Boltzmann Machine

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(SetAccess = private)
        nVisible  = 0;
        nHidden   = 0;
        Epoch     = 10;
        BatchSize = 1;
        Penalty   = 0.01;
        Momentum  = 0.5;
        LearnRate = 0.1;
        Weight    = [];
        vBias     = [];
        hBias     = [];
    end
    methods
        %% Constructor
        function obj = RBM(nVisible,nHidden,Epoch,BatchSize,Penalty,Momentum,LearnRate)
            obj.nVisible  = nVisible;
            obj.nHidden   = nHidden;
            obj.Epoch     = Epoch;
            obj.BatchSize = BatchSize;
            obj.Penalty   = Penalty;
            obj.Momentum  = Momentum;
            obj.LearnRate = LearnRate;
            obj.Weight    = 0.1 * randn(obj.nVisible,obj.nHidden);
            obj.vBias     = zeros(1,obj.nVisible);
            obj.hBias     = zeros(1,obj.nHidden);
        end
        %% Train
        function train(obj,X)
            vishidinc  = zeros(size(obj.Weight));
	        hidbiasinc = zeros(size(obj.hBias));
	        visbiasinc = zeros(size(obj.vBias));
            for epoch = 1 : obj.Epoch
                if obj.Epoch > 5
                    obj.Momentum = 0.9;
                end
                kk = randperm(size(X,1));
                for batch = 1 : size(X,1)/obj.BatchSize
                    batchdata = X(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);

                    % Positive phase
                    poshidprobs  = 1./(1+exp(-batchdata*obj.Weight-repmat(obj.hBias,obj.BatchSize,1))); 
                    poshidstates = poshidprobs > rand(obj.BatchSize,obj.nHidden);

                    % Negative phase
                    negdataprobs = 1./(1+exp(-poshidstates*obj.Weight'-repmat(obj.vBias,obj.BatchSize,1)));
                    negdata      = negdataprobs > rand(obj.BatchSize,obj.nVisible);
                    neghidprobs  = 1./(1+exp(-negdata*obj.Weight-repmat(obj.hBias,obj.BatchSize,1))); 

                    % Update weight
                    posprods   = batchdata' * poshidprobs;
                    negprods   = negdataprobs' * neghidprobs;
                    poshidact  = sum(poshidprobs);
		            posvisact  = sum(batchdata);
                    neghidact  = sum(neghidprobs);
		            negvisact  = sum(negdata); 
                    vishidinc  = obj.Momentum*vishidinc + obj.LearnRate*(((posprods-negprods)/obj.BatchSize)-obj.Penalty*obj.Weight);
                    visbiasinc = obj.Momentum*visbiasinc + (obj.LearnRate/obj.BatchSize)*(posvisact-negvisact);
                    hidbiasinc = obj.Momentum*hidbiasinc + (obj.LearnRate/obj.BatchSize)*(poshidact-neghidact);
                    obj.Weight = obj.Weight + vishidinc;
                    obj.vBias  = obj.vBias + visbiasinc;
                    obj.hBias  = obj.hBias + hidbiasinc;
                end                
            end
        end
        %% Reduce
        function H = reduce(obj,X)
            H = 1./(1+exp(-X*obj.Weight-repmat(obj.hBias,size(X,1),1))) > rand(size(X,1),size(obj.Weight,2));
        end
        %% Recover
        function X = recover(obj,H)
            X = 1./(1+exp(-H*obj.Weight'-repmat(obj.vBias,size(H,1),1))) > rand(size(H,1),size(obj.Weight,1));
        end
    end
end
```
