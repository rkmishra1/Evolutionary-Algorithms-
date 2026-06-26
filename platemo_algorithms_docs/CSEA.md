# CSEA

**Tags**: <2019> <multi/many> <real/integer> <expensive>

## Description
Classification based surrogate-assisted evolutionary algorithm

## Reference
L. Pan, C. He, Y. Tian, H. Wang, X. Zhang, and Y. Jin. A classification based surrogate-assisted evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(1): 74-88.

## Source Code

### `CSEA.m`
```matlab
classdef CSEA < ALGORITHM
% <2019> <multi/many> <real/integer> <expensive>
% Classification based surrogate-assisted evolutionary algorithm
% k    ---    6 --- Number of reference solutions
% gmax --- 3000 --- Number of solutions evaluated by surrogate model

%------------------------------- Reference --------------------------------
% L. Pan, C. He, Y. Tian, H. Wang, X. Zhang, and Y. Jin. A classification
% based surrogate-assisted evolutionary algorithm for expensive
% many-objective optimization. IEEE Transactions on Evolutionary
% Computation, 2019, 23(1): 74-88.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [k,gmax] = Algorithm.ParameterSet(6,3000);

            %% Initalize the population by Latin hypercube sampling
            N          = min(11*Problem.D-1,109);
            PopDec     = UniformPoint(N,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*PopDec+repmat(Problem.lower,N,1));
            Arc        = Population;
            
            %% Initialize the network
            hiddenLayerSize = ceil(Problem.D*2);
            layers = [featureInputLayer(Problem.D,'Normalization', 'zscore')
                    fullyConnectedLayer(hiddenLayerSize)
                    batchNormalizationLayer
                    reluLayer
                    fullyConnectedLayer(1)
                    sigmoidLayer
                    regressionLayer];

            maxEpochs = 100;
            miniBatchSize = 32;
            options = trainingOptions('adam', ...
                        'ExecutionEnvironment','auto', ...
                        'MaxEpochs',maxEpochs, ...
                        'MiniBatchSize',miniBatchSize, ...
                        'Shuffle','every-epoch', ...
                        'Plots','none', ...
                        'Verbose',false);

            %% Optimization
            while Algorithm.NotTerminated(Arc)
                % Select reference solutions and preprocess the data
                Ref    = RefSelect(Population,k);
                Input  = Population.decs;  
                Output = GetOutput(Population.objs,Ref.objs); 
                rr     = sum(Output)/length(Output);
                tr     = min(rr,1-rr)*0.5;
                [TrainIn,TrainOut,TestIn,TestOut] = DataProcess(Input,Output);
                net = trainNetwork(TrainIn,TrainOut-0,layers,options);

                % Error rates calculation
                TestPre = predict(net,TestIn);
                IndexGood = TestOut==1;
                p0 = sum(abs((TestOut(IndexGood)-TestPre(IndexGood))))/sum(IndexGood);
                p1 = sum(abs((TestOut(~IndexGood)-TestPre(~IndexGood))))/sum(~IndexGood);

                % Surrogate-assisted selection and update the population
                Next = SurrogateAssistedSelection(Problem,net,p0,p1,Ref,Population.decs,gmax,tr);
                if ~isempty(Next)
                    Arc = [Arc,Problem.Evaluation(Next)];
                end
                Population = RefSelect(Arc,Problem.N);
            end
        end
    end
end
```

### `DataProcess.m`
```matlab
function [TrainIn,TrainOut,TestIn,TestOut] = DataProcess(Input,Output)
% Divide the data into the train data and test data in proportion

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

        index1 = find(Output>0.5);
        index0 = find(Output<=0.5);
        K1     = false(1,length(index1));
        K0     = false(1,length(index0));
        K1(randperm(length(index1),ceil(3/4*length(index1)))) = true;
        K0(randperm(length(index0),ceil(3/4*length(index0)))) = true;
        K        = [index1(K1);index0(K0)];
        TrainIn  = Input(K,:);
        TrainOut = Output(K);
        TestIn   = Input(setdiff(1:size(Input,1),K),:);
        TestOut  = Output(setdiff(1:size(Input,1),K));
end
```

### `GetOutput.m`
```matlab
function Output = GetOutput(PopObj,RefPoint)
% Character with two types of solutions, 0 or 1

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    N = size(PopObj,1);
    Output = true(N,1);
    for i = 1 : size(RefPoint,1)
        Output = Output & any(PopObj<=repmat(RefPoint(i,:),N,1),2);
    end
end
```

### `RadarGrid.m`
```matlab
function [Site,RLoc] = RadarGrid(P,div)
% Calcualte the index of radar grid of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

	[N,M] = size(P);
     
    %% Calculate the radar coordinate of each solution
    theta     = 0 : 2*pi/M : 2*pi/M*(M-1);
    RLoc(:,1) = sum(P.*repmat(cos(theta),N,1),2)./sum(P,2);
    RLoc(:,2) = sum(P.*repmat(sin(theta),N,1),2)./sum(P,2);
    RLoc      = (RLoc+1)/2;
    YL        = min(RLoc,[],1);                             % Lower bounary of the transferred points
    YU        = max(RLoc,[],1);                             % Upper bounary of the transferred points  
    NRLoc     = (RLoc-repmat(YL,N,1))./repmat(YU-YL,N,1);	% Normalized points
    %% Identify the index of grid of each solution
    GLoc            = floor(NRLoc.*div);
    GLoc(GLoc>=div) = div - 1;
    UniqueGLoc      = sortrows(unique(GLoc,'rows'));
    [~,Site]        = ismember(GLoc,UniqueGLoc,'rows');
end
```

### `RefSelect.m`
```matlab
function Ref = RefSelect(Population,k)
% Reference solutions selection by RSEA strategy

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    k      = min(k,length(Population));
    PopObj = Population.objs;
	[FrontNO,MaxFNO] = NDSort(PopObj,k);
    Next = find(FrontNO<=MaxFNO);
    Pmin = min(PopObj,[],1) + 1e-6;
    Pmax = max(PopObj,[],1);
    if Pmax > Pmin
        PopObj = (PopObj-repmat(Pmin,size(PopObj,1),1))./repmat(Pmax-Pmin,size(PopObj,1),1);
    end
    
    %% Environmental selection
    Choose = LastSelection(PopObj(Next,:),ismember(Next,find(FrontNO<MaxFNO)),ceil(sqrt(k)),k);
    Ref    = Population(Next(Choose));
end
    
function Choose = LastSelection(PopObj,Choose,div,k)
% Select part of the solutions based on the radar grid
    
    %% Identify the extreme solutions
	[~,Extreme] = min(sqrt(sum(PopObj.^2,2)).*sqrt(1-(1-pdist2(PopObj,ones(1,size(PopObj,2)),'cosine')).^2),[],1); %Calculate the extreme points based on PBI
    Choose      = Choose | ismember(1:size(PopObj,1),Extreme);

    %% Calculate the convergence of each solution
	Con = sum(PopObj.^1,2).^1;
    Con = Con./max(Con);
    
    %% Calculate the radar grid of each solution
    [Site,RLoc] = RadarGrid(PopObj,div);
    RDis        = pdist2(RLoc,RLoc);
    RDis(logical(eye(length(RDis)))) = inf;
    CrowdG      = zeros(1,max(Site));
    temp        = tabulate(Site(Choose));
    CrowdG(temp(:,1)) = temp(:,2);

    %% Select k solutions
    while sum(Choose) < k
        % Delete outline solutions
        remainS  = find(~Choose);
        remainG  = unique(Site(remainS));
        bestG    = CrowdG(remainG) == min(CrowdG(remainG));
        current  = remainS(ismember(Site(remainS),remainG(bestG)));
        fitness  = 0.1.*size(PopObj,2).*Con(current) - min(RDis(current,Choose),[],2); % - 0.1.* min(Dis(current,Choose),[],2);
        [~,best] = min(fitness);
        Choose(current(best))       = true;
        CrowdG(Site(current(best))) = CrowdG(Site(current(best))) + 1;
    end
end
```

### `SurrogateAssistedSelection.m`
```matlab
function Next = SurrogateAssistedSelection(Problem,net,p0,p1,Ref,Input,wmax,tr)
% Surrogate-assisted selection for selecting promising solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    Next  = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
    Label = predict(net,Next);
    a     = tr;
    b     = 1 - tr;
    i     = 0;
    if p0<0.4 || (p1<a&&p0<b)
        while i < wmax
            [~,index] = sort(Label,'descend');
            Input     = Next(index(1:length(Ref)),:);
            Next      = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
            Label = predict(net,Next);
            i = i+size(Next,1);
        end
        Next = Next(Label>0.9,:);
    elseif p0>b && p1<a % Randomly select one to avoid loop
        randindex = randperm(length(Next));
        Next = Next(randindex(1),:);
    elseif p1 > b
        while i<wmax
            [~,index] = sort(Label);
            Input     = Next(index(1:length(Ref)),:);
            Next      = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
            Label = predict(net,Next);
            i = i+size(Next,1);
        end
        Next = Next(Label<0.1,:);
    else
        Next = Next(randi(end),:);
    end
end
```
