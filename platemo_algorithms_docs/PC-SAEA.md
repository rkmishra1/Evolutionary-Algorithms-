# PC-SAEA

**Tags**: <2023> <multi/many> <real> <expensive>

## Description
Pairwise comparison based surrogate-assisted evolutionary algorithm

## Reference
Y. Tian, J. Hu, C. He, H. Ma, L. Zhang, and X. Zhang. A pairwise comparison based surrogate-assisted evolutionary algorithm for expensive multi-objective optimization. Swarm and Evolutionary Computation, 2023, 80: 101323.

## Source Code

### `CalFitnessPC.m`
```matlab
function [Input,Output,Pa,Pmid] = CalFitnessPC(PopObj,PopDec,rate)
% Calculate the fitness value by using a new way  considering 
% convergence and the diversity 

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N      = size(PopObj,1);
    Zmin   = min(PopObj,[],1);
    Zmax   = max(PopObj);            
    PopObj = (PopObj-repmat(Zmin,N,1))./(repmat(Zmax,N,1)-repmat(Zmin,N,1));
    SDE    = zeros(N,1);
    for i = 1 : N
        SPopuObj = PopObj;
        Temp     = repmat(PopObj(i,:),N,1);
        Shifted  = PopObj < Temp;
        SPopuObj(Shifted) = Temp(Shifted);                                    
        Distance  = pdist2(real(PopObj(i,:)),real(SPopuObj));
        [~,index] = sort(Distance,2);
        Dk = Distance(index(floor(sqrt(N))+1)); % Dk denotes the distance of solution i and its floor(sqrt(N)+1)-th nearest neighbour
        SDE(i)=2./(Dk+2);
    end
    Objs = SDE;
    
   %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N                 
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end

   %% Calculate S(i)
    S = sum(Dominate,2);

    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end

   %% Calculate D(i)
    Distance = pdist2(real(Objs),real(Objs),'cosine');
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);

   %% Calculate the fitnesses
    Rmin = min(R);
    Rmax = max(R);
    R = (R-repmat(Rmin,1,N))./(repmat(Rmax,1,N)-repmat(Rmin,1,N));
    R = R';
    Fitness =  rate*R + (1-rate)*D;

    [~, index] = sort(Fitness);
    Input  = [PopDec(index(1:ceil(N/4)),:);PopDec(index(end-(ceil(N/2)-ceil(N/4))+1:end),:)];
    Input  = [Input,[Fitness(index(1:ceil(N/4)));Fitness(index(end-(ceil(N/2)-ceil(N/4))+1:end))]];
    Output = zeros(ceil(N/2),1);
    Output(1:ceil(N/4))     = 2;
    Output(ceil(N/4)+1:end) = 1;
    Pa   = Input(1:ceil(N/4),:);
    Pmid = Input(ceil(N/4)-floor(N/8):ceil(N/4)+floor(N/8),:);
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


    index1 = find(Output>1);
    index0 = find(Output<=1);
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

### `ESCalFitness.m`
```matlab
function Fitness = ESCalFitness(PopObj)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Fitness] = EnvironmentalSelection(Population,N)
% The environmental selection of SPEA2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    Fitness = ESCalFitness(Population.objs);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
```

### `PCSAEA.m`
```matlab
classdef PCSAEA < ALGORITHM
% <2023> <multi/many> <real> <expensive>
% Pairwise comparison based surrogate-assisted evolutionary algorithm
% delta ---  0.8 --- Threshold of reliability measurement
% gmax  --- 3000 --- Number of solutions evaluated by surrogate model

%------------------------------- Reference --------------------------------
% Y. Tian, J. Hu, C. He, H. Ma, L. Zhang, and X. Zhang. A pairwise
% comparison based surrogate-assisted evolutionary algorithm for expensive
% multi-objective optimization. Swarm and Evolutionary Computation, 2023,
% 80: 101323.
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
            [delta,gmax] = Algorithm.ParameterSet(0.8,3000);

            %% Initalize the population by Latin hypercube sampling
            N          = max(11*Problem.D-1,Problem.N);
            PopDec     = UniformPoint(N,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*PopDec+repmat(Problem.lower,N,1));
            Arc        = Population;
            t          = 1;
        
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                % Select a balance sample set by a new fitness
                [Input,Output,Pa,Pmid] = CalFitnessPC(Population.objs,Population.decs,(Problem.FE/Problem.maxFE));   
                % Data process
                [TrainIn,~,TestIn,TestOut] = DataProcess(Input,Output);
                % Construct and update the FNNglobal classify surrogate model
                net = RBFNNPC(0.1925);             
                net.train(TrainIn,Problem.D);              

                % Error rates calculation
                TestPre = net.lastpredict(TestIn,Problem.D,Pmid,1);
                % New and suitble reliability selection
                validIndex = TestPre~=1.5;
                Error1 = sum(TestOut(validIndex)==TestPre(validIndex))/length(TestOut);   
                Error2 = sum(TestOut(validIndex)~=TestPre(validIndex))/length(TestOut);              

                % Surrogate-assisted selection and update the population
                Next = SurrogateAssistedSelectionPC(Problem,net,Error1,Error2,Population.decs,gmax,Pa,Problem.D,0,delta);
                if ~isempty(Next)
                    Arc = [Arc,Problem.Evaluation(Next)];
                end
                Population = EnvironmentalSelection(Arc,Problem.N);                
                t = t + 1;
            end
        end
    end
end
```

### `RBFNNPC.m`
```matlab
classdef RBFNNPC < handle
% Radial Basis Network

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(SetAccess = private)
        spread   = 0;
        srgtSRGT = cell(1,1);
    end
    methods
        %% Constructor
        function obj = RBFNNPC(spread)
            obj.spread = spread;
        end

        %% Train
        function train(obj,X,distanceD)
            % Input ---(x1x2) decision number is changed to 2*D
            % Output --1 or 0 , meaning is better or worse
            N       = size(X,1);
            input   = zeros(N*N,2*distanceD);
            output  = zeros(N*N,1);
            FrontNo = X(:,end);
            for i = 1 : N
                for j = 1 : N
                    input(N*(i-1)+j,:) = [X(i,1:distanceD),X(j,1:distanceD)];
                    if FrontNo(i)<FrontNo(j)
                        output(N*(i-1)+j) = 2;
                    else
                        output(N*(i-1)+j) = 1;
                    end
                end
            end
            X    = input;
            T    = output;
            E    = eye(N);
            Find = E(:)==1;
            X(Find,:) = [];
            T(Find)   = [];

            Tc = ind2vec(T');
            obj.srgtSRGT = newpnn(X',Tc,obj.spread);
        end

        %% Predict for data
        function Y = lastpredict(obj,X,distanceD,Preference,flag)
            % change the input including the preference point
            N       = size(X,1);
            decs    = X(:,1:distanceD);
            numberP = size(Preference,1);

            Pref = zeros(N,distanceD);
            for i = 1 : N
                Pref(i,:) = Preference(mod(i+numberP,numberP)+1,1:distanceD);% better than a random reference point
            end

            % forward forecasting
            X  = [decs,Pref];
            Yc = sim(obj.srgtSRGT,X');
            Y  = vec2ind(Yc);
            Y  = Y';

            % reverse forecasting
            x  = [Pref,decs];
            yc = sim(obj.srgtSRGT,x');
            y  = vec2ind(yc);
            y  = y';

            result = Y-y;
            if flag == 0
                Y = result;
            else
                Y(result == 0) = 1.5;
            end
        end
    end
end
```

### `SurrogateAssistedSelectionPC.m`
```matlab
function Next = SurrogateAssistedSelectionPC(Problem,net,error1,error2,Input,wmax,Pa,D,flag,delta)
% Surrogate-assisted selection for selecting promising solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Pa    = Pa(:,1:D);
    Next  = OperatorGA(Problem,[Input;Pa(:,1:D)],{1,15,1,5});
    Label = net.lastpredict(Next,D,Pa,flag);
    lnum  = size(Pa,1);
    i     = 1;
    wmax  = floor(wmax/lnum);
    GoodNext  = zeros(wmax,D);
    GoodLabel = zeros(wmax,1);
    if error1 < 1-delta
        while i <= wmax
            [~,index]      = sort(Label,'descend');
            GoodNext(i,:)  = Next(index(1),:);
            GoodLabel(i,:) = Label(index(1));
            Input  = Next(index(1:lnum),:);
            Parent = [Input;Pa];
            Next   = OperatorGA(Problem,Parent(randperm(end),:),{1,15,1,5});
            Label  = net.lastpredict(Next,D,Pa,flag);
            i      = i + 1;
        end
        if (sum(GoodLabel >= 0.95) == 0) || (sum(GoodLabel >= 0.95) > floor(lnum/2))
            [~,index] = sort(GoodLabel(:,end),'descend');
            Next      = GoodNext(index(1:floor(lnum/2)),1:D);
        else
            Next = GoodNext(GoodLabel >= 0.95,:);
        end

    elseif error2 < 1-delta
        while i <= wmax
            [~,index]      = sort(Label);
            GoodNext(i,:)  = Next(index(1),:);
            GoodLabel(i,:) = Label(index(1));
            Input  = Next(index(1:lnum),:);
            Parent = [Input;Pa];
            Next   = OperatorGA(Problem,Parent(randperm(end),:),{1,15,1,5});
            Label  = net.lastpredict(Next,D,Pa,flag);
            i      = i + 1;
        end
        if (sum(GoodLabel <= -0.95) == 0) || (sum(GoodLabel <= -0.95) > floor(lnum/2))
            [~,index] = sort(GoodLabel(:,end));
            Next      = GoodNext(index(1:floor(lnum/2)),1:D);
        else
            Next = GoodNext(GoodLabel <= -0.95,:);
        end
    else
        Next = Next(randi(end),:);
    end
end
```
