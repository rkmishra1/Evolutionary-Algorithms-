# LMOEA-DS

**Tags**: <2021> <multi> <real/integer> <large/none>

## Description
Large-scale evolutionary multi-objective optimization assisted by directed sampling

## Reference
S. Qin, C. Sun, Y. Jin, Y. Tan, and J. Fieldsend. Large-scale evolutionary multi-objective optimization assisted by directed sampling. IEEE Transactions on Evolutionary Computation, 2021, 25(4): 724-738.

## Source Code

### `DecompositionSelection.m`
```matlab
function Population = DecompositionSelection(Global,Population,associate,Cosinemax)
% The decomposition-based method environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Shufen Qin
% E-mail: shufen.qin@stu.tyust.edu.cn

    np = length(Population);
    %% Normalization
    Obj = Population.objs;
    Obj = (Obj-repmat(min(Obj),np,1))./(repmat(max(Obj),np,1)-repmat(min(Obj),np,1));

    %% Select one solution for each reference vector
    list = unique(associate)';
    Next = zeros(length(list),1);
    t = 1;
    for i = list
        current = find(associate == i);
        dist = pdist2(Obj(current,:),zeros(1,Global.M),'Euclidean');
        Fan = Cosinemax(current)./dist;
        [~,best] = max(Fan);
        Next(t)  = current(best);
        t = t +1;
    end
    % Population for next generation
    Population = Population(Next);
end
```

### `DirectedSampling.m`
```matlab
function GuidingSolution = DirectedSampling(Problem,Population,Ns,Nw,RefV)
% Acquiring Guiding Solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Shufen Qin
% E-mail: shufen.qin@stu.tyust.edu.cn

    %% Classter the reference vectors
    BoundRefV               = eye(Problem.M);
    BoundRefV(BoundRefV==0) = 10e-7;
    [~,CenterRefV,~,~]      = kmeans(RefV,Nw);
    DirectRefV              = [BoundRefV;CenterRefV];
    Nw                      = size(DirectRefV,1);
    
    %% Identify guiding directions
    Best       = GenerateRepresetativeSolution(Population.objs,DirectRefV);
    PopDec     = Population.decs;
    BestX      = PopDec(Best,:);
    Upper      = Problem.upper; 
    Lower      = Problem.lower;
    Directnorm = [sqrt(sum((BestX - repmat(Lower,Nw,1)).^2,2));sqrt(sum((BestX - repmat(Upper,Nw,1)).^2,2))];
    Direction  = [BestX - repmat(Lower,Nw,1);BestX - repmat(Upper,Nw,1)]./repmat(Directnorm,1,Problem.D);
    
    %% Generate guiding solutions
    Intervalmax     = sqrt(sum((Upper-Lower).^2,2));
    Intervalmin       = 0;
    Nw              = 2*Nw;
    RandSample      = Intervalmin + rand(Ns,Nw)*(Intervalmax-Intervalmin);
    SampleSolution  = GenerateSampleSolution(Problem,RandSample,Direction);
    GuidingSolution = SampleSolution((NDSort(SampleSolution.objs,1)==1));
    
end

function Best = GenerateRepresetativeSolution(Obj,RefV)
% Find out respective solutions

    %% Normalization
    np = size(Obj,1);
    Obj = (Obj-repmat(min(Obj),np,1))./(repmat(max(Obj),np,1)-repmat(min(Obj),np,1));
    Nr   = size(RefV,1);
    Best = zeros(Nr,1);

    %% Assign individuals for each reference vector
    Cosine        = 1-pdist2(Obj,RefV,'cosine');
    [~,associate] = max(Cosine,[],2);

    Indflag = zeros(np,1);
    current = cell(Nr,1);
    for i = 1 : Nr
        current{i,1} = find(associate == i);
    end  
    for i = 1 : Nr
        if length(current{i,1})>1
            normf = sqrt(sum(Obj(current{i,1},:).^2,2));
            normRefV = sqrt(sum(repmat(RefV(i,:),length(current{i,1}),1).^2,2));
            CosinefRefV = sum(Obj(current{i,1},:).*repmat(RefV(i,:),length(current{i,1}),1),2)./normRefV./normf;
            d1 = normf .* CosinefRefV;
            [~,ind] = sort(d1,'ascend');
            Best(i,1) = current{i,1}(ind(1));
            Indflag(current{i,1}(ind(1)),1) = 1;
        elseif length(current{i,1})==1
            Best(i,1) = current{i,1}(1);
            Indflag(current{i,1}(1),1) = 1;
        end
    end
    for i = 1:Nr
        if isempty(current{i,1})
            [~,indCon] = sort(Cosine(:,i),'descend');
            k = 1;
            if length(indCon) > Nr
                while Indflag(indCon(k),1) == 1
                    k=k+1;
                end
                Best(i,1) = indCon(k);
                Indflag(indCon(k),1) = 1;
            else
                Best(i,1) = indCon(1);
            end
        end
    end
end

function SampleSolution = GenerateSampleSolution(Problem,RandSample,Direct)
% Generate some sample solutions along with the guiding directions

   [Ns,Nw] = size(RandSample);
   Nw = Nw/2;
   SampleSolution = [];
   for i = 1:Ns
       PopX = [repmat(Problem.lower,Nw,1) + repmat(RandSample(i,1:Nw)',1,Problem.D).* Direct(1:Nw,:);...
           repmat(Problem.upper,Nw,1) + repmat(RandSample(i,Nw+1:end)',1,Problem.D).* Direct(Nw+1:end,:)];
       PopX = max(min(repmat(Problem.upper,size(PopX,1),1),PopX),repmat(Problem.lower,size(PopX,1),1));
       SampleSolutiontemp = Problem.Evaluation(PopX);
       SampleSolution = [SampleSolution,SampleSolutiontemp];
   end
end
```

### `DominationSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = DominationSelection(Global,Population)
% The dominant relationship and crowding based environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Shufen Qin
% E-mail: shufen.qin@stu.tyust.edu.cn

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Global.N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:Global.N-sum(Next)))) = true;
    
    %% Population for next generation
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Population = Population(Next);
end
```

### `DoubleReproduction.m`
```matlab
function Pop = DoubleReproduction(Problem,Pop,GuidingSolution,RefV)
% Generate a promising population by the double reproduction and
% complementary environment selection strategy.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Shufen Qin
% E-mail: shufen.qin@stu.tyust.edu.cn

    %% The First Phase
    OffPopX = GAonce(Problem,Pop.decs,GuidingSolution.decs);
    OffPopX = unique(OffPopX,'rows');
    OffPop  = Problem.Evaluation(OffPopX);
    
    % Environmental Selection
    PopCom = [Pop,GuidingSolution,OffPop];
   
    [associate,Cosinemax] = Assign(PopCom,RefV);
    Ns = length(unique(associate)');
    if Ns < (2/3) * Problem.N
        Pop = DominationSelection(Problem,PopCom);
    else
        Pop = DecompositionSelection(Problem,PopCom,associate,Cosinemax);
    end
    
    %% The Second Phase
    %OffPopX = GA(Pop.decs);
    OffPopX = GATwice(Pop.decs,Problem);
    OffPopX = unique(OffPopX,'rows');
    OffPop  = Problem.Evaluation(OffPopX);
    PopCom  = [Pop,OffPop];
  
    [associate2,Cosinemax2] = Assign(PopCom,RefV);
    Ns = length(unique(associate2));
    if Ns<(2/3)*Problem.N
        Pop = DominationSelection(Problem,PopCom);
    else
        Pop = DecompositionSelection(Problem,PopCom,associate2,Cosinemax2);
    end
end

function [associate,Cosinemax] = Assign(PopCom,RefV)
    % Assign individuals 
    Obj = PopCom.objs;
    Obj = (Obj-repmat(min(Obj),length(PopCom),1))./(repmat(max(Obj),length(PopCom),1)-repmat(min(Obj),length(PopCom),1));
    Cosinetemp = pdist2(Obj,RefV,'cosine');
    Cosine = 1-Cosinetemp;
    [Cosinemax,associate] = max(Cosine,[],2);
end

function Offspring = GAonce(Problem,MatingPool,PopS)
% This function is mainly used in the first reproduction.

    N = size(MatingPool,1);
    RandList = randperm(N);
    MatingPool = MatingPool(RandList, :);

    Ns = size(PopS,1);
    Offspring=[];

    for i = 1: size(MatingPool,1)
        k = randi([1,Ns],1);
        Pop = PopS(k,:);
        Parent = [MatingPool(i,:);Pop];
        Offspringtempt = GAhalfrand(Problem,Parent);
        Offspring = [Offspring;Offspringtempt];
    end
end

function Offspring = GAhalfrand(Problem,Parent)

    %% Parameter setting

	[proC,disC,proM,disM] = deal(0.9,20,1,20);

    Parent1 = Parent(1,:);
    Parent2 = Parent(2,:);
    [N,D]   = size(Parent1);

    %% Genetic operators for real encoding
    % Simulated binary crossover
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    if rand <0.5
        Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
    else
        Offspring = (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2;
    end
    % Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
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

function Offspring = GATwice(MatingPool,Global)
% This function includes the SBX crossover operator and the polynomial
% mutatoion operator.

     MaxOffspring = Global.N;
     [N,D] = size(MatingPool);
     RandList = randperm(N);
     MatingPool = MatingPool(RandList, :);
     if  MaxOffspring < 1 || MaxOffspring > N
         MaxOffspring = N;
     end
     if(mod(N,2) == 1)
         MatingPool = [MatingPool; MatingPool(1,:)];
     end  
     
     ProC = 0.9;
     ProM = 1/D;
     
     DisC = 20;
     DisM = 20;
     Offspring = zeros(N,D);
     %crossover
     for i = 1 : 2 : N
         beta = zeros(1,D);
         miu  = rand(1,D);
         beta(miu<=0.5) = (2*miu(miu<=0.5)).^(1/(DisC+1));
         beta(miu>0.5)  = (2-2*miu(miu>0.5)).^(-1/(DisC+1));
         beta = beta.*(-1).^randi([0,1],1,D);
         beta(rand(1,D)>ProC) = 1;
         Offspring(i,:)   = (MatingPool(i,:)+MatingPool(i+1,:))/2+beta.*(MatingPool(i,:)-MatingPool(i+1,:))/2;
         Offspring(i+1,:) = (MatingPool(i,:)+MatingPool(i+1,:))/2-beta.*(MatingPool(i,:)-MatingPool(i+1,:))/2;
     end
     Offspring = Offspring(1:MaxOffspring,:);
     
     %mutation
     if MaxOffspring == 1
         MaxValue = Global.upper;
         MinValue = Global.lower;
     else
         MaxValue = repmat(Global.upper,MaxOffspring,1);
         MinValue = repmat(Global.lower,MaxOffspring,1);
     end
     k    = rand(MaxOffspring,D);
     miu  = rand(MaxOffspring,D);
     Temp = k<=ProM & miu<0.5;
     Offspring(Temp) = Offspring(Temp)+(MaxValue(Temp)-MinValue(Temp)).*((2.*miu(Temp)+(1-2.*miu(Temp)).*(1-(Offspring(Temp)-MinValue(Temp))./(MaxValue(Temp)-MinValue(Temp))).^(DisM+1)).^(1/(DisM+1))-1);
     Temp = k<=ProM & miu>=0.5;
     Offspring(Temp) = Offspring(Temp)+(MaxValue(Temp)-MinValue(Temp)).*(1-(2.*(1-miu(Temp))+2.*(miu(Temp)-0.5).*(1-(MaxValue(Temp)-Offspring(Temp))./(MaxValue(Temp)-MinValue(Temp))).^(DisM+1)).^(1/(DisM+1)));
     
     Offspring(Offspring>MaxValue) = MaxValue(Offspring>MaxValue);
     Offspring(Offspring<MinValue) = MinValue(Offspring<MinValue);
end
```

### `LMOEADS.m`
```matlab
classdef LMOEADS < ALGORITHM
% <2021> <multi> <real/integer> <large/none>
% Large-scale evolutionary multi-objective optimization assisted by directed sampling
% Nw --- 10 --- Cluster number
% Ns --- 30 --- The number of random sampling along each guiding direction

%------------------------------- Reference --------------------------------
% S. Qin, C. Sun, Y. Jin, Y. Tan, and J. Fieldsend. Large-scale
% evolutionary multi-objective optimization assisted by directed sampling.
% IEEE Transactions on Evolutionary Computation, 2021, 25(4): 724-738.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Shufen Qin
% E-mail: shufen.qin@stu.tyust.edu.cn

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            [Nw,Ns] = Algorithm.ParameterSet(10,30);

            %% Initialization
            Population = Problem.Initialization();
            [RefV,~]   = UniformPoint(Problem.N,Problem.M);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                GuidingSolution = DirectedSampling(Problem,Population,Ns,Nw,RefV);
                Population      = DoubleReproduction(Problem,Population,GuidingSolution,RefV);
            end
        end
    end
end
```
