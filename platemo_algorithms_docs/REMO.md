# REMO

**Tags**: <2022> <multi/many> <real> <expensive>

## Description
Expensive multiobjective optimization by relation learning and prediction

## Reference
H. Hao, A. Zhou, H. Qian, and H. Zhang. Expensive multiobjective optimization by relation learning and prediction. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 1157-1170.

## Source Code

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

    pha     = 3/4;
    index0  = find(Output==0);
    indexp1 = find(Output == 1);
    indexn1 = find(Output == -1);

    K0  = false(1,length(index0));
    Kp1 = false(1,length(indexp1));
    Kn1 = false(1,length(indexn1));

    K0(randperm(length(index0),ceil(pha*length(index0))))    = true;
    Kp1(randperm(length(indexp1),ceil(pha*length(indexp1)))) = true;
    Kn1(randperm(length(indexn1),ceil(pha*length(indexn1)))) = true;

    K        = [index0(K0);indexp1(Kp1);indexn1(Kn1)];
    TrainIn  = Input(K,:);
    TrainOut = Output(K);

    TestIn  = Input(setdiff(1:size(Input,1),K),:);
    TestOut = Output(setdiff(1:size(Input,1),K));

    Train_randindex = randperm(size(TrainOut,1),size(TrainOut,1));
    TrainIn         = TrainIn(Train_randindex,:);
    TrainOut        = TrainOut(Train_randindex);

    Test_randindex = randperm(size(TestOut,1),size(TestOut,1));
    TestIn         = TestIn(Test_randindex,:);
    TestOut        = TestOut(Test_randindex);
end
```

### `Delequalsamples.m`
```matlab
function [XXs,YYs] = Delequalsamples(XXs,YYs)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    zerosindex        = YYs == 0;
    XXs(zerosindex,:) = [];
    YYs(zerosindex)   = [];
end
```

### `GetOutput_PBI.m`
```matlab
function [Output,r] = GetOutput_PBI(varargin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    selfadapt = true;
    if nargin == 3
        selfadapt = false;
        delt      = varargin{3};
    end
    
    Pop = varargin{1};
    Ref = varargin{2};
    
    if selfadapt
        delt_l = -20;
        delt_u = 20;
        r = 0;     
        while r>0.7 || r<0.3
            delt_c = (delt_l + delt_u)/2;
            if abs(delt_l-delt_u)<1e-1
                break;
            end
            [l,r] = split_data(Pop,Ref,delt_c);
            if r > 0.7
               delt_l = delt_c;
            elseif r < 0.3
               delt_u = delt_c;
            end
        end
    else
        [l,~] = split_data(Pop,Ref,delt);
    end
    Output = l;
end

function [Output,rate] = split_data(Pop,Ref,delt)
    N      = size(Pop,1);
    popind = 1 : N;
    Output = true(N,1);
    [~,ref_index] = max(1-pdist2(Pop,Ref,'cosine'),[],2);
    Z = min(Pop,[],1);
    for i = 1 : size(Ref,1)
        sub_pop    = Pop(ref_index==i,:);
        sub_popind = popind(ref_index==i);
        BOUND      = Ref(i,:);
        w = BOUND-Z;
        W = w./sqrt(sum((w).^2,2));
        normW   = sqrt(sum((W).^2,2));
        normP   = sqrt(sum((sub_pop-repmat(Z,size(sub_pop,1),1)).^2,2));
        normR   = sqrt(sum((BOUND-Z).^2,2));
        CosineP = (sum((sub_pop-repmat(Z,size(sub_pop,1),1)).*repmat(W,size(sub_pop,1),1),2)./normW./normP)-1e-6;
        g = normP.*CosineP + delt*normP.*sqrt(1-CosineP.^2);
        k = normR;
        g = g./k;
        Output(sub_popind(g>1)) = false;
    end
	rate = sum(Output == 1)/length(Output);
end
```

### `GetRelationPairs.m`
```matlab
function [XXs,Ls] = GetRelationPairs(Input,Catalog)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    C1_index = Catalog == 1;
    C2_index = Catalog ~= 1;
    C1C1 = combvec(Input(Catalog ==1,:)',Input(Catalog ==1,:)')';
    C1C2 = combvec(Input(Catalog ==1,:)',Input(Catalog ~=1,:)')';
    C2C1 = combvec(Input(Catalog ~=1,:)',Input(Catalog ==1,:)')';
    C2C2 = combvec(Input(Catalog ~=1,:)',Input(Catalog ~=1,:)')';

    t_ind     = combvec(1:sum(C1_index),1:sum(C1_index));
    t_equ_ind = t_ind(1,:) == t_ind(2,:);
    C1C1(t_equ_ind,:) = [];
    
    t_ind     = combvec(1:sum(C2_index),1:sum(C2_index));
    t_equ_ind = t_ind(1,:) == t_ind(2,:);
    C2C2(t_equ_ind,:) = [];

    t_num = ceil(size(C1C2,1)/2);
    if size(C1C1,1) > t_num && size(C2C2,1) > t_num
        C1C1 = C1C1(randperm(size(C1C1,1),t_num),:);
        C2C2 = C2C2(randperm(size(C2C2,1),t_num),:);
    elseif size(C1C1,1) < t_num
        C2C2 = C2C2(randperm(size(C2C2,1),t_num*2-size(C1C1,1)),:);
    elseif size(C2C2,1) < t_num
        C1C1 = C1C1(randperm(size(C1C1,1),t_num*2-size(C2C2,1)),:);
    end

    XXs = [C1C1;C2C2;C1C2;C2C1];
    Ls  = [zeros(size(C1C1,1),1);zeros(size(C2C2,1),1);ones(size(C1C2,1),1);-1.*ones(size(C2C1,1),1)];
end
```

### `REMO.m`
```matlab
classdef REMO < ALGORITHM
% <2022> <multi/many> <real> <expensive>
% Expensive multiobjective optimization by relation learning and prediction
% k    ---    6 --- Number of reference solutions
% gmax --- 3000 --- Number of solutions evaluated by surrogate model

%------------------------------- Reference --------------------------------
% H. Hao, A. Zhou, H. Qian, and H. Zhang. Expensive multiobjective
% optimization by relation learning and prediction. IEEE Transactions on
% Evolutionary Computation, 2022, 26(5): 1157-1170.
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
            %% Parameterr setting
            [k,gmax] = Algorithm.ParameterSet(6,3000);

            %% Initalize the population by Latin hypercube sampling
            if Problem.D <= 10
                N = 11*Problem.D-1;
            else
                N = 100;
            end
            PopDec     = UniformPoint(N,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*PopDec+repmat(Problem.lower,N,1));
            Archive    = Population;

            %% Optimization
            while Algorithm.NotTerminated(Archive)
                % Select reference solutions and preprocess the data
                Ref       = RefSelect(Population,k);
                Input     = Population.decs; 
                Catalog   = GetOutput_PBI(Population.objs,Ref.objs); 
                [XXs,YYs] = GetRelationPairs(Input,Catalog);
                [TrainIn,TrainOut,TestIn,TestOut] = DataProcess(XXs,YYs);
                xDim = size(TrainIn,2);
                
                % Train relation model
                [TrainIn_nor,TrainIn_struct] = mapminmax(TrainIn');
                TrainIn_nor     = TrainIn_nor';
                TrainOut_onehot = onehotconv(TrainOut,1);
                net = patternnet([ceil(xDim*1.5),xDim*1,ceil(xDim/2)]);
                net.trainParam.showWindow =0;
                net        = train(net,TrainIn_nor',TrainOut_onehot');
                TestIn_nor = mapminmax('apply',TestIn',TrainIn_struct)';
                TestPre    = onehotconv(net(TestIn_nor')',2);             
                p_err      = sum(TestPre ~= TestOut)/size(TestPre,1);
                Smodel.X   = Input;
                Smodel.Y   = Catalog;
                Smodel.mp_struct = TrainIn_struct;
                Smodel.net       = net;
                Smodel.p_err     = p_err;
                Next = RSurrogateAssistedSelection(Problem,Ref,Population.decs,gmax,Smodel);
                if ~isempty(Next)
                    Archive = [Archive,Problem.Evaluation(Next)];
                end
                Population = RefSelect(Archive,Problem.N);
            end
        end
	end
end
```

### `RSurrogateAssistedSelection.m`
```matlab
function Next = RSurrogateAssistedSelection(Problem,Ref,Input,wmax,Smodel)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Next = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
    i    = 0;
    while i < wmax
        [soerted_index,~] = model_select(Smodel,Next);
        Input = Next(soerted_index(1:length(Ref)),:);
        Next  = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
        i     = i + size(Next,1);
    end
    [~,scores] = model_select(Smodel,Next);
    if sum(scores>3.9) < 4
        [~,ind] = sort(scores,'descend');
        Next    = Next(ind(1:4),:); 
    else
        Next = Next(scores>3.9,:);
    end
end

function [ind,scores] = model_select(Smodel,Next)
    model_x = Smodel.X;    
    C1_data = model_x(Smodel.Y ==1,:);
    C2_data = model_x(Smodel.Y ~=1,:);

    C1_num   = size(C1_data,1);
    C2_num   = size(C2_data,1);
    Next_num = size(Next,1);

    scores = zeros(Next_num,2);
    
    all_testdata = zeros(2*(C1_num+C2_num)*Next_num,2*size(C1_data,2));
    for i = 1 : size(Next,1)
        original = (i-1)*2*(C1_num+C2_num);
        Xi       = repmat(Next(i,:),size(C1_data,1),1);
        all_testdata(original+1:original+C1_num,:)          = [C1_data,Xi];  %C1_Xi
        all_testdata(original+1+C1_num:original+C1_num*2,:) = [Xi,C1_data]; %Xi_C1
        
        Xi = repmat(Next(i,:),size(C2_data,1),1);
        all_testdata(original+1+C1_num*2:original+C1_num*2+C2_num,:)          = [C2_data,Xi]; %C2_Xi
        all_testdata(original+1+C2_num+C1_num*2:original+C2_num*2+C1_num*2,:) = [Xi,C2_data];%Xi_C2
    end
    
    TestIn_nor = mapminmax('apply',all_testdata',Smodel.mp_struct)';
    pre_out    = Smodel.net(TestIn_nor')';  
    
    for i = 1 : size(Next,1)
        C_SCORE    = zeros(1,2);
        original   = (i-1)*2*(C1_num+C2_num);
        pre_C1Xi   = sum(pre_out(original+1:original+C1_num,:),1)./C1_num;
        C_SCORE(1) = C_SCORE(1) + pre_C1Xi(2)+pre_C1Xi(3);   
        C_SCORE(2) = C_SCORE(2) + pre_C1Xi(1);               
        
        pre_XiC1   = sum(pre_out(original+1+C1_num:original+C1_num*2,:),1)./C1_num;
        C_SCORE(1) = C_SCORE(1) + pre_XiC1(2) + pre_XiC1(1);  
        C_SCORE(2) = C_SCORE(2) + pre_XiC1(3);                 
        
        pre_C2Xi   = sum(pre_out(original+1+C1_num*2:original+C1_num*2+C2_num,:),1)./C2_num;
        C_SCORE(1) = C_SCORE(1) + pre_C2Xi(3);
        C_SCORE(2) = C_SCORE(2) + pre_C2Xi(2) + pre_C2Xi(1);
        
        pre_XiC2   = sum(pre_out(original+1+C2_num+C1_num*2:original+C2_num*2+C1_num*2,:),1)./C2_num;
        C_SCORE(1) = C_SCORE(1) + pre_XiC2(1);
        C_SCORE(2) = C_SCORE(2) + pre_XiC2(2) + pre_XiC2(3);
        
        scores(i) = C_SCORE(1)-C_SCORE(2);
    end      
    [~,ind] = sort(scores,'descend');  
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

function [Site,RLoc] = RadarGrid(P,div)

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

### `onehotconv.m`
```matlab
function varargout = onehotconv(varargin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if varargin{2}== 1
        %% conv onehot
        l        = varargin{1};      
        l_onehot = zeros(size(l,1),3);
     
        l_onehot(l == 1 ,1) = 1;
        l_onehot(l == 0,2)  = 1;
        l_onehot(l == -1,3) = 1;
        
        varargout = {l_onehot};
        
    elseif varargin{2} == 2
        %% deconv onehot
        onehot_l = varargin{1};
        res_l    = zeros(size(onehot_l,1),1);

        [~,maxind] = max(onehot_l,[],2);

        res_l(maxind==1) = 1;
        res_l(maxind==3) = -1;

        varargout = {res_l};
    end
end
```
