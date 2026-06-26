# CMaDPPs

**Tags**: <2023> <multi/many> <real/integer/label/binary/permutation> <constrained/none>

## Description
Constrained many-objective optimization with determinantal point processes

## Reference
F. Ming, W. Gong, S. Li, L. Wang, and Z. Liao. Handling constrained many-objective optimization problems via determinantal point processes. Information Sciences, 2023, 643: 119260.

## Source Code

### `CMaDPPs.m`
```matlab
classdef CMaDPPs < ALGORITHM
% <2023> <multi/many> <real/integer/label/binary/permutation> <constrained/none>
% Constrained many-objective optimization with determinantal point processes

%------------------------------- Reference --------------------------------
% F. Ming, W. Gong, S. Li, L. Wang, and Z. Liao. Handling constrained
% many-objective optimization problems via determinantal point processes.
% Information Sciences, 2023, 643: 119260.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    methods
        function main(Algorithm,Problem)
            
            %% Parameter setting
            theta = Algorithm.ParameterSet(0);
            
            %% Generate random population
            Population = Problem.Initialization();
            CSA = Population;
            CV = sum(max(0,Population.cons),2);
            [z,znad] = deal(min(Population.objs),max(Population.objs));
            
            % parameters for epsilon
            CVmax = max(CV);
            G = ceil(Problem.maxFE/Problem.N);
            Tc = 0.8 * G;
            cp = 2;
            alpha = 0.95;
            tao = 0.05;
            epsilon = inf;
            
            % parameters for change
            last_gen         = 20;
            change_threshold = 1e-1;
            max_change       = 1;
            ideal_points     = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            nadir_points     = zeros(ceil(Problem.maxFE/Problem.N),Problem.M);
            [Population,Archive] = EnvironmentalSelection(Population,[],[],Problem.N,z,znad,theta,CSA.objs,epsilon);       
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                gen        = ceil(Problem.FE/Problem.N);
                CV = sum(max(0,Population.cons),2);
                CV_max = max(CV);
                CVmax = max([CV_max,CVmax]);
                rf         = sum(CV <= 1e-6) / Problem.N;
                ideal_points(gen,:) = z;
                nadir_points(gen,:) = znad;
                
                % The maximumrate of change of ideal and nadir points rk is calculated.
                if gen >= last_gen
                    max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen);
                end
                
                % update epsilon before selection
                if max_change > change_threshold && gen < 0.4 * G
                    epsilon = inf;
                else
                    epsilon =  update_epsilon(tao,epsilon,CVmax,rf,alpha,gen,Tc,cp,Problem.M);
                end
                
                ParentC = MatingSelection([Population,Archive],CSA,Problem.N,z,znad);
                Offspring = OperatorGA(Problem,ParentC);
                
                z = min(z,min(Offspring.objs,[],1));
                CSA = UpdateCSA(CSA,Offspring,Problem.N,epsilon);
                
                [Population,Archive] = EnvironmentalSelection(Population,Offspring,Archive,Problem.N,z,znad,theta,CSA.objs,epsilon);
                znad = max(znad,max(Population.objs,[],1));
                if Problem.FE >= Problem.maxFE
                    Population = Archive;
                end
            end
        end
    end
end

% Calculate the Maximum Rate of Change
function max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen)
    delta_value = 1e-6 * ones(1,size(ideal_points,2));
    rz = abs((ideal_points(gen,:) - ideal_points(gen - last_gen + 1,:)) ./ max(ideal_points(gen - last_gen + 1,:),delta_value));
    nrz = abs((nadir_points(gen,:) - nadir_points(gen - last_gen + 1,:)) ./ max(nadir_points(gen - last_gen + 1,:),delta_value));
    max_change = max([rz, nrz]);
end

% Update the value of epsilon
function epsilon = update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp,M)
    if gen > Tc
        epsilon = 0;
    else
        if rf < alpha
            epsilon = (1 - tao)^cp * epsilon_k;
        else
            epsilon = cp ^ M * epsilon_0 * ((1 - (gen / Tc)) ^ cp);
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Archive] = EnvironmentalSelection(Population,Offspring,Archive,MaxSize,z,znad,theta,CSAObjs,epsilon)
% Environmental selection of CMaDPPs

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    PopOff  = [Offspring,Population,Archive];
    PopObjs = PopOff.objs;
    PopCons = PopOff.cons;
    
    ND = NDSort(PopObjs,PopCons,inf);
    Population1 = PopOff(ND==1);
    
    IMatrix = Indicator_Cal(PopOff,z);
    IrFitness1  = min(IMatrix,[],2);
    Norm_IrFitness1 = IrFitness1;

    Population2 = PopOff(Norm_IrFitness1 >= 0);
    
    Population = [Population1,Population2];
    
    CAObj=Population.objs;
    [~,ia,~] = unique(CAObj,'rows');
    Population = Population(ia);
    N = length(Population);
    
    if N <= MaxSize
        return;
    end
    
    IMatrix = Indicator_Cal(Population,z);
    IrFitness  = min(IMatrix,[],2);
    Norm_IrFitness = IrFitness;
    
    L = Norm_IrFitness*Norm_IrFitness';
    
    CAObj = Population.objs;
    CAObj2 = (CAObj-repmat(z,N,1))./(repmat(znad,N,1)-repmat(z,N,1));
   
    D = pdist2(CAObj2,CAObj2,'cosine');

    D(1:size(D,1)+1:end) = 0;
    
    H=(1-theta)*exp(-D);

    KM = H.*L;
    
    L = decompose_kernel(KM);
    Choose = sample_dpp(L,MaxSize);
    Population = Population(Choose);
    
    PopCons = PopOff.cons;
    CV = sum(max(0,PopCons),2);
    FeasibleInd = CV <= epsilon;
    Archive = UpdateArchive(PopOff(FeasibleInd),MaxSize,z,znad,CSAObjs,theta,epsilon);
end
```

### `Indicator_Cal.m`
```matlab
function IMatrix = Indicator_Cal(MaxPop,Zmin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    PopObj   = MaxPop.objs;
    [Num,~]  = size(PopObj);
    Cons     = max(0,MaxPop.cons);
    NormCons = Cons./repmat(max(1,max(Cons,[],1)),Num,1);
    CV       = sum(NormCons,2);

    % shift the objective space to R+
    PopObj = PopObj - repmat(Zmin,Num,1) + 1e-6;

    % calculate the indicator matrix
    IMatrix = ones(Num,Num);
    for i = 1:1:Num
        Ci = CV(i);    
        if Ci == 0 %%%%% Xi is feasible
            Fi               = PopObj(i,:);
            Ir               = log(repmat(Fi,Num,1)./PopObj);
            MaxIr            = max(Ir,[],2);
            MinIr            = min(Ir,[],2);
            CVA              = MaxIr;
            DomInds          = find(MaxIr<=0);
            CVA(DomInds)     = MinIr(DomInds);
            IndicatorV       = CVA;
        else  %%%%% Xi is an infeasible solution
            IC               = repmat(Ci+1e-6,Num,1)./(CV+1e-6); 
            Fi = PopObj(i,:);
            MaxF = max(repmat(Fi,Num,1),PopObj);
            MinF = min(repmat(Fi,Num,1),PopObj);
            CVF = max(MaxF./MinF,[],2);
            IndicatorV       = log(max([CVF,IC],[],2));
        end
        IMatrix(:,i)     = IndicatorV;
        IMatrix(i,i)     = Inf;
    end
end
```

### `MatingSelection.m`
```matlab
function ParentC = MatingSelection(CA,DA,N,z,znad)
% The mating selection of Two_Arch2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    EA=[CA,DA];
    CAObj=EA.objs;
    [N1,~]=size(CAObj);
    CAObj2 = (CAObj-repmat(z,N1,1))./(repmat(znad,N1,1)-repmat(z,N1,1));
    D = pdist2(CAObj2,CAObj2,'cosine');
    D=D+eye(N1);
    [cos,mincos]=min(D);
    CAO=sum(CAObj2.^2,2);
    minCAO=CAO(mincos);
    ch=(minCAO-CAO)>0;
    ch3=(1-ch).*mincos'+(ch).*(1:N1)';
    cos=1-cos;
    cos=(cos-min(cos))./repmat((max(cos)-min(cos)),1,N1);
    ParentC=[];
    for i=1:1:2*N
        k=randi(N1);
        if rand<cos(k)
            ParentC=[ParentC,EA(ch3(k))];
        else
            ParentC=[ParentC,EA(k)];
        end
    end
end
```

### `UpdateArchive.m`
```matlab
function Population = UpdateArchive(PopOff,MaxSize,z,znad,DAobj,theta,epsilon)
% Update the archive for feasible solutions by MaOEADPPs approach

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    PopObjs = PopOff.objs;
    PopCons = PopOff.cons;
    CV = sum(max(0,PopCons),2);
    fIndex = all(CV <= epsilon,2);
    PopCons(fIndex,:) = 0;
    
    ND = NDSort(PopObjs,PopCons,1);
    Population = PopOff(ND==1);
    
    CAObj=Population.objs;
    [~,ia,~] = unique(CAObj,'rows');
    Population = Population(ia);

    N  = length(Population);
    
    if N <= MaxSize
        return;
    end
    
    CAObj=Population.objs;
    CAObj2 = (CAObj-repmat(z,N,1))./(repmat(znad,N,1)-repmat(z,N,1));
    
    nad=max(DAobj)+1E-6;
    nn=sum((nad-CAObj)<0,2);
    [N1,~]=size(DAobj);
    DAobj = (DAobj-repmat(z,N1,1))./(repmat(znad,N1,1)-repmat(z,N1,1));

    big=max(sqrt(sum(DAobj.^2,2)))+1E-6;
    nbig=sqrt(sum(CAObj2.^2,2))>big;
    nn=nn+nbig;
   
    D = pdist2(CAObj2,CAObj2,'cosine');

    D(1:size(D,1)+1:end) = 0;

    H=(1-theta)*exp(-D);

    value=(sum((CAObj2).^2,2));

    value(nn==0)=min(value)/2;
    value=value/max(value);
    L=value*value';
    H=H./L;
    
    L = decompose_kernel(H);
    Choose = sample_dpp(L,MaxSize);
    Population = Population(Choose);
end
```

### `UpdateCSA.m`
```matlab
function [CSA]= UpdateCSA(CSA,Offspring,MaxSize,epsilon)
% Update CSA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    % Select epsilon feasible solutions
    CSA = [Offspring,CSA];
    CV = sum(max(0,CSA.cons),2);
    fIndex = all(CV <= epsilon,2);
    CSA = CSA(fIndex);
	CSAObj = CSA.objs;
    
    [N,M] = size(CSAObj);

    if N <= MaxSize
        return;
    end
    
    cunum=ceil((MaxSize)/(3*M));
    DAObj2 = CSAObj;
    DAObj3=DAObj2.^2;
    DAO=sum(DAObj3,2);
    CHIndex = [];
    minIndex2=[];
    for i=1:M
        minfx=CSAObj(:,i);
        tempminfx=min(minfx)+1E-6;

        if sum(minfx<tempminfx)>cunum
            mindex=find((minfx<tempminfx));
            [~,minIndex]=sort(DAO(mindex));
            minIndex=mindex(minIndex(1:cunum));
        else
            [~,minIndex]=sort(minfx);
            minIndex=minIndex(1:cunum);
        end
      
        minfx2=DAO-DAObj3(:,i);
        tempminfx=min(minfx2)+1E-6;

        if sum(minfx2<tempminfx)>2*cunum
            mindex=find((minfx2<tempminfx));
            [~,minndex]=sort(DAO(mindex));
            index=mindex(minndex(1:2*cunum));
        else
            [~,index]=sort(minfx2);
            index=index(1:2*cunum);
        end

       CHIndex = [CHIndex,index'];

       minIndex2=[minIndex2,minIndex'];
    end

    Choose = [CHIndex,minIndex2];
    CSA    = CSA(Choose);
end
```

### `decompose_kernel.m`
```matlab
function L = decompose_kernel(M)
% Get the eigenvectors and lambdas of M

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    L.M   = M;
    [V,D] = eig(M);
    L.V   = real(V);
    L.D   = real(diag(D));
end
```

### `sample_dpp.m`
```matlab
 function Y = sample_dpp(L,k)
% The function of DPPs, the inputs include the kernal matrix L and the size k

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Fei Ming (email: 20151000334@cug.edu.cn)

    n     = size(L.D,1);
    [~,i] = sort(L.D);
    v     = i(n-k+1:n);
    k     = length(v);
    V     = L.V(:,v);

    % iterate
    Y = zeros(k,1);
    for i = k:-1:1

        if size(V,2)==1 && i~=1
            V=abs(V);
            [~,b1]=sort(V);
            bb=b1(n-i+1:n);
            for ii=i:-1:1
                Y(ii)=bb(ii);
            end
            break;

        end
        % compute probabilities for each item
        P = sum(V.^2,2);
        P = P / sum(P);

        [~,Y(i)] = max(P);

        % choose a vector to eliminate
        j  = find(V(Y(i),:),1);
        Vj = V(:,j);
        V  = V(:,[1:j-1 j+1:end]);

        % update V

        V = V - bsxfun(@times,Vj,V(Y(i),:)/Vj(Y(i)));

        % orthogonalize
        V = orth(V);
    end
    Y = sort(Y);
end
```
