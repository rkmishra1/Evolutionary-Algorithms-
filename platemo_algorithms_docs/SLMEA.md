# SLMEA

**Tags**: <2022> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>

## Description
Super-large-scale multi-objective evolutionary algorithm

## Reference
Y. Tian, Y. Feng, X. Zhang, and C. Sun. A fast clustering based evolutionary algorithm for super-large-scale sparse multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2022, 9(4): 1-16.

## Source Code

### `CalculateFitness.m`
```matlab
function [Fitness,c,Z,O] = CalculateFitness(ArchiveMask,D)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    len     = length(ArchiveMask(:,1));
    ratio   = sum(ArchiveMask,1)./len;
    R       = abs(ratio-0.5);
    [a,~]   = min(R,[],2);
    EL      = find(R==a);
    ll      = length(EL);
    lo      = randperm(ll,1);
    vector  = ArchiveMask(:,EL(lo));
    vector  = repmat(vector,1,D);
    Hd      = sum(xor(vector,ArchiveMask),1);
    Ad      = sum(and(vector,ArchiveMask),1);
    Fitness = Hd./(Hd+Ad);
    c       = find(ratio>0&ratio<1);
    Z       = find(ratio==0);
    O       = find(ratio==1);
end
```

### `CpuGroup.m`
```matlab
function [Index,MAX] = CpuGroup(numberOfGroups,xPrime,numberOfVariables)       

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    varsPerGroup = floor(numberOfVariables/numberOfGroups);
    if varsPerGroup == 1
        Index = linspace(1,numberOfVariables,numberOfVariables);
        MAX   = numberOfVariables;
    else
        B        = ones(1,varsPerGroup*numberOfGroups);
        remain   = ones(1,(numberOfVariables-varsPerGroup*numberOfGroups))*(numberOfGroups+1);
        R        = reshape(B,varsPerGroup,numberOfGroups);
        k        = linspace(1,numberOfGroups,numberOfGroups);
        index    = R.*repmat(k,varsPerGroup,1);
        index    = reshape(index,1,varsPerGroup*numberOfGroups);
        INDEX    = [index remain];
        [~,I]    = sort(xPrime);
        Index(I) = INDEX;
        if(mod(numberOfVariables,numberOfGroups)==0)
            MAX = numberOfGroups;
        else
            MAX = numberOfGroups + 1;
        end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,N)
% The environmental selection of SLMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions
    objs = Population.objs;
    objs = gather(objs);
    
    [objs,uni] = unique(objs,'rows');
    Population = Population(uni);
    Dec        = Dec(uni,:);
    Mask       = Mask(uni,:);
    N          = min(N,length(Population));
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(objs,gather(Population.cons),N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `GpuGroup.m`
```matlab
function [Index,MAX] = GpuGroup(numberOfGroups,xPrime,numberOfVariables)       

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    varsPerGroup = floor(numberOfVariables/numberOfGroups);
    if varsPerGroup ==1
        Index = gpuArray.linspace(1,numberOfVariables,numberOfVariables);
        MAX   = numberOfVariables;
    else
        B        = gpuArray.ones(1,varsPerGroup*numberOfGroups);
        remain   = gpuArray.ones(1,(numberOfVariables-varsPerGroup*numberOfGroups))*(numberOfGroups+1);
        R        = reshape(B,varsPerGroup,numberOfGroups);
        k        = gpuArray.linspace(1,numberOfGroups,numberOfGroups);
        index    = R.*repmat(k,varsPerGroup,1);
        index    = reshape(index,1,varsPerGroup*numberOfGroups);
        INDEX    = [index remain];
        [~,I]    = sort(xPrime);
        Index(I) = INDEX;
        if(mod(numberOfVariables,numberOfGroups)==0)
            MAX = numberOfGroups;
        else
            MAX = numberOfGroups + 1;
        end
    end
end
```

### `Operator.m`
```matlab
function [OffDec,OffMask,long,numGroup] = Operator(Problem,ParentDec,ParentMask,Fitness,Mix,P,T,Zero,One,Upper,Lower,dt,numGroup,useGPU,GlobalGen)
% The operator of SLMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Initialization
    [N,D]        = size(ParentDec);
    ClOffMask    = [];
    Parent1Mask  = ParentMask(1:N/2,:);
    Parent2Mask  = ParentMask(N/2+1:end,:);
    Parent11Mask = [];
    Parent12Mask = [];
    
    ClOffDec    = [];
    Parent1Dec  = ParentDec(1:N/2,:);
    Parent2Dec  = ParentDec(N/2+1:end,:);
    Parent11Dec = [];
    Parent12Dec = []; 
    
    long    = 0;
    Fitness = Fitness(Mix);
    
   %% Group
    if GlobalGen >= T
        numGroup = min(200,min(length(Mix),ceil(numGroup*dt)));
        if ~isempty(Mix)
           if ~useGPU
               [index,MAX] = CpuGroup(numGroup,Fitness,length(Mix));
           else
               [index,MAX] = GpuGroup(numGroup,Fitness,length(Mix));
           end
        else
            index = [];
            MAX   = 0;
        end
       if ~useGPU
          Index = zeros(1,length(Lower)); 
       else
          Index = gpuArray.zeros(1,length(Lower)); 
       end
       if ~isempty(Zero)&&~isempty(One)     % Grouping all 1 & all 0
           Index(Zero) = MAX+1;
           Index(One)  = MAX+2;
           MAX = MAX+2;
       elseif isempty(Zero)&&~isempty(One)  % Only Grouping all 1
           Index(One) = MAX+1;
           MAX = MAX+1;
       elseif isempty(One)&&~isempty(Zero)  % Only Grouping all 0
           Index(Zero) = MAX+1;
           MAX = MAX+1;
       end
       Index(Mix) = index;
    end
    %% Group and encoder
    location = rand(1,N/2)<P;
    if ~isempty(find(location>0, 1)) && GlobalGen>=T
       [Parent11Mask]    = Encoder(Parent1Mask(location,:),Index,MAX,useGPU);
       [Parent12Mask]    = Encoder(Parent2Mask(location,:),Index,MAX,useGPU);
       Parent21Mask      = Parent1Mask(~location,:);
       Parent22Mask      = Parent2Mask(~location,:);
       Parent11EncodeDec = (Parent1Dec(location,:)-repmat(Lower,length(find(location==1)),1))./repmat((Upper-Lower),length(find(location==1)),1);
       Parent12EncodeDec = (Parent2Dec(location,:)-repmat(Lower,length(find(location==1)),1))./repmat((Upper-Lower),length(find(location==1)),1);
       Parent11Dec       = EncoderDec(Parent11EncodeDec,Index,MAX);
       Parent12Dec       = EncoderDec(Parent12EncodeDec,Index,MAX);
       Parent21Dec       = Parent1Dec(~location,:);
       Parent22Dec       = Parent2Dec(~location,:);
       if useGPU == 0
           lower = zeros(1,MAX);
           upper = ones(1,MAX);           
       else
           lower = gpuArray.zeros(1,MAX);
           upper = gpuArray.ones(1,MAX);           
       end
    else
        Parent21Mask = Parent1Mask;
        Parent22Mask = Parent2Mask;
        
        Parent21Dec = Parent1Dec;
        Parent22Dec = Parent2Dec;                
    end
    %% Crossover and mutation for mask
    if ~isempty(find(location>0, 1)) && GlobalGen>=T
        ClOffMask = SLMEA_GAhalf([Parent11Mask;Parent12Mask],lower,upper,'binary',useGPU);
        ClOffMask = Decode(ClOffMask,Index);
        long      = length(ClOffMask(:,1));
    end
    OrOffMask = SLMEA_GAhalf([Parent21Mask;Parent22Mask],Lower,Upper,'binary',useGPU);
    OffMask   = [ClOffMask;OrOffMask];
    if any(Problem.encoding~=4)
        if ~isempty(find(location>0, 1)) && GlobalGen>=T 
            ClOffDec = SLMEA_GAhalf([Parent11Dec;Parent12Dec],lower,upper,'real',useGPU);
            ClOffDec = DecodeDec(ClOffDec,Index);
            ClOffDec = ClOffDec.*repmat((Upper-Lower),length(find(location==1)),1)+repmat(Lower,length(find(location==1)),1);
        end
        OrOffDec = SLMEA_GAhalf([Parent21Dec;Parent22Dec],Lower,Upper,'real',useGPU);
        OffDec   = [ClOffDec;OrOffDec];
        OffDec   = min(max(OffDec,repmat(Lower,N/2,1)),repmat(Upper,N/2,1));
        OffDec(:,Problem.encoding==4) = 1;
    else
        if ~useGPU
        	OffDec = ones(N/2,D); 
        else
        	OffDec = gpuArray.ones(N/2,D); 
        end
    end
end

function NewMask= Encoder(Mask,Index,numGroup,useGPU)
    C  = Index;
    Uc = linspace(1,numGroup,numGroup);
    N  = size(Mask,1);
    CC = arrayfun(@(i)find(C==i),Uc,'UniformOutput',false);
    if ~useGPU
        A = cellfun(@(i)mean(Mask(:,i),2)>rand(N,1),CC,'UniformOutput',false);
    else
        A = cellfun(@(i)mean(Mask(:,i),2)>gpuArray.rand(N,1),CC,'UniformOutput',false);
    end
    NewMask = cell2Mat(A);
end

function OMask = Decode(NewMask,Index)
	OMask = NewMask(:,Index);
end
function NewDec = EncoderDec(Dec,Index,numGroup)
    C      = Index;
    Uc     = linspace(1,numGroup,numGroup);
    CC     = arrayfun(@(i)find(C==i),Uc,'UniformOutput',false);
    A      = cellfun(@(i)mean(Dec(:,i),2),CC,'UniformOutput',false);
    NewDec = cell2Mat(A);
end

function ODec = DecodeDec(NewDec,Index)
	ODec = NewDec(:,Index);
end

function m = cell2Mat(c)
    rows = size(c,1);
    cols = size(c,2);   
    if rows < cols
        m = cell(rows,1);
        for n = 1 : rows
            m{n} = cat(2,c{n,:});
        end
        m = cat(1,m{:});
    else
        m = cell(1, cols);
        for n = 1 : cols
            m{n} = cat(1,c{:,n});
        end    
        m = cat(2,m{:});
    end
end
```

### `SLMEA.m`
```matlab
classdef SLMEA < ALGORITHM
% <2022> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse>
% Super-large-scale multi-objective evolutionary algorithm
% useGPU --- 0 --- Whether use GPU acceleration

%------------------------------- Reference --------------------------------
% Y. Tian, Y. Feng, X. Zhang, and C. Sun. A fast clustering based
% evolutionary algorithm for super-large-scale sparse multi-objective
% optimization. IEEE/CAA Journal of Automatica Sinica, 2022, 9(4): 1-16.
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
            useGPU = Algorithm.ParameterSet(0);
             
            %% Generate initial variables
            P           = 0.5;
            ArchiveObj  = [];
            ArchiveMask = [];
            numGroup    = 10;
            Lr          = 0;
            RC          = 1;
            T           = 20;
            GlobalGen   = 1;
            
            %% Population initialization
            Mask = false(Problem.N,Problem.D);
            for i = 1 : Problem.N
                Mask(i,TournamentSelection(2,ceil(rand*Problem.D),ones(1,Problem.D))) = 1;
            end
            if ~useGPU
                Lower = Problem.lower;
                Upper = Problem.upper;
                Dec   = unifrnd(repmat(Lower,Problem.N,1),repmat(Upper,Problem.N,1));
                Dec(:,Problem.encoding==4) = 1;
            else
                Lower = gpuArray(Problem.lower);
                Upper = gpuArray(Problem.upper);
                Mask  = gpuArray(Mask);
                Dec   = unifrnd(repmat(Lower,Problem.N,1),repmat(Upper,Problem.N,1));
                Dec(:,Problem.encoding==4) = 1;
            end            
            Population = Problem.Evaluation(Dec.*Mask);
            
           %% Update Archive
            [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,Problem.N);
            PopObj      = gather(Population.objs);
            ArchiveObj  = [ArchiveObj;PopObj(FrontNo==1,:)];
            ArchiveMask = [ArchiveMask;Mask(FrontNo==1,:)];
            [Fitness,Mix,Zero,One] = CalculateFitness(ArchiveMask,Problem.D);
            
           %% optimization
            while Algorithm.NotTerminated(Population)
                % Update population
                MatingPool = TournamentSelection(2,2*Problem.N,FrontNo,-CrowdDis);
                [OffDec,OffMask,long,numGroup] = Operator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),Fitness,Mix,P,T,Zero,One,Upper,Lower,RC,numGroup,useGPU,GlobalGen);
                Offspring  = Problem.Evaluation(OffDec.*OffMask);
                GlobalGen  = GlobalGen + 1;
                [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
                % Update P
                if GlobalGen >= T+1
                    Fn  = NDSort(gather(Offspring.objs),1); 
                    LOC = find(Fn==1);
                    a   = find(LOC<=long);
                    b   = find(LOC>long);
                    P   = min(0.95,max(0.05,0.5*(P+(length(a)*(Problem.N-long)/(length(a)*(Problem.N-long)+length(b)*long)))));
                    RC  = exp((length(a)/(long+0.00001)-Lr)/numGroup);
                    Lr  = length(a)/long;
                end
                % Update Archive
                PopObj          = gather(Population.objs);
                ArchiveObj      = [ArchiveObj;PopObj(FrontNo==1,:)];
                ArchiveMask     = [ArchiveMask;Mask(FrontNo==1,:)];
                [ArchiveObj,ia] = unique(ArchiveObj,'rows');
                [frontNo,~]     = NDSort(ArchiveObj,1);
                ArchiveObj      = ArchiveObj(frontNo==1,:);
                ArchiveMask     = ArchiveMask(ia,:);
                ArchiveMask     = ArchiveMask(frontNo==1,:);
                if size(ArchiveObj,1) > 100
                    ArLocation  = randperm(size(ArchiveObj,1),100);
                    ArchiveObj  = ArchiveObj(ArLocation,:);
                    ArchiveMask = ArchiveMask(ArLocation,:);
                end
                % Calculate Fitness
                [Fitness,Mix,Zero,One] = CalculateFitness(ArchiveMask,Problem.D);
             end
         end
     end
end
```

### `SLMEA_GAhalf.m`
```matlab
function Offspring = SLMEA_GAhalf(Parent,Lower,Upper,Encoding,useGPU,Parameter)
% Genetic operators accelerated by GPU

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    if nargin > 5
        [proC,disC,proM,disM] = deal(Parameter{:});
    else
        [proC,disC,proM,disM] = deal(1,20,1,20);
    end
    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);
 
    switch Encoding
        case 'binary'
            %% Genetic operators for binary encoding
            % One point crossover
            k = repmat(1:D,N,1) > repmat(randi(D,N,1),1,D);
            k(repmat(rand(N,1)>proC,1,D)) = false;
            Offspring    = Parent1;
            Offspring(k) = Parent2(k);
            % Bitwise mutation
            Site = rand(N,D) < proM/D;
            Offspring(Site) = ~Offspring(Site);
        otherwise
            %% Genetic operators for real encoding
            % Simulated binary crossover
            if ~useGPU
                beta = zeros(N,D);
                mu   = rand(N,D); 
            else 
                beta = gpuArray.zeros(N,D);
                mu   = gpuArray.rand(N,D);                
            end
            beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
            beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
            if ~useGPU
                beta = beta.*(-1).^randi([0,1],N,D);       
                beta(rand(N,D)<0.5) = 1;
                beta(repmat(rand(N,1)>proC,1,D)) = 1;                
            else
                beta = beta.*(-1).^gpuArray.randi([0,1],N,D);       
                beta(gpuArray.rand(N,D)<0.5) = 1;
                beta(repmat(gpuArray.rand(N,1)>proC,1,D)) = 1;                
            end
            Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
            % Polynomial mutation
            Lower = repmat(Lower,N,1);
            Upper = repmat(Upper,N,1);
            if ~useGPU
                Site = rand(N,D) < proM/D;
                mu   = rand(N,D);                
            else
                Site = gpuArray.rand(N,D) < proM/D;
                mu   = gpuArray.rand(N,D);                
            end
            temp = Site & mu<=0.5;
            Offspring       = min(max(Offspring,Lower),Upper);
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                              (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
            temp = Site & mu>0.5; 
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                              (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
            Offspring       = min(max(Offspring,Lower),Upper);              
    end
end
```
