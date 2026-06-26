# LRMOEA

**Tags**: <2025> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse> <robust>

## Description
Large-scale robust multi-objective evolutionary algorithm

## Reference
S. Shao, Y. Tian, L. Zhang, K. C. Tan, and X. Zhang. An evolutionary algorithm for solving large-scale robust multi-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2476-2490.

## Source Code

### `ArchUpdate.m`
```matlab
function [Arch] = ArchUpdate(Problem,arch,TemArch,thea,eta)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    TDec = arch.decs;
    Mask = arch.masks;
    % Perturb Solutions
    PopDec = TDec.*Mask;
    Population = Problem.Perturb(PopDec,1);

    Pop  = Population.objs;
    arch = Memory(arch,Pop);
    Mr   = arch.mrs;
    arch = arch(Mr<=thea);
    if isempty(arch)
        Arch = TemArch;
    else
        Tdec = (TemArch.decs).*(TemArch.masks);
        Decs = (arch.decs).*(arch.masks);
        L    = size(Tdec,1);
        n    = size(Decs,1);
        Tobj = TemArch.objs;
        a    = ones(L,1);
        for i = 1 : L
            for j = 1 : n
                if all(Tdec(i,:)==Decs(j,:))
                    a(i,1)  = 0;
                    arch(j) = Memory(arch(j),Tobj(i,:));
                    break;
                end
            end
        end
        t = TemArch(a==1);
        if ~isempty(t)
            maf   = max(t.objs);
            Tobjs = arch.objs;
            inx   = (sum(Tobjs,2)<(eta.*sum(maf)));
            arch  = arch(inx);
        end
        Arch = [arch,t];
    end
end

function value = Memory(arch,pop)
    for k = 1 : size(arch,2)
        arch(k).mobj = [arch(k).mobj;pop(k,:)];
        arch(k).Gn   = arch(k).Gn + 1;
        ob           = arch(k).mobj;
        arch(k).mr   = mean(mean(abs(ob-min(ob)))./mean(ob));
    end
    value = arch;
end
```

### `Construction.m`
```matlab
function [rf,mf,ar,wr] = Construction(arch,W,score)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [~,n]  = size(arch);
    [N ,~] = size(W);
    if size(arch,2) > size(W,1)
        a(N+1).r = [];
        de       = arch.decs;
        ma       = arch.masks;
        m        = arch.Gns;
        A_sp     = zeros(1,n);
        Sp       = zeros(1,n);
        for i = 1 : n
            a(arch(i).tno).r = [a(arch(i).tno).r,i];
            sp        = sum(arch(i).mobj,2);
            sp_b      = min(sp);
            A_sp(1,i) = mean(sum(abs(sp - sp_b)));
            A_sp(1,i) = 1/((sum(ma(i,:)&(score>0.5)))/size(score,2));
            Sp(1,i)   = mean(sp);
        end
        rch = [];
        w   = [];
        j   = 0;
        for r = 1 : N
            if a(r).r > 0
                [~,c] = size(a(r).r);    
                s     = zeros(1,c);
                t     = zeros(1,c);
                gn    = 0;
                for d = 1 : c
                    gn = gn + a(r).r(1,d);
                end
                gn = gn/c;
                for d = 1 : c
                    no = a(r).r(1,d);
                    if( m(no)>gn )
                        T = 0;
                    else
                        T = gn-m(no);
                    end
                    s(1,d) = Sp(1, no)+(A_sp(1,no));
                    t(1,d) = no;
                end
                [~,b]   = sort(s(1,:));
                j       = j + 1;
                tt      = t(1,b(1,1));
                RF(j,:) = de(t(1,b(1,1)),:);
                MF(j,:) = ma(t(1,b(1,1)),:);
                rch     = [rch,tt];
                w       = [w,r];
            end
        end
    end
    ar = rch;
    wr = w;
    rf = RF;
    mf = MF;
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,N)
% Environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Delete duplicated solutions
    [~,uni]    = unique(Population.objs,'rows');
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
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
end
```

### `Final.m`
```matlab
function [Population] = Final(Problem,Arch,N,W,score)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if length(Arch) <= N
        Population = Problem.Evaluation(Arch.decs.*Arch.masks);
    else
        RF = [];
        MF = [];
        while size(W,1) > 0
            [srch,W]  = assign(Arch,W);
            [rf,mf,rch,wr] = Construction(srch,W,score);
            RF        = [rf;RF];
            MF        = [mf;MF];
            Arch(rch) = [];
            W(wr,:)   = [];
        end
        Population = Problem.Evaluation(RF.*MF);
    end
end

function [ar,w] = assign(arch,W)
    Popobj = arch.objs;
    L      = max(Popobj)-min(Popobj);
    objs   = (Popobj-min(Popobj))./L ;
    [~,n]  = size(arch);
    [N ,~] = size(W);
    for x = 1 : n
        obj = objs(x,:);
        for y = 1 : N
            s = sum(W(y,:).*obj,2);
            m = sqrt(sum(W(y,:).*W(y,:),2)*sum(obj.*obj,2));
            dang(1,y) = acos(s/m);
            [~,h]     = sort(dang);
        end
        arch(x).tno = h(1);
    end
    w  = W;
    ar = arch;
end
```

### `FitnessCal.m`
```matlab
function[score] = FitnessCal(Mask,FrontNo,score)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,d] = size(Mask);
    Fit   = zeros(1,d);
    b     = 0;
    h     = 0;
    for i = 1 : N
        h = h + 1;
        if FrontNo(h) <= 1
            b = b + 1;
            a = 1;
            for j = 1 : d
                fit = 1/(2+sqrt(N)*Mask(i,j));
                Fit(1,a)  = fit ;
                a = a + 1;
            end
            score = score + Fit;
        end
    end
    score = score./b;
end
```

### `Initialization.m`
```matlab
function [Arch,score] = Initialization(Population,Dec,Mask,FrontNo)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [Next]     =  find(FrontNo==1);
    Population = Population(Next).objs;
    Dec        = Dec(Next,:);
    Mask       = Mask(Next,:);
    score      = sum(Mask,1)/size(Mask,1);
    Arch       = archives(Population,Dec,Mask);
end
```

### `LRMOEA.m`
```matlab
classdef LRMOEA < ALGORITHM
% <2025> <multi> <real/integer/binary> <large/none> <constrained/none> <sparse> <robust>
% Large-scale robust multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% S. Shao, Y. Tian, L. Zhang, K. C. Tan, and X. Zhang. An evolutionary
% algorithm for solving large-scale robust multi-objective optimization
% problems. IEEE Transactions on Evolutionary Computation, 2025, 29(6):
% 2476-2490.
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
            [thea,eta] = Algorithm.ParameterSet(0.2,1.15);
            W = UniformPoint(Problem.N,Problem.M);
            if Problem.encoding(1)==4
                Dec = ones(Problem.N,Problem.D);
            else
                Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            end
            Mask = false(size(Dec));
            for i = 1 : Problem.N
                Mask(i,randperm(end,ceil(rand.^2*end))) = true;
            end
            Population = Problem.Evaluation(Dec.*Mask);
            [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Dec,Mask,Problem.N);
            obj   = Population.objs;
            score = ones(1,Problem.D);
            score = FitnessCal(Mask,FrontNo,score);
            Arch  = archives(obj,Dec,Mask);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool       = TournamentSelection(2,2*Problem.N,FrontNo,-CrowdDis);
                [OffDec,OffMask] = LROperator(Problem,Dec(MatingPool,:),Mask(MatingPool,:),score);
                Offspring = Problem.Evaluation(OffDec.*OffMask);
                [Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
                score = sum(Arch.masks)/size(Arch,2);
                if size(score,2) == 1
                    score = ones(1,Problem.D);
                end
                TemArch = Initialization(Population,Dec,Mask,FrontNo);
                Arch    = ArchUpdate(Problem,Arch,TemArch,thea,eta);
                if Problem.FE >= Problem.maxFE
                    Population = Final(Problem,Arch,Problem.N,W,score);
                end
            end
        end
    end
end
```

### `LROperator.m`
```matlab
function [OffDec,OffMask] = LROperator(Problem,ParentDec,ParentMask,Fitness)
% Operator

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);
    
    %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        if rand < 0.5
            index = find(Parent1Mask(i,:)&~Parent2Mask(i,:));
            index = index(TS(Fitness(index)));
            OffMask(i,index) = 0;
        else
            index = find(~Parent1Mask(i,:)&Parent2Mask(i,:));
            index = index(TS(-Fitness(index)));
            OffMask(i,index) = Parent2Mask(i,index);
        end
    end
    
    %% Mutation for mask
    for i = 1 : N/2
        if rand < 0.5
            index = find(OffMask(i,:));
            index = index(TS(Fitness(index)));
            OffMask(i,index) = 0;
        else
            index = find(~OffMask(i,:));
            index = index(TS(-Fitness(index)));
            OffMask(i,index) = 1;
        end
    end
    
    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

function index = TS(Fitness)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
        index = TournamentSelection(2,1,Fitness);
    end
end
```

### `archives.m`
```matlab
classdef archives < handle
    
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        dec;
        mask;
        obj;
        mobj;
        mr;
        tno;
        Gn;
    end
    methods      
        function [Arch] = archives(Pop,Dec,Mask)
            if nargin > 0
                Arch(1,size(Pop,1)) = Arch;
                for i = 1 : size(Pop,1)
                    Arch(i).dec  = Dec(i,:);
                    Arch(i).mask = Mask(i,:);
                    Arch(i).obj  = Pop(i,:);
                    Arch(i).mobj = Pop(i,:);
                    Arch(i).mr   = 0;
                    Arch(i).tno  = 0;
                    Arch(i).Gn   = 1;
                end
            end
        end
        function value = decs(Arch)
            value = cat(1,Arch.dec);
        end
        function value = objs(Arch)
            value = cat(1,Arch.obj);
        end
        function value = masks(Arch)
            value = cat(1,Arch.mask);
        end
        function value = mrs(Arch)
            value = cat(1,Arch.mr);
        end
        function value = tnos(Arch)
            value = cat(1,Arch.tno);
        end
        function value =Gns(Arch)
            value = cat(1,Arch.Gn);
        end
    end
end
```
