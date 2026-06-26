# MOEA-RE

**Tags**: <2020> <multi> <real/integer/label/binary/permutation> <robust>

## Description
Multi-objective evolutionary algorithm with robustness enhancement

## Reference
Z. He, G. G. Yen, and J. Lv. Evolutionary multiobjective optimization with robustness enhancement. IEEE Transactions on Evolutionary Computation, 2020, 24(3): 494-507.

## Source Code

### `Distrubance.m`
```matlab
function [PopObjX,OffObjX,ArcObjX] = Distrubance(Problem,PopDec,OffDec,ArcDec)
% Re-evaluate solutions with the same disturbance

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N1 = size(PopDec,1);
    N2 = size(OffDec,1);
    P  = Problem.Perturb([PopDec;OffDec;ArcDec],1);
    PopObjX = P(1:N1).objs;
    OffObjX = P(N1+1:N1+N2).objs;
    ArcObjX = P(N1+N2+1:end).objs;
end
```

### `EnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis,PopObjX] = EnvironmentalSelection(Population,N,PopObjX)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(PopObjX,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(PopObjX,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    PopObjX    = PopObjX(Next,:);
end
```

### `FinalSelection.m`
```matlab
function Population = FinalSelection(Archive,W,ArcW,ArcSP)
% Select final robust solutions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the convergence performance of solutions
    genx = cellfun(@length,ArcW);
    A_SP = cellfun(@(s)sum(s-min(s)),ArcSP);
    SM   = mean(A_SP);
    GM   = min(30,mean(genx));
    S    = cellfun(@mean,ArcSP) + SM/GM*max(0,GM-genx);
    
    %% Select a solution for each weight vector
    Selected = zeros(1,size(W,1));
    while true
        % Find a solution for each weight vector
        ClosestW  = cellfun(@mode,ArcW);
        AssignedW = [];
        for i = find(Selected==0)
            current = find(ClosestW==i);
            if ~isempty(current)
                [~,best]    = min(S(current));
                Selected(i) = current(best);
                AssignedW   = [AssignedW,i];
            end
        end
        % Delete the assigned weight vectors
        ArcW = cellfun(@(s)s(~ismember(s,AssignedW)),ArcW,'UniformOutput',false);
        if isempty(AssignedW)
            break;
        end
    end
    Population = Archive(Selected(Selected~=0));
end
```

### `MOEARE.m`
```matlab
classdef MOEARE < ALGORITHM  
% <2020> <multi> <real/integer/label/binary/permutation> <robust>
% Multi-objective evolutionary algorithm with robustness enhancement
% alpha --- 1.5 --- Parameter for deleting solutions from archive

%------------------------------- Reference --------------------------------
% Z. He, G. G. Yen, and J. Lv. Evolutionary multiobjective optimization
% with robustness enhancement. IEEE Transactions on Evolutionary
% Computation, 2020, 24(3): 494-507.
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
            alpha = Algorithm.ParameterSet(1.5);
            
            %% Generate random population
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N,Population.objs);
            z        = min(Population.objs,[],1);
            Archive  = Population;
            [~,ArcW] = min(pdist2(Archive.objs-z,W,'cosine'),[],2);
            ArcW     = num2cell(ArcW);
            ArcSP    = num2cell(sum(Archive.objs,2));

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Optimization
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                z          = min([z;Offspring.objs],[],1);
                [PopObjX,OffObjX,ArcObjX]             = Distrubance(Problem,Population.decs,Offspring.decs,Archive.decs);
                [Population,FrontNo,CrowdDis,PopObjX] = EnvironmentalSelection([Population,Offspring],Problem.N,[PopObjX;OffObjX]);
                % Update archive
                [SOI,SOIObjX]        = SelectSOI(Population,PopObjX,20,z);
                [Archive,ArcW,ArcSP] = UpdateArchive(Archive,ArcObjX,SOI,SOIObjX,alpha,z,W,ArcW,ArcSP);
                % Final robust solution selection
                if Problem.FE >= Problem.maxFE
                    Population = FinalSelection(Archive,W,ArcW,ArcSP);
                end
            end
        end
    end
end
```

### `SelectSOI.m`
```matlab
function [SOI,SOIObjX] = SelectSOI(Population,PopObjX,N,z)
% Select the solutions of interest

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate angles between solutions and ASF
    Angle = acos(1-pdist2(PopObjX,PopObjX,'cosine'));
    Angle(logical(eye(length(Angle)))) = inf;
    ASF = max((PopObjX-z).*sum(PopObjX,2)./PopObjX,[],2);
    
    %% Delete solutions
    Remain = 1 : size(PopObjX,1);
    while length(Remain) > N
        [Dis,x] = min(Angle(Remain,Remain),[],2);
        [~,y]   = min(Dis);
        x = x(y);
        if ASF(Remain(x)) > ASF(Remain(y))
            Remain(x) = [];
        else
            Remain(y) = [];
        end
    end
    SOI     = Population(Remain);
    SOIObjX = PopObjX(Remain,:);
end
```

### `UpdateArchive.m`
```matlab
function [Archive,ArcW,ArcSP] = UpdateArchive(Archive,ArcObjX,SOI,SOIObjX,alpha,z,W,ArcW,ArcSP)
% Update the archive

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Update the archive
    Tr      = max(sum(abs(SOIObjX-z),2));
    Remain  = sum(abs(ArcObjX-z),2) <= alpha*Tr;
    Archive = [Archive(Remain),SOI];
    
    %% Update other information
    [~,SOIW]    = min(pdist2(SOIObjX-z,W,'cosine'),[],2);
    [~,ArcWNew] = min(pdist2(ArcObjX-z,W,'cosine'),[],2);
    ArcW        = arrayfun(@(i)[ArcW{i},ArcWNew(i)],(1:length(ArcW))','UniformOutput',false);
    ArcW        = [ArcW(Remain);num2cell(SOIW)];
    ArcSPNew    = sum(ArcObjX,2);
    ArcSP       = arrayfun(@(i)[ArcSP{i},ArcSPNew(i)],(1:length(ArcSP))','UniformOutput',false);
    ArcSP       = [ArcSP(Remain);num2cell(sum(SOIObjX,2))];
end
```
