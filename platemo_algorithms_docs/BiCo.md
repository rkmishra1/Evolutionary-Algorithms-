# BiCo

**Tags**: <2022> <multi> <real/integer/label/binary/permutation> <constrained>

## Description
Bidirectional coevolution constrained multiobjective evolutionary algorithm

## Reference
Z. Liu, B. Wang, and K. Tang. Handling constrained multiobjective optimization problems via bidirectional coevolution. IEEE Transactions on Cybernetics, 2022, 52(10): 10163-10176.

## Source Code

### `BiCo.m`
```matlab
classdef BiCo < ALGORITHM
% <2022> <multi> <real/integer/label/binary/permutation> <constrained>
% Bidirectional coevolution constrained multiobjective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Z. Liu, B. Wang, and K. Tang. Handling constrained multiobjective
% optimization problems via bidirectional coevolution. IEEE Transactions on
% Cybernetics, 2022, 52(10): 10163-10176.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
       function main(Algorithm,Problem)
           %% Generate random population
           Population = Problem.Initialization();
           ArcPop     = [];
           
           %% Optimization
           while Algorithm.NotTerminated(Population)
               AllPop     = [Population,ArcPop];
               MatingPool = MatingSelection(Population,ArcPop,Problem.N);
               Offspring  = OperatorGA(Problem,MatingPool(1:Problem.N));
               ArcPop     = UpdateArc([AllPop,Offspring],Problem.N);
               Population = EnvironmentalSelection([Population,Offspring],Problem.N);
           end
       end
    end
end
```

### `EnvironmentalSelection.m`
```matlab
function Population = EnvironmentalSelection(Population,N)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front based on their crowding distances
    Last = find(FrontNo==MaxFNo);
    if sum(Next)+size(Last,2)-N == 0
    	Next(Last)=1;
    else
        Del  = Truncation(Population,Last,sum(Next)+size(Last,2)-N);
        Next(Last(~Del)) = true; 
    end

    %% Population for next generation
    Population = Population(Next);
end


function Del = Truncation(Population,Last,K)
% Select part of the solutions by truncation  

    %% Truncation
    Zmin   = min(Population.objs,[],1);
    PopObj = (Population.objs-repmat(Zmin,length(Population.objs),1))./(repmat(max(Population.objs),length(Population.objs),1)-repmat(Zmin,length(Population.objs),1)+1e-10)+1e-10;
    PopObj = (PopObj(Last,:)); 
    
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

### `MatingSelection.m`
```matlab
function MatingPool = MatingSelection(Population,ArcPop,N)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    MatingPool = [];
    if length(ArcPop) < N
        SelectedIndex = TournamentSelection(2,N,-sum(max(0,Population.cons),2));
        MatingPool    = Population(SelectedIndex);
    else
        AllPop = [Population,ArcPop]; 
        Zmin   = min(AllPop.objs,[],1);
        PopObj = (AllPop.objs-repmat(Zmin,length(AllPop.objs),1))./(repmat(max(AllPop.objs),length(AllPop.objs),1)-repmat(Zmin,length(AllPop.objs),1)+1e-10)+1e-10;
        Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
        Cosine = Cosine.*(1-eye(size(PopObj,1)));

        Temp     = sort(-Cosine,2);
        [~,Rank] = sortrows(Temp);

        CV1 = sum(max(0,Population.cons),2);
        CV2 = sum(max(0,ArcPop.cons),2);

        Angle1 = Rank(1:N);
        Angle2 = Rank(N+1:length(AllPop));

        Index1 = randi(N,1,N);
        Index2 = randi(length(ArcPop),1,N);

        i = 0;
        while length(MatingPool)< N  
            if CV1(Index1(i+1))< CV2(Index2(i+1))     
                MatingPool = [MatingPool,Population(Index1(i+1))];
            else
                MatingPool = [MatingPool,ArcPop(Index2(i+1))];
            end
            if Angle1(Index1(i+2))< Angle2(Index2(i+2))
                MatingPool = [MatingPool,Population(Index1(i+2))];
            else
                MatingPool = [MatingPool,ArcPop(Index2(i+2))];
            end    
            i = i + 2 ;
        end
    end
end
```

### `UpdateArc.m`
```matlab
function ArcPop = UpdateArc(Population,N)

%--------------------------------------------------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    FrontNo    = NDSort([Population.objs,sum(max(0,Population.cons),2)],1);
    Population = Population(FrontNo==1);
    Population = Population(sum(max(0,Population.cons),2)>0);

    if length(Population)<N 
        ArcPop = Population;
    else
        Zmax = max(Population.objs,[],1);
        Next(1:size(Population,2)) = true;
        % Select the solutions in the last front
        Delete = LastSelection(Population(Next).objs,sum(max(0,Population.cons),2),sum(Next)-N,Zmax);
        Temp = find(Next);
        Next(Temp(Delete)) = false;
        ArcPop = Population(Next);
    end
end


function Delete = LastSelection(PopObj,PopCons,K,Zmax)
% Select part of the solutions in the last front

    N      = size(PopObj,1);
    PopObj = (PopObj-repmat(Zmax,N,1))./(repmat(min(PopObj),N,1)-repmat(Zmax,N,1)- 1e-10);
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    Cosine = Cosine.*(1-eye(size(PopObj,1)));

    %% Environmental selection
    Delete = false(1,N);
    % Select K solutions one by one
    while sum(Delete) < K
        [Jmin_row,Jmin_column] = find(Cosine==max(max(Cosine)));
        j      = randi(length(Jmin_row));
        Temp_1 = Jmin_row(j);
        Temp_2 = Jmin_column(j);
        if (PopCons(Temp_1)>PopCons(Temp_2)) || (PopCons(Temp_1)==PopCons(Temp_2) && rand<0.5)
            Delete(Temp_1)   = true;
            Cosine(:,Temp_1) = 0;
            Cosine(Temp_1,:) = 0;
        else
            Delete(Temp_2)   = true;
            Cosine(:,Temp_2) = 0;
            Cosine(Temp_2,:) = 0;
        end
    end
end
```
