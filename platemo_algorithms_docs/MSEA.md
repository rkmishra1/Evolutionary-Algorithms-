# MSEA

**Tags**: <2021> <multi> <real/integer/label/binary/permutation>

## Description
Multi-stage multi-objective evolutionary algorithm

## Reference
Y. Tian, C. He, R. Cheng, and X. Zhang. A multi-stage evolutionary algorithm for better diversity preservation in multi-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2021, 51(9): 5880-5894.

## Source Code

### `MSEA.m`
```matlab
classdef MSEA < ALGORITHM
% <2021> <multi> <real/integer/label/binary/permutation>
% Multi-stage multi-objective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, C. He, R. Cheng, and X. Zhang. A multi-stage evolutionary
% algorithm for better diversity preservation in multi-objective
% optimization. IEEE Transactions on Systems, Man, and Cybernetics:
% Systems, 2021, 51(9): 5880-5894.
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
            %% Generate random population
            Population = Problem.Initialization();
            FrontNo    = NDSort(Population.objs,inf);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Normalize the population
                PopObj = Population.objs;
                fmax   = max(PopObj(FrontNo==1,:),[],1);
                fmin   = min(PopObj(FrontNo==1,:),[],1);
                PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);

                % Calculate the distance between each two solutions
                Distance = pdist2(PopObj,PopObj);
                Distance(logical(eye(length(Distance)))) = inf;

                % Local search
                for i = 1 : Problem.N
                    % Determining the stage
                    sDis = sort(Distance,2);
                    Div  = sDis(:,1) + 0.01*sDis(:,2);
                    if max(FrontNo) > 1
                        stage = 1;
                    elseif min(Div) < max(Div)/2
                        stage = 2;
                    else
                        stage = 3;
                    end

                    % Generate an offspring
                    switch stage
                        case 1
                            MatingPool = TournamentSelection(2,2,FrontNo,sum(PopObj,2));
                        case 2
                            [~,MatingPool(1)] = max(Div);
                            MatingPool(2)     = TournamentSelection(2,1,-Div);
                        otherwise
                            MatingPool(1) = TournamentSelection(2,1,sum(PopObj,2));
                            MatingPool(2) = TournamentSelection(2,1,-Div);
                    end
                    Offspring = OperatorGAhalf(Problem,Population(MatingPool));
                    OffObj    = (Offspring.obj-fmin)./(fmax-fmin);

                    % Non-dominated sorting
                    NewFront = UpdateFront([PopObj;OffObj],FrontNo);
                    if NewFront(end) > 1
                        continue;
                    end

                    % Calculate the distances
                    OffDis = pdist2(OffObj,PopObj);

                    % Determining the stage
                    if max(NewFront) > 1
                        stage = 1;
                    elseif min(Div) < max(Div)/2
                        stage = 2;
                    else
                        stage = 3;
                    end

                    % Update the population
                    replace = false;
                    switch stage
                        case 1
                            Worse = find(NewFront==max(NewFront));
                            [~,q] = max(sum(PopObj(Worse,:),2));
                            q     = Worse(q);
                            OffDis(q) = inf;
                            replace   = true;
                        case 2
                            [~,q]     = min(Div);
                            OffDis(q) = inf;
                            sODis     = sort(OffDis);
                            ODiv      = sODis(1) + 0.01*sODis(2);
                            if ODiv >= Div(q)
                                replace = true;
                            end
                        otherwise
                            [~,q]     = min(OffDis);
                            OffDis(q) = inf;
                            sODis     = sort(OffDis);
                            ODiv      = sODis(1) + 0.01*sODis(2);
                            if sum(OffObj) <= sum(PopObj(q,:)) && ODiv >= Div(q)
                                replace = true;
                            end
                    end
                    if replace
                        % Update the front numbers
                        FrontNo = UpdateFront([PopObj;OffObj],NewFront,q);
                        FrontNo = [FrontNo(1:q-1),FrontNo(end),FrontNo(q:end-1)];
                        % Update the population
                        Population(q) = Offspring;
                        PopObj(q,:)   = OffObj;
                        % Update the distances
                        Distance(q,:) = OffDis;
                        Distance(:,q) = OffDis';
                    end
                end
            end
        end
    end
end
```

### `UpdateFront.m`
```matlab
function FrontNo = UpdateFront(PopObj,FrontNo,x)
% Update the front number of each solution when a solution is added or
% deleted

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    if nargin < 3
        %% Add a new solution (has been stored in the last of PopObj)
        FrontNo  = [FrontNo,0];
        Move     = false(1,N);
        Move(N)  = true;
        CurrentF = 1;
        % Locate the front No. of the new solution
        while true
            Dominated = false;
            for i = 1 : N-1
                if FrontNo(i) == CurrentF
                    m = 1;
                    while m <= M && PopObj(i,m) <= PopObj(end,m)
                        m = m + 1;
                    end
                    Dominated = m > M;
                    if Dominated
                        break;
                    end
                end
            end
            if ~Dominated
                break;
            else
                CurrentF = CurrentF + 1;
            end
        end
        % Move down the dominated solutions front by front
        while any(Move)
            NextMove = false(1,N);
            for i = 1 : N
                if FrontNo(i) == CurrentF
                    Dominated = false;
                    for j = 1 : N
                        if Move(j)
                            m = 1;
                            while m <= M && PopObj(j,m) <= PopObj(i,m)
                                m = m + 1;
                            end
                            Dominated = m > M;
                            if Dominated
                                break;
                            end
                        end
                    end
                    NextMove(i) = Dominated;
                end
            end
            FrontNo(Move) = CurrentF;
            CurrentF      = CurrentF + 1;
            Move          = NextMove;
        end
    else
        %% Delete the x-th solution
        Move     = false(1,N);
        Move(x)  = true;
        CurrentF = FrontNo(x) + 1;
        while any(Move)
            NextMove = false(1,N);
            for i = 1 : N
                if FrontNo(i) == CurrentF
                    Dominated = false;
                    for j = 1 : N
                        if Move(j)
                            m = 1;
                            while m <= M && PopObj(j,m) <= PopObj(i,m)
                                m = m + 1;
                            end
                            Dominated = m > M;
                            if Dominated
                                break;
                            end
                        end
                    end
                    NextMove(i) = Dominated;
                end
            end
            for i = 1 : N
                if NextMove(i)
                    Dominated = false;
                    for j = 1 : N
                        if FrontNo(j) == CurrentF-1 && ~Move(j)
                            m = 1;
                            while m <= M && PopObj(j,m) <= PopObj(i,m)
                                m = m + 1;
                            end
                            Dominated = m > M;
                            if Dominated
                                break;
                            end
                        end
                    end
                    NextMove(i) = ~Dominated;
                end
            end
            FrontNo(Move) = CurrentF - 2;
            CurrentF      = CurrentF + 1;
            Move          = NextMove;
        end
        FrontNo(x) = [];
    end
end
```
