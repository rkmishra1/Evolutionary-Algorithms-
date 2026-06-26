# M-PAES

**Tags**: <2000> <multi> <real/integer>

## Description
Memetic algorithm with Pareto archived evolution strategy

## Reference
J. D. Knowles and D. W. Corne. M-PAES: A memetic algorithm for multiobjective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2000, 325-332.

## Source Code

### `GridDensity.m`
```matlab
function varargout = GridDensity(div,varargin)
% Calculate the number of solutions in the grid of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the grid location of each solution
    PopObj = cat(1,varargin{:});
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    d      = (fmax-fmin)/div;
    GLoc   = floor((PopObj-repmat(fmin,size(PopObj,1),1))./repmat(d,size(PopObj,1),1));
    GLoc(GLoc>=div)   = div - 1;
    GLoc(isnan(GLoc)) = 0;
    
    %% Calculate the number of solutions in the grid of each solution
    [~,~,Site] = unique(GLoc,'rows');
    CrowdG     = hist(Site,1:max(Site));
    Crowd      = CrowdG(Site);
    varargout  = mat2cell(Crowd,1,cellfun(@(S)size(S,1),varargin));
end
```

### `MPAES.m`
```matlab
classdef MPAES < ALGORITHM
% <2000> <multi> <real/integer>
% Memetic algorithm with Pareto archived evolution strategy
% l_fails   ---  5 --- Maximum number of consecutive failing local moves
% l_opt     --- 10 --- Maximum number of local moves
% cr_trials --- 20 --- Number of crossover trials
% div       --- 10 --- The number of divisions in each objective

%------------------------------- Reference --------------------------------
% J. D. Knowles and D. W. Corne. M-PAES: A memetic algorithm for
% multiobjective optimization. Proceedings of the IEEE Congress on
% Evolutionary Computation, 2000, 325-332.
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
            [l_fails,l_opt,cr_trials,div] = Algorithm.ParameterSet(5,10,20,10);

            %% Generate random population
            P = Problem.Initialization();
            G = P(NDSort(P.objs,1)==1);

            %% Optimization
            while Algorithm.NotTerminated(G)
                for i = 1 : Problem.N
                    H = G(~all(G.objs<=repmat(P(i).obj,length(G),1),2));
                    [P(i),G] = PAES(Problem,P(i),G,[H,P(i)],Problem.N,l_fails,l_opt,div);
                end
                P1(1:Problem.N) = SOLUTION();
                for i = 1 : Problem.N
                    for r = 1 : cr_trials
                        Combine = [P,G];
                        parents = Combine(randperm(length(Combine),2));
                        c       = OperatorGAhalf(Problem,parents,{1,20,0,0});
                        [G,dominated,GCrowd,cCrowd,pCrowd] = UpdateArchive(G,c,parents,Problem.N,div);
                        if ~dominated && any(cCrowd<=pCrowd)
                            break;
                        end
                    end
                    if dominated
                        c = G(TournamentSelection(2,1,GCrowd));
                    end
                    P1(i) = c;
                end
                P = P1;
            end
        end
    end
end
```

### `PAES.m`
```matlab
function [c,G] = PAES(Problem,c,G,H,N,l_fails,l_opt,div)
% PAES local search

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    fails = 0;
    moves = 0;
    while fails < l_fails && moves < l_opt
        m = OperatorFEP(Problem,c);
        if all(c.obj<=m.obj)
            fails = fails + 1;
        else
            [H,dominated,~,mCrowd,cCrowd] = UpdateArchive(H,m,c,N,div);
            if all(c.obj>=m.obj)
                c = m;
                fails = 0;
            elseif ~dominated && mCrowd < cCrowd
                c = m;
            end
        end
        G     = UpdateArchive(G,m,SOLUTION(),N,div);
        moves = moves + 1;
    end
end
```

### `UpdateArchive.m`
```matlab
function [G,dominated,GCrowd,oCrowd,pCrowd] = UpdateArchive(G,offspring,parents,N,div)
% Update the archive by the offspring, and return whether the offspring is
% dominated by the archive, the crowding degrees of solutions in the
% original archive, the crowding degree of the offspring, and the crowding
% degrees of the parents

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    oobj = repmat(offspring.obj,length(G),1);
    Gobj = G.objs;
    domi = any(oobj<=Gobj,2) - any(oobj>=Gobj,2);
    dominated              = any(domi==-1);
    [GCrowd,oCrowd,pCrowd] = GridDensity(div,Gobj,offspring.obj,parents.objs);
    if any(domi==1)
        G = [G(domi~=1),offspring];
    elseif ~dominated
        if length(G) < N
            G = [G,offspring];
        elseif any(oCrowd<GCrowd)
            worst    = find(GCrowd==max(GCrowd));
            worst    = worst(randi(length(worst)));
            G(worst) = offspring;
        end
    end
end
```
