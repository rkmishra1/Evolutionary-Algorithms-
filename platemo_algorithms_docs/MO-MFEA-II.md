# MO-MFEA-II

**Tags**: <2021> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>

## Description
Multi-objective multifactorial evolutionary algorithm II

## Reference
K. K. Bali, A. Gupta, Y. Ong, and P. S. Tan. Cognizant multitasking in multiobjective multifactorial evolution: MO-MFEA-II. IEEE Transactions on Cybernetics, 2021, 51(4): 1784-1796.

## Source Code

### `CreateOff.m`
```matlab
function SubOffspring = CreateOff(Problem,Population,SubPopulation,RMP)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parent selection
    Parent11 = [];
    Parent21 = [];
    Parent12 = [];
    Parent22 = [];
    for i = 1 : floor(length(Population)/2)
        P1  = Population(i);
        P2  = Population(i+floor(length(Population)/2));
        rmp = RMP(P1.dec(end),P2.dec(end));
        if (P1.dec(end) == P2.dec(end)) || (rand<rmp)
            Parent11 = [Parent11,P1];
            Parent21 = [Parent21,P2];
        else
            Parent12 = [Parent12,P1,P2];
            Parent22 = [Parent22,SubPopulation{P1.dec(end)}(randi(end)),SubPopulation{P2.dec(end)}(randi(end))];
        end
    end
    
    %% Offspring generation
    if ~isempty(Parent11)
        Parent1Dec     = Parent11.decs;
        Parent2Dec     = Parent21.decs;
        OffDec1        = OperatorGA(Problem,[Parent1Dec;Parent2Dec]);
        OffDec1(:,end) = [Parent1Dec(:,end);Parent2Dec(:,end)];
    else
        OffDec1 = [];
    end
    if ~isempty(Parent12)
        Parent1Dec     = Parent12.decs;
        Parent2Dec     = Parent22.decs;
        OffDec2        = OperatorGAhalf(Problem,[Parent1Dec;Parent2Dec]);
        OffDec2(:,end) = Parent1Dec(:,end);
    else
        OffDec2 = [];
    end
    SubOffspring = Divide(Problem.Evaluation([OffDec1;OffDec2]),length(Problem.SubD));
end
```

### `Divide.m`
```matlab
function SubPopulation = Divide(Population,SubCount)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopDec = Population.decs;
    skills = PopDec(:,end);
    for i = 1 : SubCount
        SubPopulation{i} = Population(skills==i);
    end
end
```

### `EnviSelect.m`
```matlab
function [Population,FrontNo,CrowdDis] = EnviSelect(Population,N)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

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
end
```

### `MOMFEAII.m`
```matlab
classdef MOMFEAII < ALGORITHM
% <2021> <multi> <real/integer/label/binary/permutation> <constrained/none> <multitask>
% Multi-objective multifactorial evolutionary algorithm II

%------------------------------- Reference --------------------------------
% K. K. Bali, A. Gupta, Y. Ong, and P. S. Tan. Cognizant multitasking in
% multiobjective multifactorial evolution: MO-MFEA-II. IEEE Transactions on
% Cybernetics, 2021, 51(4): 1784-1796.
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
            %% Initialize population
            Population    = Problem.Initialization();
            SubPopulation = Divide(Population,length(Problem.SubD));
            
            %% Optimization
            while Algorithm.NotTerminated([SubPopulation{:}])
                RMP                  = learnRMP(Problem,SubPopulation);
                [SubPopulation,Rank] = Sort(Problem,SubPopulation);
                Population           = [SubPopulation{:}];
                ParentPool           = Population(TournamentSelection(2,length(Population),[Rank{:}]));
                SubOffspring         = CreateOff(Problem,ParentPool,SubPopulation,RMP);
                for i = 1 : length(Problem.SubD)
                    SubPopulation{i} = EnviSelect([SubPopulation{i},SubOffspring{i}],length(SubPopulation{i}));
                end
            end
        end
    end
end
```

### `Sort.m`
```matlab
function [SubPopulation,Rank] = Sort(Problem,SubPopulation)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Rank = {};
    for i = 1 : length(Problem.SubD)
        [~,FrontNo,CrowdDis] = EnviSelect(SubPopulation{i},length(SubPopulation{i}));
        [~,rank]             = sortrows([FrontNo',-CrowdDis']);
        SubPopulation{i}     = SubPopulation{i}(rank);
        Rank{i}              = 1 : length(rank);
    end
end
```

### `learnRMP.m`
```matlab
function RMP = learnRMP(Problem,SubPopulation)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

   Population = [SubPopulation{:}];
    for i = 1 : length(Problem.SubD)
        subpops(i).data = [];
        vars(i)         = Problem.SubD(i);
    end
    for i = 1 : length(Population)
        subpops(Population(i).dec(end)).data = [subpops(Population(i).dec(end)).data;Population(i).dec(1:Problem.SubD(1))];
    end
    RMP = learnRMP_sub(subpops,vars);
end
    
function rmpMatrix = learnRMP_sub(subpops,vars)
% There are two inputs. subpops(i).data corresponds to the population
% corresponding to the ith task; vars(i) is the number of design variables
% of the ith task.
    
    numtasks  = length(subpops);
    maxdims   = max(vars);
    rmpMatrix = eye(numtasks);
    % Add noise and Build probabilistic models
    for i = 1 : numtasks
        probmodel(i).nsamples = size(subpops(i).data,1);
        nrandsamples          = floor(0.1*probmodel(i).nsamples);
        randMat               = rand(nrandsamples,maxdims);
        probmodel(i).mean     = mean([subpops(i).data;randMat]);	% Univariate distribution mean
        probmodel(i).stdev    = std([subpops(i).data;randMat]);     % Univariate distribution standard deviation
    end
    for i = 1 : numtasks
        for j = i+1 : numtasks
            popdata(1).probmatrix = ones(probmodel(i).nsamples,2);
            popdata(2).probmatrix = ones(probmodel(j).nsamples,2);
            dims = min([vars(i),vars(j)]);
            for k = 1 : probmodel(i).nsamples
                for l = 1 : dims
                    popdata(1).probmatrix(k,1) = popdata(1).probmatrix(k,1)*pdf('Normal',subpops(i).data(k,l),probmodel(i).mean(l),probmodel(i).stdev(l));
                    popdata(1).probmatrix(k,2) = popdata(1).probmatrix(k,2)*pdf('Normal',subpops(i).data(k,l),probmodel(j).mean(l),probmodel(j).stdev(l));
                end
            end
            for k = 1 : probmodel(j).nsamples
                for l = 1 : dims
                    popdata(2).probmatrix(k,1) = popdata(2).probmatrix(k,1)*pdf('Normal',subpops(j).data(k,l),probmodel(i).mean(l),probmodel(i).stdev(l));
                    popdata(2).probmatrix(k,2) = popdata(2).probmatrix(k,2)*pdf('Normal',subpops(j).data(k,l),probmodel(j).mean(l),probmodel(j).stdev(l));
                end
            end
            rmpMatrix(i,j) = max([0,fminbnd(@(x)loglik(x,popdata,numtasks),0,1)+normrnd(0,0.01)]);  %fminbnd(@(x)loglik(x,popdata,numtasks),0,1)
            rmpMatrix(i,j) = min(rmpMatrix(i,j),1);
            rmpMatrix(j,i) = rmpMatrix(i,j);
        end
    end
end

function f = loglik(rmp,popdata,ntasks)
    f = 0;
    for i = 1 : 2
        for j = 1 : 2
            if i == j
                popdata(i).probmatrix(:,j) = popdata(i).probmatrix(:,j)*(1-(0.5*(ntasks-1)*rmp/ntasks));
            else
                popdata(i).probmatrix(:,j) = popdata(i).probmatrix(:,j)*0.5*(ntasks-1)*rmp/ntasks;
            end
        end
        f = f + sum(-log(sum(popdata(i).probmatrix,2)));
    end
end
```
