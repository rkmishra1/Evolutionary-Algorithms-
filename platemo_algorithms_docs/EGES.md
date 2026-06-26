# EGES

**Tags**: <2025> <multi/many> <real> <large> <expensive>

## Description
Efficient grouping evolutionary search

## Reference
H. Zhen, W. Gong, L. Wang, and X. Hu. Surrogate-assisted efficient grouping evolutionary search for expensive large-scale multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2025.

## Source Code

### `CI.m`
```matlab
function CI = CI(ND_PopObj,TSObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    % Non-dominated sorting of archived data
    [FrontNo,~] = NDSort(TSObj,inf);
    ND_TSObj    = TSObj(FrontNo==1,:);
    
    % angle 
    angle     = acos(1-pdist2((max(ND_PopObj,0)),max(ND_TSObj,0),'cosine'));   % Calculate the angle between the two sets of vectors
    [Angle,~] = min(angle,[],2); % Calculate the minimum angle between each candidate solution and the ND sample

    % Objective Normalization
    ND_PopObj = (ND_PopObj - min(TSObj,[],1))./(max(TSObj,[],1)-min(TSObj,[],1));
    TSObj     = (TSObj- min(TSObj,[],1))./(max(TSObj,[],1)-min(TSObj,[],1));
    Z         = min(TSObj,[],1);

    % CN
    ddt         = pdist2(ND_PopObj, TSObj,'euclidean');
    ddt(ddt==0) = inf;
    CN          = min(ddt,[],2);

    % DC
    DC = pdist2(ND_PopObj, Z,'euclidean');

    % Indicators Normalization
    N_angle = (Angle - min(Angle,[],1))./(max(Angle,[],1)-min(Angle,[],1));
    N_CN    = (CN - min(CN,[],1))./(max(CN,[],1)-min(CN,[],1));
    N_DC    = (DC - min(DC,[],1))./(max(DC,[],1)-min(DC,[],1));
    
    % CI
    r1 = rand;
    r2 = rand;
    r3 = rand;
    CI = r1*N_CN + r2*N_angle - r3*N_DC;
end
```

### `CIS.m`
```matlab
function Popreal = CIS(PopDec,PopObj,TSDec,TSObj)
% PopDec: candidate solution data; TSDec: archived data

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

     %% Non-dominated sorting of archived data
     [FrontNo,~] = NDSort(TSObj,inf);
     ND_TSObj    = TSObj(FrontNo==1,:);
     ND_TSDec    = TSDec(FrontNo==1,:);
     
     %% Non-dominated sorting of candidate solution data
     [FrontNo1,~] = NDSort(PopObj,inf);
     ND_PopObj    = PopObj(FrontNo1==1,:);
     ND_PopDec    = PopDec(FrontNo1==1,:);   
     
     %% parameters
     Popreal = [];
     rand1   = 1;

     %% CI sampling
     if rand < rand1
         if ~isempty(PopDec)
             CI_predicted = CI(ND_PopObj,TSObj);
             [~, index]   = max(CI_predicted);
             Popreal      = [Popreal; PopDec(index,:)];
         end
     end
end
```

### `CreateGroups.m`
```matlab
function [outIndexArray,numberOfGroupsArray] = CreateGroups(numberOfGroups, xPrime, numberOfVariables, method)
% ----------------------------------------------------------------------- 
%  Copyright (C) 2020 Heiner Zille
%
%  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
%  International License. (CC BY-NC-SA 4.0). To view a copy of this license, 
%  visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or see the 
%  pdf-file "License-CC-BY-NC-SA-4.0.pdf" that came with this code. 
%
%  You are free to: 
%  * Share ? copy and redistribute the material in any medium or format
%  * Adapt ? remix, transform, and build upon the material 
%  Under the following terms:
%  * Attribution ? You must give appropriate credit, provide a link to the 
%     license, and indicate if changes were made. You may do so in any reasonable 
%     manner, but not in any way that suggests the licensor endorses you or your use.
%  * NonCommercial ? You may not use the material for commercial purposes.
%  * ShareAlike ? If you remix, transform, or build upon the material, you must 
%    distribute your contributions under the same license as the original.
%  * No additional restrictions ? You may not apply legal terms or technological 
%    measures that legally restrict others from doing anything the license permits.
% 
%  Author of this Code: 
%   Heiner Zille <heiner.zille@ovgu.de> or <heiner.zille@gmail.com>
%
%  This code is based on the following publications:
%
%  1) Heiner Zille 
%     "Large-scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art"  
%     PhD Thesis, Otto von Guericke University Magdeburg, 2019 
%     http://dx.doi.org/10.25673/32063 
% 
%  2) Heiner Zille, Hisao Ishibuchi, Sanaz Mostaghim and Yusuke Nojima
%     "Mutation Operators Based on Variable Grouping for Multi-objective Large-scale Optimization"
%     IEEE Symposium Series on Computational Intelligence (SSCI), IEEE, Athens, Greece, December 2016
%     https://ieeexplore.ieee.org/document/7850214 
%
%  This file is intended to work with the PlatEMO framework version 2.5. 
%  Date of publication of this code: 06.04.2020 
%  Last Update of this code: 06.04.2020
%  A newer version of this algorithm may be available. Please contact the author 
%  or see http://www.ci.ovgu.de/Research/Codes.html. 
%
% The files may have been modified in Feb 2021 by the authors of the Platemo framework to work with the Platemo 3.0 release. 
% -----------------------------------------------------------------------

% works for variables only, no INDIVIDUALS needed. 
% works also for arrays of solution variables. 

    outIndexArray       = [];
    numberOfGroupsArray = [];
    
    noOfSolutions = size(xPrime,1);
    for sol = 1 : noOfSolutions % Group the values ​​of the dimensions in each individual, and then randomly select an individual variable as the group number
        switch method
            case 1 %linear grouping
                varsPerGroup = floor(numberOfVariables/numberOfGroups);
                outIndexList = [];
                for i = 1 : numberOfGroups-1
                    outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
                end
                outIndexList = [outIndexList, ones(1,numberOfVariables-size(outIndexList,2)).*numberOfGroups];
            case 2 %orderByValueGrouping
                varsPerGroup = floor(numberOfVariables/numberOfGroups);
                vars  = xPrime(sol,:);
                [~,I] = sort(vars);
                outIndexList = ones(1,numberOfVariables);
                for i = 1 : numberOfGroups-1
                    outIndexList(I(((i-1)*varsPerGroup)+1:i*varsPerGroup)) = i;
                end
                outIndexList(I(((numberOfGroups-1)*varsPerGroup)+1:end)) = numberOfGroups;
            case 3 %random Grouping
                varsPerGroup = floor(numberOfVariables/numberOfGroups);
                outIndexList = [];
                for i = 1 : numberOfGroups-1
                    outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
                end
                outIndexList = [outIndexList, ones(1,numberOfVariables-size(outIndexList,2)).*numberOfGroups];
                outIndexList = outIndexList(randperm(length(outIndexList)));
        end
        outIndexArray       = [outIndexArray;outIndexList];
        numberOfGroupsArray = [numberOfGroupsArray;numberOfGroups];
    end
end
```

### `EGES.m`
```matlab
classdef EGES < ALGORITHM
% <2025> <multi/many> <real> <large> <expensive>
% Efficient grouping evolutionary search

%------------------------------- Reference --------------------------------
% H. Zhen, W. Gong, L. Wang, and X. Hu. Surrogate-assisted efficient
% grouping evolutionary search for expensive large-scale multi-objective
% optimization. IEEE Transactions on Evolutionary Computation, 2025.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    methods
       function main(Algorithm,Problem)
            %% Population initialization
            Problem.N = 11*Problem.D-1;                                                        % Initial sample size
            if Problem.N >100
                Problem.N = 100;
            end
            P       = UniformPoint(Problem.N,Problem.D,'Latin');                               % LHS initializes the population P, where normalized samples are generated
            Achieve = Problem.Evaluation(repmat(Problem.upper-Problem.lower,Problem.N,1).*P+repmat(Problem.lower,Problem.N,1)); % Population evaluation, get archive Achieve

            %% Optimization
            while Algorithm.NotTerminated(Achieve)
                % Termination
                if length(Achieve) >= Problem.maxFE
                    break;
                end

                % DATA
                DATA  = Achieve;
                TSDec = DATA.decs;
                TSObj = DATA.objs; 
                maxnumData = 300;                                                              
                numTS      = size(TSDec,1);
                if size(TSDec,1)>=maxnumData
                   trainX = TSDec(numTS-maxnumData+1:end,:);
                   trainY = TSObj(numTS-maxnumData+1:end,:);
                else
                   trainX = TSDec;
                   trainY = TSObj;
                end
                [trainX, trainY] = dsmerge(trainX, trainY);                                     % Merge duplicate training points

                % RBF model
                pair   = pdist2(trainX, trainX);                                                % Calculate the distance matrix
                D_max  = max(max(pair, [], 2));                                                 % Find the maximum distance
                spread = D_max * (Problem.D * Problem.N) ^ (-1 / Problem.D);                    % Calculate empirical parameters
                net    = newrbe(transpose(trainX), transpose(trainY), spread);
                Model  = @(x) sim(net,x');                                                      % Build surrogate model 
                
                % GES
                [PopDec,PopObj] = GES(Achieve,Model,Problem);
                
                % CIS
                PopNew = CIS(PopDec,PopObj,Achieve.decs,Achieve.objs);
                PopNew = unique(PopNew, 'rows');

                % Expensive evaluation
                New = [];
                if ~isempty(PopNew)                                                             % Avoid empty sampling and delete duplicate points
                    [~,ib]       = intersect(PopNew,Achieve.decs,'rows');
                    PopNew(ib,:) = [];
                    if ~isempty(PopNew)
                        New = Problem.Evaluation(PopNew);                                       % Evaluated sampling points
                    else
                        [~,ib]       = intersect(PopDec,Achieve.decs,'rows');
                        PopDec(ib,:) = [];
                        [a1, a2]     = size(PopDec);
                        index        = randi(a1);
                        PopNew       = PopDec(index,:);
                        New          = Problem.Evaluation(PopNew);
                    end
                end
                Achieve = cat(2,Achieve,New);                                                   % Update archive
            end
       end
    end
end
```

### `GES.m`
```matlab
function [PopDec,PopObj] = GES(A1,Model,Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    typeOfGroups   = 2;
    numberOfGroups = 4;

    [V, Problem.N] = UniformPoint(Problem.N,Problem.M); % Reference Points
    
    Zmin = min(A1.objs,[],1);
    if size(A1.decs,1) >= Problem.N
        Next = NSGAIIIEnvironmentalSelection(A1,Problem.N,V,Zmin); 
    end
    
    Population = Next;
    Zmin       = min(Population(all(Population.cons<=0,2)).objs,[],1); % Ideal point
    
    w    = 1;
    wmax = 20;
    while w <= wmax
        MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));                        % Tournament Selection
        Offspring  = GESoperating(Problem, Population(MatingPool), numberOfGroups, typeOfGroups, Model, A1);    % GES
        Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);                             % Ideal point
        Population = NSGAIIIEnvironmentalSelection([Population,Offspring],Problem.N,V,Zmin);                % GLMO_NSGAIIIEnvironmentalSelection
        w          = w+1;
    end
    PopDec = Population.decs;
    PopObj = Population.objs;
end
```

### `GESoperating.m`
```matlab
function Offspring = GESoperating(Problem, Parent, numberOfGroups, typeOfGroups, Model, A1)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Huixiang Zhen (email: zhenhuixiang@cug.edu.cn)

    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        Parent    = Parent.decs;
    else
        evaluated = false;
    end
    [N,D]     = size(Parent);
    Parent1   = Parent(randperm(size(Parent, 1)), :);
    Parent2   = Parent(randperm(size(Parent, 1)), :);
    Offspring = Parent;
    C         = size(A1.cons,2);

    %% grouping
    [outIndexList,~] = CreateGroups(numberOfGroups,Parent,D,typeOfGroups);  % Create group
    chosengroups     = randi(numberOfGroups,size(outIndexList,1),1);        % An individual only selects a kind of grouping variables for optimization
    Site             = outIndexList == chosengroups; 

    %% GDE operators
    [CR,F,proM,disM] = deal(0.9,0.5,1,20);
    temp             = Site & rand(N,D) < CR;   % This includes crossover operations
    Offspring(temp)  = Parent(temp) + F*(Parent1(temp)-Parent2(temp));

    %% Linked Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    mu    = rand(N,1);
    mu    = repmat(mu,1,D);
    temp  = Site & mu<=0.5; 
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

    %% Evaluation
    if evaluated
        PopObj    = Model(Offspring)';
        PopDec    = Offspring;
        PopCon    = zeros(Problem.N,C);
        PopAdd    = zeros(Problem.N,1);
        Offspring = SOLUTION(PopDec,PopObj,PopCon,PopAdd);
    end
end
```

### `NSGAIIIEnvironmentalSelection.m`
```matlab
function Population = NSGAIIIEnvironmentalSelection(Population,N,Z,Zmin)
% The environmental selection of NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = LastSelection(Population(Next).objs,Population(Last).objs,N-sum(Next),Z,Zmin);
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Choose = LastSelection(PopObj1,PopObj2,K,Z,Zmin)
% Select part of the solutions in the last front

    PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
    [N,M]  = size(PopObj);
    N1     = size(PopObj1,1);
    N2     = size(PopObj2,1);
    NZ     = size(Z,1);

    %% Normalization
    % Detect the extreme points
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));
    end
    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes
    Hyperplane = PopObj(Extreme,:)\ones(M,1);
    a = 1./Hyperplane;
    if any(isnan(a))
        a = max(PopObj,[],1)';
    end
    % Normalization
    PopObj = PopObj./repmat(a',N,1);
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    % Associate each solution with its nearest reference point
    [d,pi] = min(Distance',[],1);

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = hist(pi(1:N1),1:NZ);
    
    %% Environmental selection
    Choose  = false(1,N2);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);
        Jmin = find(rho(Temp)==min(rho(Temp)));
        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose==0 & pi(N1+1:end)==j);
        % Then select one solution associated with this reference point
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(d(N1+I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end
```
