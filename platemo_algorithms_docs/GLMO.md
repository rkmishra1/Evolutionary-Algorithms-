# GLMO

**Tags**: <2019> <multi> <real/integer> <large/none>

## Description
Grouped and linked mutation operator algorithm

## Reference
H. Zille. Large-scale Multi-objective Optimisation: New Approaches and a Classification of the State-of-the-Art. PhD Thesis, Otto von Guericke University Magdeburg, 2019.

## Source Code

### `GLMO.m`
```matlab
classdef GLMO < ALGORITHM
% <2019> <multi> <real/integer> <large/none>
% Grouped and linked mutation operator algorithm
% optimiser      --- 3 --- The optimisation method. 1 = SMPSO, 2 = NSGA-II, 3 = NSGA-III. Default = NSGA-III
% typeOfGroups   --- 2 --- Grouping method, 1 = linear, 2 = ordered, 3 = random. Default = ordered
% numberOfGroups --- 4 --- The number of varibale Groups. Default = 4  

%------------------------------- Reference --------------------------------
% H. Zille. Large-scale Multi-objective Optimisation: New Approaches and a
% Classification of the State-of-the-Art. PhD Thesis, Otto von Guericke
% University Magdeburg, 2019.
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

    methods
        function main(Algorithm,Problem)
            [optimiser,typeOfGroups,numberOfGroups] = Algorithm.ParameterSet(3,2,4);
            if optimiser == 1
                GLMO_SMPSO(Algorithm,Problem,numberOfGroups,typeOfGroups);
            elseif optimiser == 2
                GLMO_NSGAII(Algorithm,Problem,numberOfGroups,typeOfGroups);
            else
                GLMO_NSGAIII(Algorithm,Problem,numberOfGroups,typeOfGroups);
            end
        end
        function GLMO_SMPSO(Algorithm,Problem,numberOfGroups,typeOfGroups)
            Population       = Problem.Initialization();
            Pbest            = Population;
            [Gbest,CrowdDis] = GLMO_UpdateGbest(Population,Problem.N);
            while Algorithm.NotTerminated(Gbest)
                Population       = GLMO_SMPSOOperator(Problem, Population, Pbest, Gbest(TournamentSelection(2,Problem.N,-CrowdDis)), numberOfGroups, typeOfGroups);
                [Gbest,CrowdDis] = GLMO_UpdateGbest([Gbest,Population],Problem.N);
                Pbest            = GLMO_UpdatePbest(Pbest,Population);
            end
        end
        function GLMO_NSGAII(Algorithm,Problem,numberOfGroups,typeOfGroups)
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = GLMO_NSGAIIEnvironmentalSelection(Population,Problem.N);
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = GLMO_GA(Problem, Population(MatingPool), numberOfGroups, typeOfGroups);
                [Population,FrontNo,CrowdDis] = GLMO_NSGAIIEnvironmentalSelection([Population,Offspring],Problem.N);
            end
        end
        function GLMO_NSGAIII(Algorithm,Problem,numberOfGroups,typeOfGroups)
            [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();
            Zmin          = min(Population(all(Population.cons<=0,2)).objs,[],1);
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));
                Offspring  = GLMO_GA(Problem, Population(MatingPool), numberOfGroups, typeOfGroups);
                Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                Population = GLMO_NSGAIIIEnvironmentalSelection([Population,Offspring],Problem.N,Z,Zmin);
            end
        end
    end
end
```

### `GLMO_GA.m`
```matlab
function Offspring = GLMO_GA(Problem, Parent, numberOfGroups, typeOfGroups)
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
% This file is derived from its original version containied in the PlatEMO 
% framework.
% ----------------------------------------------------------------------- 

    [proC,disC,proM,disM] = deal(1,20,1,20);
    
    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        Parent    = Parent.decs;
    else
        evaluated = false;
    end
    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);

    %% Genetic operators for real encoding
    % Simulated binary crossover
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];

    % Polynomial mutation
    Lower = repmat(Problem.lower,2*N,1);
    Upper = repmat(Problem.upper,2*N,1);

    [outIndexList,~] = GLMO_createGroups(numberOfGroups,Offspring,D,typeOfGroups); 
    chosengroups = randi(numberOfGroups,size(outIndexList,1),1);
    Site = outIndexList == chosengroups;
    mu    = rand(2*N,1);
    mu = repmat(mu,1,D);

    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end
```

### `GLMO_NSGAIIEnvironmentalSelection.m`
```matlab
function [Population,FrontNo,CrowdDis] = GLMO_NSGAIIEnvironmentalSelection(Population,N)
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
% This file is derived from its original version containied in the PlatEMO 
% framework. The original copyright disclaimer can be found below. 
% ----------------------------------------------------------------------- 

% The environmental selection of NSGA-II

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

### `GLMO_NSGAIIIEnvironmentalSelection.m`
```matlab
function Population = GLMO_NSGAIIIEnvironmentalSelection(Population,N,Z,Zmin)
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
% This file is derived from its original version containied in the PlatEMO 
% framework. The original copyright disclaimer can be found below. 
% ----------------------------------------------------------------------- 

% The environmental selection of NSGA-III

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

### `GLMO_SMPSOOperator.m`
```matlab
function Offspring = GLMO_SMPSOOperator(Problem, Particle, Pbest, Gbest, numberOfGroups, typeOfGroups)
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
% This file is derived from its original version containied in the PlatEMO 
% framework. The original copyright disclaimer can be found below. 
% ----------------------------------------------------------------------- 

% Particle swarm optimization in SMPSO

    %% Parameter setting
    ParticleDec = Particle.decs;
    PbestDec    = Pbest.decs;
    GbestDec    = Gbest.decs;
    [N,D]       = size(ParticleDec);
    ParticleVel = Particle.adds(zeros(N,D));

    %% Particle swarm optimization
    W  = repmat(unifrnd(0.1,0.5,N,1),1,D);
    r1 = repmat(rand(N,1),1,D);
    r2 = repmat(rand(N,1),1,D);
    C1 = repmat(unifrnd(1.5,2.5,N,1),1,D);
    C2 = repmat(unifrnd(1.5,2.5,N,1),1,D);
    OffVel = W.*ParticleVel + C1.*r1.*(PbestDec-ParticleDec) + C2.*r2.*(GbestDec-ParticleDec);
    phi    = max(4,C1+C2);
    OffVel = OffVel.*2./abs(2-phi-sqrt(phi.^2-4*phi));
    delta  = repmat((Problem.upper-Problem.lower)/2,N,1);
    OffVel = max(min(OffVel,delta),-delta);
    OffDec = ParticleDec + OffVel;
    
    %% Deterministic back
    Lower  = repmat(Problem.lower,N,1);
    Upper  = repmat(Problem.upper,N,1);
    repair = OffDec < Lower | OffDec > Upper;
    OffVel(repair) = 0.001*OffVel(repair);
    OffDec = max(min(OffDec,Upper),Lower);
    
    %% Polynomial mutation
    disM  = 20;
    Site1 = repmat(rand(N,1)<0.15,1,D);

    [outIndexList,~] = GLMO_createGroups(numberOfGroups,OffDec,D,typeOfGroups); % 3 = random groups
    chosengroups = randi(numberOfGroups,size(outIndexList,1),1);
    Site2 = outIndexList == chosengroups;
    mu    = rand(N,1);
    mu    = repmat(mu,1,D);       
    
    temp  = Site1 & Site2 & mu<=0.5;
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site1 & Site2 & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `GLMO_UpdateGbest.m`
```matlab
function [Gbest,CrowdDis] = GLMO_UpdateGbest(Gbest,N)
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
% This file is derived from its original version containied in the PlatEMO 
% framework. The original copyright disclaimer can be found below. 
% ----------------------------------------------------------------------- 

% Update the global best set

    Gbest    = Gbest(NDSort(Gbest.objs,1)==1);
    CrowdDis = CrowdingDistance(Gbest.objs);
    [~,rank] = sort(CrowdDis,'descend');
    Gbest    = Gbest(rank(1:min(N,length(Gbest))));
    CrowdDis = CrowdDis(rank(1:min(N,length(Gbest))));
end
```

### `GLMO_UpdatePbest.m`
```matlab
function Pbest = GLMO_UpdatePbest(Pbest,Population)
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
% This file is derived from its original version containied in the PlatEMO 
% framework. The original copyright disclaimer can be found below. 
% ----------------------------------------------------------------------- 

% Update the local best position of each particle

    replace        = ~all(Population.objs>=Pbest.objs,2);
    Pbest(replace) = Population(replace);
end
```

### `GLMO_createGroups.m`
```matlab
function [outIndexArray,numberOfGroupsArray] = GLMO_createGroups(numberOfGroups, xPrime, numberOfVariables, method)
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

outIndexArray = [];
numberOfGroupsArray = [];

noOfSolutions = size(xPrime,1);
for sol = 1:noOfSolutions
    
    switch method
        case 1 %linear grouping
            varsPerGroup = floor(numberOfVariables/numberOfGroups);
            outIndexList = [];
            for i = 1:numberOfGroups-1
               outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
            end
            outIndexList = [outIndexList, ones(1,numberOfVariables-size(outIndexList,2)).*numberOfGroups];
        case 2 %orderByValueGrouping
            varsPerGroup = floor(numberOfVariables/numberOfGroups);
            vars = xPrime(sol,:);
            [~,I] = sort(vars);
            outIndexList = ones(1,numberOfVariables);
            for i = 1:numberOfGroups-1
               outIndexList(I(((i-1)*varsPerGroup)+1:i*varsPerGroup)) = i;
            end
            outIndexList(I(((numberOfGroups-1)*varsPerGroup)+1:end)) = numberOfGroups;
        case 3 %random Grouping
            varsPerGroup = floor(numberOfVariables/numberOfGroups);
            outIndexList = [];
            for i = 1:numberOfGroups-1
               outIndexList = [outIndexList, ones(1,varsPerGroup).*i];
            end
            outIndexList = [outIndexList, ones(1,numberOfVariables-size(outIndexList,2)).*numberOfGroups];
            outIndexList = outIndexList(randperm(length(outIndexList)));
    end
    outIndexArray = [outIndexArray;outIndexList];
    numberOfGroupsArray = [numberOfGroupsArray;numberOfGroups];
    
end
end
```
