# DEA-GNG

**Tags**: <2020> <multi/many> <real/integer/label/binary/permutation>

## Description
Decomposition based evolutionary algorithm guided by growing neural gas

## Reference
Y. Liu, H. Ishibuchi, N. Masuyama, and Y. Nojima. Adapting reference vectors and scalarizing functions by growing neural gas to handle irregular Pareto fronts. IEEE Transactions on Evolutionary Computation, 2020, 24(3): 439-453.

## Source Code

### `ArchiveUpdate.m`
```matlab
function [Data,nND] = ArchiveUpdate(Data,N,Z1,Z2,Zmin)
% Update Input Signal Archive in DEA-GNG

%--------------------------------------------------------------------------
% Copyright Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    %% Delete duplicate
    Data = unique(Data,'rows');

    %% Non-dominated sorting
    FrontNo = NDSort(Data,N);
    Data = Data((FrontNo == 1)',:);
    nND = size(Data,1);
    
    %% Selection
    if nND > N        
       Choose = LastSelection(Data,N,Z1,Z2,Zmin);
        Data = Data(Choose',:); 
        nND = N;      
    end      
end

function Choose = LastSelection(PopObj,K,Z1,Z2,Zmin)
% Select part of the solutions in the last front

    %% Initialize
    Z = [Z1;Z2];    
    [N,M]  = size(PopObj);
    NZ     = size(Z,1);
    NZ1     = size(Z1,1);
    
    Zmax = max(PopObj,[],1);
    PopObj = (PopObj - repmat(Zmin,N,1))./repmat(Zmax-Zmin,N,1);      
        
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Distance   = pdist2(PopObj,Z,'cosine');  
    % Associate each solution with its nearest reference point
    [d2,pi] = min(Distance',[],1);  
    
    %% Calculate the number of associated solutions of each reference point  
    cn = zeros(1,NZ);
    Choose  = false(1,N);
    Zchoose = true(1,NZ);
    while sum(Choose) < K
        Temp = find(Zchoose);
        Jmin = cn(Temp) == min(cn(Temp));
        Temp = Temp(Jmin);
        Temp1 = Temp(Temp<=NZ1);
        if sum(Temp1) > 0
            j = Temp1(randi(length(Temp1)));
        else
            j = Temp(randi(length(Temp)));
        end     
        I = find(Choose==0 & pi==j);%the solutions relate to j
        % Then delete one solution associated with this reference point
        if ~isempty(I)
            if cn(j) == 0
                [~,s] = min(d2(I));
            elseif cn(j) < M
                [~,s] = max(d2(I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            cn(j) = cn(j) + 1;
        else
            Zchoose(j) = false;
        end
        
    end
end
```

### `DEAGNG.m`
```matlab
classdef DEAGNG < ALGORITHM
% <2020> <multi/many> <real/integer/label/binary/permutation>
% Decomposition based evolutionary algorithm guided by growing neural gas
% aph ---   0.1 --- Parameter alpha
% eps --- 0.314 --- Parameter epsilon

%------------------------------- Reference --------------------------------
% Y. Liu, H. Ishibuchi, N. Masuyama, and Y. Nojima. Adapting reference
% vectors and scalarizing functions by growing neural gas to handle
% irregular Pareto fronts. IEEE Transactions on Evolutionary Computation,
% 2020, 24(3): 439-453.
%--------------------------------------------------------------------------
% Copyright Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [aph,eps] = Algorithm.ParameterSet(0.1,0.314); 
                           
            %% DEA Initialization
            [Ru,Problem.N] = UniformPoint(Problem.N,Problem.M);	% Uniform reference vectors
            Population     = Problem.Initialization();          % Random population
            Zmin           = min(Population.objs,[],1);         % Ideal Point  
            AS             = [];                                % Input Signal Archive
            Ruq            = Ru;                                % Refernce vectors in Ru for selection
            [FrontNo,~]    = NDSort(Population.objs,Problem.N); % Fitness for the first mating selection
            crd            = zeros(1,Problem.N);                % Fitness for the first mating selection   
            MaxGen         = ceil(Problem.maxFE/Problem.N);    	% Maximum Generation
            
            %% GNG Initialization
            ArchiveSize = Problem.M*Problem.N;	% Size of Input Signal Archive
            NoG = aph*MaxGen;                   % Number of generations of Not Training GNG
            GNGnet.maxIter = 1;                 % Number of iterations to train GNG per Generation  
            GNGnet.maxAge = Problem.N;          % Maximum cluster age     
            GNGnet.maxNode = Problem.N;         % Max number of nodes
            GNGnet.lambda = 0.2*Problem.N;      % Cycle for topology reconstruction   
            GNGnet.hp = [];                     % Hit point of node 
            GNGnet.maxHP = 2*ArchiveSize;       % Max HP of node 
            GNGnet.Node = [];                   % Node
            GNGnet.NodeS = [];                  % Expanded node 
            GNGnet.NodeP = [];                  % Node mapped to hyperplane 
            GNGnet.Err = [];                    % Error 
            GNGnet.edge = zeros(2,2);           % Edge between nodes 
            GNGnet.age = zeros(2,2);            % Age of edge 
            GNGnet.epsilon_a = 0.2;             % Learning coefficient
            GNGnet.epsilon_nb = 0.01;           % Learning coefficient of neighbor
            GNGnet.alpha = 0.5;                 % Nodes r1max and r2max error reduction constant
            GNGnet.delta = 0.9;                 % Error reduction coefficient 

            %% Optimization
            while Algorithm.NotTerminated(Population)        
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,crd);
                Offspring  = OperatorGA(Problem,Population(MatingPool));       
                Zmin       = min([Zmin;Offspring.objs],[],1);

                %% GNG-based adaptation
                if ceil(Problem.FE/Problem.N) <= MaxGen - NoG   
                    % Input Signal Archive Update
                    AS = ArchiveUpdate([AS;Offspring.objs],ArchiveSize,Ruq,GNGnet.NodeS,Zmin);
                    nAS = length(AS);      
                    % GNG Update (and Algorithm 3)
                    GNGnet.maxNode = min(Problem.N,floor(nAS/2)); % paramter reset 
                    GNGnet.maxHP = 2*nAS; % paramter reset
                    GNGnet = GNGUpdate(AS,GNGnet);
                    % Reference Vector Adaptation (Algorithm 4) 
                    if size(GNGnet.NodeS,1)>2
                        [Ruq,GNGnet] = ReferenceCombination(Ru,GNGnet);
                    end
                    % Scalarizing Function Adaptation
                    theta = TunePBI(GNGnet,eps); % Tune theta in PBI function                    
                end

               %% Environmental Selection                 
               [Population,FrontNo,crd] = ESelection([Population,Offspring],Problem.N,Ruq,GNGnet.NodeS,theta,Zmin);
            end
        end
    end
end
```

### `ESelection.m`
```matlab
function [Population,FrontNo,crd] = ESelection(Population,N,Ruq,Rnode,theta,Zmin)
% The environmental selection of DEA-GNG

%--------------------------------------------------------------------------
% Copyright Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;
    F1 = Population(FrontNo==1);
    Zmax = max(F1.objs,[],1); % Nadir Point
    
    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    [Choose,crd] = LastSelection(Population(Next).objs,Population(Last).objs,N-sum(Next),Ruq,Rnode,theta,Zmin,Zmax);
    Next(Last(Choose)) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);        
end

function [Choose, crd] = LastSelection(PopObj1,PopObj2,K,Ruq,Rnode,theta,Zmin,Zmax)
% Select part of the solutions in the last front

    %% Initialize
    PopObj = [PopObj1;PopObj2]; %Candidate solutions
    R = [Ruq;Rnode]; % Reference Vectors   
    [N,~]  = size(PopObj);
    N1     = size(PopObj1,1);
    N2     = size(PopObj2,1);
    NR     = size(R,1);
    NR1    = size(Ruq,1);   
        
    %% Normalization [0-1]
    PopObj = (PopObj - repmat(Zmin,N,1))./repmat(Zmax-Zmin,N,1);
    
    %% Scalarizing function value
    g = zeros(1,N); 
    d1 = zeros(1,N); %d1 in PBI
    theta = [Inf(1,NR1),theta]; %theta in PBI
        
    %% Associate each solution with one reference vector
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,R,'cosine');
    NormP = sqrt(sum(PopObj.^2,2));
    Distance2 = repmat(NormP,1,NR).*sqrt(1-Cosine.^2);    
    % Associate each solution with its nearest reference vector
    [d2,pi] = min(Distance2',[],1);  
    
    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = hist(pi(1:N1),1:NR);
       
    %% Calculate scalarizing functions (PBI)
    for i = N1+1:N
        d1(i) = NormP(i).*Cosine(i,pi(i));
        if theta(pi(i))==Inf
            g(i) = d2(i);
        else         
            g(i) = d1(i)+theta(pi(i)).*d2(i); 
        end    
    end
       
    %% Select solutions
    Choose  = false(1,N2);
    Zchoose = true(1,NR);
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);      
        Jmin = find(rho(Temp)==min(rho(Temp)));
        j = Temp(Jmin(randi(length(Jmin)))); 
        I = find(Choose==0 & pi(N1+1:end)==j);       
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(g(N1+I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end   
    crd = rho(pi);
    crd = [crd(1:N1),crd(N1+find(Choose))];   
end
```

### `GNGUpdate.m`
```matlab
function net = GNGUpdate(oSignals,net)
% Update GNG

%--------------------------------------------------------------------------
% Copyright Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------
    
    %% Parameters 
    [nSig,D]=size(oSignals);
    maxAge = net.maxAge; 
    lambda = net.lambda;   
    epsilon_a = net.epsilon_a;    
    epsilon_nb = net.epsilon_nb;   
    alpha = net.alpha;           
    delta = net.delta;           
    maxHP = net.maxHP;
    maxIter = net.maxIter;
    Node = net.Node;
    Err  = net.Err;   
    edge = net.edge;  
    age  = net.age;      
    hp  = net.hp;

    
    %% Randamize Input Signal
    ran = randperm(nSig);
    oSignals = oSignals(ran,:);   
   
    %% Input Signal Normalization [0-1]
    oMax = max(oSignals,[],1);
    oMin = min(oSignals,[],1);
    oRange = oMax-oMin;
    Signals = (oSignals - repmat(oMin,nSig,1))./repmat(oRange,nSig,1);

    %% Update GNG   
    for nitr = 1:maxIter      
        % Step 0: Initialization. Start with two neural units (nodes) selected from input data
        if size(Node,1) <= 2
            Ni = 2;
            Xmin = min(Signals,[],1);
            Xmax = max(Signals,[],1);
            for i = 1:Ni
                Node(i,:) = unifrnd(Xmin, Xmax);
            end     
            Err = [0; 0];
            edge = zeros(2,2);  
            age  = zeros(2,2);          
            hp = ones(1,2).*maxHP;
        end
                       
        for numSig = 1:nSig

            % Step 1: Input one signal
            pattern = Signals(numSig,:);

            % Step 2: Find the two nearest nodes ra and rb to new signal
            d = pdist2(pattern, Node);
            [~, SortOrder] = sort(d);
            ra = SortOrder(1);
            rb = SortOrder(2);

            % Step 2.5: change HP
            hp = hp-1;
            hp(ra) = maxHP;
            hp(rb) = hp(rb)+1;

            % Steps 3: Increment the age of all edges emanating from ra
            age(ra, :) = age(ra, :) + 1;
            age(:, ra) = age(:, ra) + 1;

            % Step 4: Add the squared distance to a local error counter variable
            Err(ra) = Err(ra) + d(ra)^2;    

            % Step 5: Move ra and its topological neighbors towards singal         
            Node(ra,:) = Node(ra,:) + epsilon_a * (pattern - Node(ra,:));
            for j = find(edge(ra,:)==1)
                Node(j,:) = Node(j,:) + epsilon_nb * (pattern - Node(j,:));% for Neighbor nodes which are connecting to ra.      
            end

            % Step 6:
            % If ra and rb are connected by an edge, set the age of this edge to zero.
            % If such an edge does not exist, create it.
            edge(ra,rb) = 1;
            edge(rb,ra) = 1;
            age(ra,rb) = 0;
            age(rb,ra) = 0;

            % Step 7(1):
            % Remove edges from node if age>maxAge.
            edge(age>maxAge) = 0;

            % Step 7(2):
            % Remove dead node and their edges.       
            DeadNodes = (hp<=0);
            edge(DeadNodes, :) = [];
            edge(:, DeadNodes) = [];
            age(DeadNodes, :) = [];
            age(:, DeadNodes) = [];
            Node(DeadNodes, :) = [];
            Err(DeadNodes) = [];
            hp(DeadNodes) = [];

            % Step 8: Node Insertion Procedure.
            % rnew: new node
            % r1max: node which has maximum accumulated error
            % r2max: neighbor node of r1max
            if mod(numSig, lambda) == 0 && net.maxNode > size(Node,1)
                [~, r1max] = max(Err);
                [~, r2max] = max(edge(:,r1max).*Err);
                rnew = size(Node,1) + 1;
                Node(rnew,:) = (Node(r1max,:) + Node(r2max,:))/2;   
                edge(r1max,r2max) = 0;  
                edge(r2max,r1max) = 0;
                edge(r1max,rnew) = 1;  
                edge(rnew,r1max) = 1;
                edge(rnew,r2max) = 1;  
                edge(r2max,rnew) = 1;
                age(rnew,:) = 0;   
                age(:,rnew) = 0;
                Err(r1max) = alpha * Err(r1max); 
                Err(r2max) = alpha * Err(r2max);
                Err(rnew) = Err(r1max);  
                hp(rnew) = maxHP;
            end

            % Step 9: Decrease the error of all units.
            Err = delta*Err;

        end     
    end     
    
    %% Output net
    net.Node = Node;
    net.Err = Err;
    net.edge = edge;
    net.age = age;
    net.hp = hp;
    
   
    %% Expansion (Algorithm 3)
    % Accociate signal to its closest node
    Distance = pdist2(Node,Signals);
    [~,pi] = min(Distance,[],1); 
    % Node Label based on edge (detect sub-network)
    connection = graph(edge ~= 0);
    NetLabel = conncomp(connection);
    % Data Label based on Node 
    DataLable = NetLabel(pi);
    % Expand each sub-network 
    for i=1:max(NetLabel)
        subData = find(DataLable == i);
        subNet = find(NetLabel == i);       
        if length(subData)<=1 || length(subNet)<=1
            continue;
        end        
        DataMax = max(Signals(subData,:),[],1);
        DataMin = min(Signals(subData,:),[],1);
        NetMax = max(Node(subNet,:),[],1);
        NetMin = min(Node(subNet,:),[],1);
        DataRange = DataMax-DataMin;
        NetRange = NetMax-NetMin;
        Ratio = DataRange./NetRange;
        for k = 1:D
            for j = subNet
                Node(j,k)= (Node(j,k)- NetMin(k)).*Ratio(k)+DataMin(k);
            end
        end
    end
    net.NodeS = Node;   
end
```

### `ReferenceCombination.m`
```matlab
function [Ruq,net] = ReferenceCombination(Ru,net)
% Combine uniform reference vectors and nodes in GNG (Algorihtm 4)

%--------------------------------------------------------------------------
% Copyright Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------

    numNode = size(net.NodeS,1);    
    net.NodeP = net.NodeS;
    
    %% Map nodes to hyperplane
    for i = 1:numNode
       x = 1./sum(net.NodeS(i,:));
       net.NodeP(i,:) = net.NodeS(i,:)*x; 
    end    
    
    %% Average distance ammong nodes
    Distance1 = pdist2(net.NodeP,net.NodeP).*net.edge;   
    AvgDis = sum(Distance1(:))./sum(net.edge(:)); 
    
    %% Min distance among Ru
    Distance2 = pdist2(Ru,Ru);
    Distance2(Distance2 == 0) = 1;   
    MinDis = min(min(Distance2));  
  
    %% Choose the smaller one
    AvgDis = min(AvgDis,MinDis);

    %% Remove some reference vectors in Ru which are too close to nodes
    Distance3 = pdist2(Ru,net.NodeP);      
    Choose = all(Distance3 > AvgDis,2)==1;   
    Ruq = Ru(Choose,:);   
end
```

### `TunePBI.m`
```matlab
function theta = TunePBI(net,eps)
% Adapt scalarizing functions by tune theta in PBI

%--------------------------------------------------------------------------
% Copyright Yiping Liu
% Please contact {yiping0liu@gmail.com} if you have any problem.
%--------------------------------------------------------------------------
 
    N = size(net.NodeS,1);
    theta=zeros(1,N);
    for i=1:N
        Neighbor = find(net.edge(i,:)==1);
        N1 = length(Neighbor);
        if N1==0
            theta(i)=Inf;
            continue;
        end
        EdgeVector = repmat(net.NodeS(i,:),N1,1)-net.NodeS(Neighbor,:);      
        flag=0;
        for j=1:N1
            if all(EdgeVector(j,:)==0)
                flag=1;
                break;
            end
        end
        if flag==1
            theta(i)=Inf;
            continue; 
        end
        Cosine = 1 - pdist2(net.NodeS(i,:),EdgeVector,'cosine');
        Cosine = max(Cosine);
        Angle = acos(Cosine);
        Angle =  Angle - eps;
        if Angle<0
            Angle=0;
        end
        theta(i)=1./tan(Angle);
        if theta(i)<0
            theta(i)=0;
        end
    end   
end
```
