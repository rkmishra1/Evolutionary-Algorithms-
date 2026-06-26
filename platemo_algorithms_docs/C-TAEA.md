# C-TAEA

**Tags**: <2018> <multi/many> <real/integer/label/binary/permutation> <constrained>

## Description
Two-archive evolutionary algorithm for constrained MOPs

## Reference
K. Li, R. Chen, G. Fu, and X. Yao. Two-archive evolutionary algorithm for constrained multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2018, 23(2): 303-315.

## Source Code

### `CTAEA.m`
```matlab
classdef CTAEA < ALGORITHM
% <2018> <multi/many> <real/integer/label/binary/permutation> <constrained>
% Two-archive evolutionary algorithm for constrained MOPs

%------------------------------- Reference --------------------------------
% K. Li, R. Chen, G. Fu, and X. Yao. Two-archive evolutionary algorithm for
% constrained multi-objective optimization. IEEE Transactions on
% Evolutionary Computation, 2018, 23(2): 303-315.
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
            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);

            %% Generate random population
            Population = Problem.Initialization();
            CA = UpdateCA([],Population,W);
            DA = UpdateDA(CA,[],Population,W);

            %% Optimization
            while Algorithm.NotTerminated(CA)
                %% mating pool choosing
                % calculate the ratio of non-dominated solutions of CA and DA in Hm
                Hm = [CA,DA];                         
                [FrontNo,~] = NDSort(Hm.objs,inf);
                FrontNo_C   = FrontNo(1:ceil(length(Hm)/2));
                Nc = size(find(FrontNo_C==1),2);      
                Pc = Nc/length(Hm);
                FrontNo_D = FrontNo(ceil(length(Hm)/2)+1:length(Hm));
                Nd = size(find(FrontNo_D==1),2);      
                Pd = Nd/length(Hm);

                % calculate the proportion of non-dominated solutions in CA
                [FrontNo,~] = NDSort(CA.objs,inf);
                NC = size(find(FrontNo==1),2);         
                PC = NC/length(CA);                     % PC denotes the proportion of non-dominated solutions in CA,it is different from Pc

                %reproduction
                Q = [];
                for i = 1 : size(W,1)
                    if Pc > Pd
                        P1 = MatingSelection(CA); 
                    else
                        P1 = MatingSelection(DA);
                    end
                    pf = rand();
                    if pf < PC
                        P2 = MatingSelection(CA);
                    else
                        P2 = MatingSelection(DA);
                    end
                    MatingPool = [P1,P2];
                    Offspring  = OperatorGAhalf(Problem,MatingPool);
                    Q = [Q,Offspring];
                end

               %% update CA and DA
                CA = UpdateCA(CA,Q,W);
                DA = UpdateDA(CA,DA,Q,W);
            end
        end
    end
end
```

### `MatingSelection.m`
```matlab
function P = MatingSelection(S)
% Mating selection of C-TAEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
  
    number=length(S);
    rnd=randi(number,1,2);
    x_1=rnd(1);
    x_2=rnd(2);
    CV1 = sum(max(0,S(x_1).con),2);
    CV2 = sum(max(0,S(x_2).con),2);
    if CV1>0 && CV2>0
        x=randi(number,1);
        P=S(x);
    elseif CV1<=0 && CV2>0
        P=S(x_1);
    elseif CV1>0  && CV2<=0
        P=S(x_2);
    elseif CV1==0 && CV2==0
          [FrontNo,~] = NDSort(S(rnd).objs,inf);
          if FrontNo(1)<=FrontNo(2)
              P=S(x_1);
          else
              P=S(x_2);
          end
     end
end
```

### `UpdateCA.m`
```matlab
function UpdatedCA=UpdateCA(CA,Q,W)
% Update CA
% CA is the Archive that has not been updated.
% Q is the set of offspring.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    S=[];       % S is the set used for output
    Sc=[];      % Sc is used to collect feasible solutions
    Hc=[CA,Q];
    N=size(W,1);
    CV=sum(max(0,Hc.cons),2);                                    
    Sc=[Sc,Hc(CV==0)];

    if length(Sc)==N
        UpdatedCA=Sc;
    elseif length(Sc)>N
        [FrontNO,MaxNO]=NDSort(Sc.objs,inf);
        for i=1:MaxNO
            S=cat(2,S,Sc(FrontNO==i));
            if length(S)>=N
                break;
            end
        end
        while length(S)>N 
            %normalization
            Zmax=max(S.objs,[],1);
            Zmin=min(S.objs,[],1);
            SPopObj=(S.objs-repmat(Zmin,size(S.objs,1),1))./(repmat(Zmax,size(S.objs,1),1)-repmat(Zmin,size(S.objs,1),1));

            [~,Region] = max(1-pdist2(SPopObj,W,'cosine'),[],2);% associate each solution in S with their corresponding subregion
            [value,~]=sort(Region,'ascend');
            flag=max(value);
            counter=histc(value,1:flag);                        % counter denotes the number of indiviudals in each subregion
            [~,most_crowded]=max(counter);
            S_crowdest=S(Region==most_crowded);                 % S_crowdest is the set of individuals from the most crowded subregion
            dist=pdist2(S_crowdest.objs,S_crowdest.objs);                     
            dist(dist==0)=inf;
            [row,~]=find(min(min(dist))==dist);
            St=S_crowdest(row);                                 % St is the set of individuals having the smallest distance in S_crowdest
            [~,Region_St] = max(1-pdist2(St.objs,W,'cosine'),[],2);
            Z = min(St.objs,[],1);
            g_tch=max(abs(St.objs-repmat(Z,length(St),1))./W(Region_St,:),[],2);
            [~,order]=max(g_tch);
            x_wrost=St(order);
            S=setdiff(S,x_wrost);
        end
        UpdatedCA=S;

    elseif length(Sc)<N
        SI=setdiff(Hc,Sc);	% SI is the set of infeasible solutions in Hc
        f1=sum(max(0,SI.cons),2);
        [~,Region_SI] = max(1-pdist2(SI.objs,W,'cosine'),[],2);
        Z = min(SI.objs,[],1) ;
        f2=max(abs(SI.objs-repmat(Z,length(SI),1))./W(Region_SI,:),[],2);
        PopObj=[f1,f2];           
        [FrontNO,MaxNO]=NDSort(PopObj,inf);                
        S=[S,Sc];
        for i=1:MaxNO
            S=cat(2,S,SI(FrontNO==i));
            if length(S)>=N 
                last=i;
                break;
            end
        end
        F_last=SI(FrontNO==last);	% find the individuals in the last front joined into S
        delete_n=size(S,2)-N;
        CV=sum(max(0,F_last.cons),2);
        [~,index]=sort(CV,'descend');
        x_wrost=F_last(index(1:delete_n));
        S=setdiff(S,x_wrost);
        UpdatedCA=S;
    end
end
```

### `UpdateDA.m`
```matlab
function UpdatedDA=UpdateDA(CA,DA,Q,W)
% Update DA
% CA is the Archive that has  been updated.
% DA is the Archive that has not been updated
% Q  is the set of offspring
% W  is the set of weight vectors

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    S=[];                                                       % S is the set used for output
    Hd=[DA,Q];
    N=size(W,1);
    [~,Region_Hd] = max(1-pdist2(real(Hd.objs),W,'cosine'),[],2);
    [~,Region_CA] = max(1-pdist2(CA.objs,W,'cosine'),[],2);     % associat the individuals in Hd and CA with their corresponding subregions
    itr=1;
    while length(S)<N
        for i=1:N                                               % here i denotes the order of the subregion
            current_c=find(Region_CA==i);                       % current_c denotes that the current_c_th individual is/are already in the ith region
            if length(current_c)<itr 
                for j=1:itr-length(current_c)                   % j denotes the number of solutions from Hd that need to join into the region(i)
                    current_d=find(Region_Hd==i);
                    if isempty(current_d)~=1
                        [FrontNO,~]=NDSort(Hd(current_d).objs,inf);
                        O=Hd(current_d(FrontNO==1));            % O is the set of nondominated solutions from region(i) in Hd
                        [~,Region_O] = max(1-pdist2(real(O.objs),W,'cosine'),[],2);
                        Z = min(O.objs,[],1);
                        g_tch=max(abs(O.objs-repmat(Z,length(O),1))./W(Region_O,:),[],2);
                        [~,order]=min(g_tch);
                        x_best=O(order); 

                        Hd(current_d(Hd(current_d)==x_best))=[];% update Region_Hd
                        if isempty(Hd)==1
                            Region_Hd=[];
                        else
                            [~,Region_Hd] = max(1-pdist2(real(Hd.objs),W,'cosine'),[],2);
                        end    
                        if length(S)<N                          % add the best individual into S
                            S=cat(2,S,x_best);
                        end
                    else
                        break;
                    end
               end
            end
            if length(S)==N
                break;
            end
        end
        itr=itr+1;
    end
    UpdatedDA=S;
end
```
