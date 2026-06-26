# MMOEA-BH

**Tags**: <2025> <multi> <real/integer/label/binary/permutation> <multimodal>

## Description
Multimodal MOEA with block optimization and hybrid clustering

## Reference
Y. Zhang and W. Hu. Block optimization and switchable hybrid clustering for multimodal multiobjective evolutionary optimization with shifted local Pareto front. Swarm and Evolutionary Computation, 2025, 98: 102151.

## Source Code

### `Environmental_Selection.m`
```matlab
function [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection(Method, Problem, Archive, Population, LBA, PBA, n_PBA, varargin)
% Environmental_Selection - Unified entry point for environmental selection strategies.
% Method: 'APC', 'DBSCAN', or 'kmeans'
% varargin: Additional parameters needed by specific methods (e.g., maxCluster, N, epsilon, minpts)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    switch upper(Method)
        case 'APC'
            [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_APC(Problem, Archive, Population, LBA, PBA, n_PBA);
        case 'DBSCAN'
            % DBSCAN requires: N, maxCluster, epsilon(optional), minpts(optional)
            N          = varargin{1};
            maxCluster = varargin{2};
            if length(varargin) >= 4
                epsilon = varargin{3};
                minpts  = varargin{4};
                [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_DBSCAN(Problem, Archive, Population, LBA, PBA, n_PBA, N, maxCluster, epsilon, minpts);
            else
                [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_DBSCAN(Problem, Archive, Population, LBA, PBA, n_PBA, N, maxCluster);
            end
        case 'KMEANS'
            % KMEANS requires: maxCluster
            maxCluster = varargin{1};
            [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_kmeans(Problem, Archive, Population, LBA, PBA, n_PBA, maxCluster);
        otherwise
            error('Unknown Environmental Selection Method: %s', Method);
    end
end

function [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_APC(Problem, Archive, Population, LBA, PBA, n_PBA)
    N         = Problem.N;
    temp_p    = [Population, Archive, LBA{:}];
    temp_decs = temp_p.decs;
    [~,idx]   = unique(temp_decs(:,1));
    Archive   = temp_p(idx);
    
    temp_Archive      = Archive;
    [idx, maxCluster] = MMOEA_Utils.APC(temp_Archive.decs);
    temp_Archive_nd   = [];
    temp_idx_d_nd     = [];
    
    for i = 1 : maxCluster
        indv_t1         = temp_Archive(idx==i);
        num_ND          = NDSort(indv_t1.objs,1)==1;
        temp_Archive_nd = [temp_Archive_nd,indv_t1(num_ND)];
        temp_idx_d_nd   = [temp_idx_d_nd;i*ones(sum(num_ND),1)];
    end
    Archive = temp_Archive_nd;
    idx     = temp_idx_d_nd;
    
    while numel(Archive)>N
        num_cluster = tabulate(idx);
        [~,idx_max] = max(num_cluster(:,2));
        density_y   = MMOEA_Utils.calc_pccs(Archive.objs);
        sel_cluster = find(idx == idx_max);
        density_yt  = density_y(sel_cluster);
        [~,idx1]    = sort(density_yt,'descend');
        try
            idx2 = idx1(1:Problem.M+1);
        catch
            idx2 = idx1;
        end
        density_x  = MMOEA_Utils.calc_pccs(Archive.decs);
        density_xt = density_x(sel_cluster(idx2));
        [~,idx3]   = max(density_xt);
        Archive(sel_cluster(idx2(idx3))) = [];
        idx(sel_cluster(idx2(idx3)))     = [];
    end
    
    a1    = tabulate(idx);
    idx_t = sort(find(a1(:,2)==0),'descend')';
    for i = idx_t
        idx(idx>i,:) = idx(idx>i,:)-1;
    end
    maxCluster = max(idx);
    
    for i = 1 : N
        [~, temp_PBA,temp_PBA_SCD] = MMOEA_Utils.nd_pccs_sort([Population(i), PBA{i}]);
        PBA{i}                     = temp_PBA(1:min(numel(temp_PBA),n_PBA));
        PBA_SCD{i}                 = temp_PBA_SCD(1:min(numel(temp_PBA),n_PBA));
    end
    for i = 1 : maxCluster
        [~, LBA{i}, LBA_SCD{i}] = MMOEA_Utils.nd_pccs_sort(Archive(i==idx),ceil(N/maxCluster));
    end
    LBA(maxCluster+1:end)     = [];
    LBA_SCD(maxCluster+1:end) = [];
end

function [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_DBSCAN(Problem, Archive, Population, LBA, PBA, n_PBA, N, maxCluster, epsilon, minpts)
    if ~exist('epsilon','var'), [epsilon, minpts] = deal(0.15, 5); end
    
    temp_p    = [Population, Archive, LBA{:}];
    temp_decs = temp_p.decs;
    [~,idx]   = unique(temp_decs(:,1));
    Archive   = temp_p(idx);
    
    temp_Archive = Archive;
    idx_1        = dbscan(normalize(temp_Archive.decs, 'range'), epsilon, minpts);
    if ismember(-1, idx_1), maxCluster_1 = numel(unique(idx_1))-1; else, maxCluster_1 = numel(unique(idx_1)); end
    
    idx_2        = dbscan(normalize(temp_Archive.objs, 'range'), epsilon, minpts);
    if ismember(-1, idx_2), maxCluster_2 = numel(unique(idx_2))-1; else, maxCluster_2 = numel(unique(idx_2)); end
    
    temp_num = tabulate(idx_1);
    if ismember(-1, idx_1), temp_num = temp_num(2:end,2); else, temp_num = temp_num(:,2); end
    if max(temp_num)/min(temp_num) > 5, flag_dec = 0; else, flag_dec = 1; end
    
    if maxCluster_1==1 || maxCluster_2==1
        if maxCluster_1==1, flag_dec_one = 1; else, flag_dec_one = 0; end
        if flag_dec_one, [idx, maxCluster_d] = deal(idx_2, maxCluster_2); else, [idx, maxCluster_d] = deal(idx_1, maxCluster_1); end
    else
        if maxCluster_1 >= maxCluster_2 && flag_dec, [idx, maxCluster_d] = deal(idx_1, maxCluster_1); else, [idx, maxCluster_d] = deal(idx_2, maxCluster_2); end
    end
    
    temp_Archive_nd = [];
    for i = -1 : maxCluster_d
        indv_t1         = temp_Archive(idx==i);
        temp_Archive_nd = [temp_Archive_nd,indv_t1(NDSort(indv_t1.objs,1)==1)];
    end
    Archive = temp_Archive_nd;
    
    idx_1 = dbscan(normalize(Archive.decs, 'range'), epsilon, minpts);
    idx_2 = dbscan(normalize(Archive.objs, 'range'), epsilon, minpts);
    if ismember(-1, idx_1), maxCluster_1 = numel(unique(idx_1))-1; else, maxCluster_1 = numel(unique(idx_1)); end
    if ismember(-1, idx_2), maxCluster_2 = numel(unique(idx_2))-1; else, maxCluster_2 = numel(unique(idx_2)); end
    
    if maxCluster_1 == maxCluster_2, idx_del = any([idx_1==-1,idx_2==-1],2); else, idx_del = all([idx_1==-1,idx_2==-1],2); end
    if sum(idx_del==1) == numel(idx_del), idx_del = []; end
    Archive(idx_del) = [];
    
    idx_1 = dbscan(normalize(Archive.decs, 'range'), epsilon, minpts);
    idx_2 = dbscan(normalize(Archive.objs, 'range'), epsilon, minpts);
    if maxCluster_1 >= maxCluster_2, idx = idx_1; else, idx = idx_2; end
    
    while numel(Archive) > N
        num_cluster = tabulate(idx);
        [~,idx_max] = max(num_cluster(:,2));
        idx_max     = num_cluster(idx_max,1);
        density_y   = MMOEA_Utils.calc_pccs(Archive.objs);
        sel_cluster = find(idx == idx_max);
        density_yt  = density_y(sel_cluster);
        [~,idx1]    = sort(density_yt,'descend');
        try
            idx2 = idx1(1:Problem.M+1);
        catch
            idx2 = idx1;
        end
        density_x  = MMOEA_Utils.calc_pccs(Archive.decs);
        density_xt = density_x(sel_cluster(idx2));
        [~,idx3]   = max(density_xt);
        Archive(sel_cluster(idx2(idx3))) = [];
        idx(sel_cluster(idx2(idx3)))     = [];
    end
    
    maxCluster = numel(unique(idx));
    if ismember(-1, idx)
        idx(idx==-1) = maxCluster;
    end
    
    for i = 1 : numel(Population)
        [~, temp_PBA,temp_PBA_SCD] = MMOEA_Utils.nd_pccs_sort([Population(i), PBA{i}]);
        PBA{i}                     = temp_PBA(1:min(numel(temp_PBA),n_PBA));
        PBA_SCD{i}                 = temp_PBA_SCD(1:min(numel(temp_PBA),n_PBA));
    end
    for i = 1 : maxCluster
        [~, LBA{i}, LBA_SCD{i}] = MMOEA_Utils.nd_pccs_sort(Archive(i==idx),ceil(N/maxCluster));
    end
    LBA(maxCluster+1:end)     = [];
    LBA_SCD(maxCluster+1:end) = [];
end

function [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx, maxCluster] = Environmental_Selection_kmeans(Problem, Archive, Population, LBA, PBA, n_PBA, maxCluster)
    N         = Problem.N;
    temp_p    = [Population, Archive, LBA{:}];
    temp_decs = temp_p.decs;
    [~,idx_1] = unique(temp_decs(:,1));
    Archive   = temp_p(idx_1);
    
    temp_Archive    = Archive;
    idx             = kmeans(Archive.decs, maxCluster);    
    temp_Archive_nd = [];
    temp_idx_d_nd   = [];
    
    for i = 1 : maxCluster
        indv_t1         = temp_Archive(idx==i);
        num_ND          = NDSort(indv_t1.objs,1)==1;
        temp_Archive_nd = [temp_Archive_nd,indv_t1(num_ND)];
        temp_idx_d_nd   = [temp_idx_d_nd;i*ones(sum(num_ND),1)];
    end
    Archive = temp_Archive_nd;
    idx     = temp_idx_d_nd;
    
    while numel(Archive) > N
        num_cluster = tabulate(idx);
        [~,idx_max] = max(num_cluster(:,2));
        density_y   = MMOEA_Utils.calc_pccs(Archive.objs);
        sel_cluster = find(idx == idx_max);
        density_yt  = density_y(sel_cluster);
        [~,idx1]    = sort(density_yt,'descend');
        try
            idx2 = idx1(1:Problem.M+1);
        catch
            idx2 = idx1;
        end
        density_x  = MMOEA_Utils.calc_pccs(Archive.decs);
        density_xt = density_x(sel_cluster(idx2));
        [~,idx3]   = max(density_xt);
        Archive(sel_cluster(idx2(idx3))) = [];
        idx(sel_cluster(idx2(idx3)))     = [];
    end
    
    a1    = tabulate(idx);
    idx_t = sort(find(a1(:,2)==0),'descend')';
    for i = idx_t
        idx(idx>i,:) = idx(idx>i,:)-1;
    end
    maxCluster = max(idx);
    
    for i = 1 : N
        [~, temp_PBA,temp_PBA_SCD] = MMOEA_Utils.nd_pccs_sort([Population(i), PBA{i}]);
        PBA{i}                     = temp_PBA(1:min(numel(temp_PBA),n_PBA));
        PBA_SCD{i}                 = temp_PBA_SCD(1:min(numel(temp_PBA),n_PBA));
    end
    for i = 1 : maxCluster
        [~, LBA{i}, LBA_SCD{i}] = MMOEA_Utils.nd_pccs_sort(Archive(i==idx),ceil(N/maxCluster));
    end
    LBA(maxCluster+1:end)     = [];
    LBA_SCD(maxCluster+1:end) = [];
end
```

### `MMOEABH.m`
```matlab
classdef MMOEABH < ALGORITHM
% <2025> <multi> <real/integer/label/binary/permutation> <multimodal>
% Multimodal MOEA with block optimization and hybrid clustering
% epsilon --- 0.15 --- The neighborhood radius for DBSCAN
% minpts  ---    5 --- The minimum number of points required to form a dense region

%------------------------------- Reference --------------------------------
% Y. Zhang and W. Hu. Block optimization and switchable hybrid clustering
% for multimodal multiobjective evolutionary optimization with shifted
% local Pareto front. Swarm and Evolutionary Computation, 2025, 98: 102151.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yu Zhang
% If you have any questions, please open an issue in our GitHub repository:
% https://github.com/yuzhang576/MMOEA-BH

    methods
        function main(Algorithm,Problem)
            %% Parameter setting            
            [g, epsilon, minpts] = Algorithm.ParameterSet(10, 0.15, 5);
            
            %% Generate random population
            Population = Problem.Initialization();
            [PBA,LBA]  = deal(cell(1, numel(Population)));                    
            n_PBA      = 5;
            for i = 1:numel(Population)
                [~, temp_PBA,temp_PBA_SCD] = MMOEA_Utils.nd_pccs_sort(Population(i));
                PBA{i}                     = temp_PBA(1:min(numel(temp_PBA),n_PBA));
                PBA_SCD{i}                 = temp_PBA_SCD(1:min(numel(temp_PBA),n_PBA));
            end
            [idx_APC, maxCluster] = MMOEA_Utils.APC(Population.decs);
            for i = 1:maxCluster
                [~, LBA{i}, LBA_SCD{i}] = MMOEA_Utils.nd_pccs_sort(Population(i==idx_APC));
            end
            Archive          = [LBA{1:maxCluster}];
            [gen, count_cso] = deal(1);
            
            %% Optimization
            while Algorithm.NotTerminated(Archive)
                if Problem.FE < Problem.maxFE*0.5 % Stage 1: PSO
                    [Pbest, Nbest] = MMOEA_Utils.rep_selection_pso(Problem, PBA, PBA_SCD, LBA, LBA_SCD, idx_APC, maxCluster);
                    Population     = operator_PSO(Problem, Population, Pbest, Nbest);
                    if mod(gen,g) == 0                        
                        [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx_APC, maxCluster] = Environmental_Selection('APC', Problem, Archive, Population, LBA, PBA, n_PBA);
                    else                        
                        [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx_APC, maxCluster] = Environmental_Selection('kmeans', Problem, Archive, Population, LBA, PBA, n_PBA, maxCluster);
                    end
                else                              % Stage 2: CSO
                    if mod(count_cso,2) == 1
                        count_cso          = count_cso + 1;
                        pop_cso            = [];
                        [orig_vol, center] = MMOEA_Utils.get_orig_vol(Problem, maxCluster, LBA);
                        for i = 1 : maxCluster
                            pop                   = LBA{i};
                            [new_up_s, new_low_s] = MMOEA_Utils.get_upper_lower(Problem, pop, orig_vol);
                            [new_up_s, new_low_s] = MMOEA_Utils.clip_upper_lower(new_up_s, new_low_s, center);
                            
                            N                     = ceil(Problem.N/maxCluster);
                            FEs                   = N*25;
                            result                = operator_CSO(Problem, pop, new_low_s, new_up_s, N, FEs);
                            pop_cso               = [pop_cso,result];
                        end
                        if numel(pop_cso) < Problem.N
                            repmat_num = Problem.N - numel(pop_cso);
                            pop_cso    = [pop_cso,pop_cso(randi(numel(pop_cso),repmat_num,1))];
                        end
                        [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx_APC, maxCluster] = Environmental_Selection('DBSCAN', Problem, Archive, pop_cso(1:Problem.N), LBA, PBA, n_PBA, Problem.N, maxCluster, epsilon, minpts);
                    else
                        count_cso      = count_cso + 1;
                        [Pbest, Nbest] = MMOEA_Utils.rep_selection_pso(Problem, PBA, PBA_SCD, LBA, LBA_SCD, idx_APC, maxCluster);
                        Population     = operator_PSO(Problem, Population, Pbest, Nbest);
                        [Archive, LBA, LBA_SCD, PBA, PBA_SCD, idx_APC, maxCluster] = Environmental_Selection('APC', Problem, Archive, Population, LBA, PBA, n_PBA);
                    end
                end
                gen = gen + 1;
            end
        end
    end
end
```

### `MMOEA_Utils.m`
```matlab
classdef MMOEA_Utils
% MMOEA_Utils - Static class for all utility and auxiliary functions

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    methods(Static)
        %% 1. APC Clustering
        function varargout = APC(data)
            N    = size(data,1);
            DM   = pdist2(data,data);
            S    = -DM;
            meds = median(S,'all');
            n    = 5;
            
            S(logical(eye(N))) = n*meds;
            A     = zeros(N,N);
            R     = zeros(N,N); 
            lam   = 0.6; 
            count = 0;
            
            for iter = 1 : 500
                Eold  = A + R;
                Rold  = R;
                AS    = A+S;
                [Y,I] = max(AS,[],2);
                for i = 1 : N
                    AS(i,I(i)) = -inf;
                end
                Y2 = max(AS,[],2);
                R  = S-Y;
                for i = 1 : N
                    R(i,I(i)) = S(i,I(i))-Y2(i);
                end
                R = (1-lam)*R+lam*Rold; 
                
                Aold = A;
                Rp   = max(R,0);    
                Rp(logical(eye(N))) = R(logical(eye(N)));
                A    = sum(Rp,1)-Rp;
                dA   = diag(A);
                A    = min(A,0);    
                A(logical(eye(N))) = dA;
                A    = (1-lam)*A+lam*Aold; 
                E    = A + R;
                
                if diag(Eold) == diag(E)
                    count = count + 1;
                    if count == 10
                        break;
                    end
                else
                    count=0;
                end
            end
            E      = R + A; 
            I      = find(diag(E)>0);
            K      = length(I);
            [~, c] = max(E(:,I),[],2);
            c(I)   = 1 : K;                 
            idx    = I(c); 
            
            maxCluster = 0;
            for i = unique(idx)'
                maxCluster  = maxCluster + 1;
                idx(idx==i) = maxCluster;
            end
            varargout = {idx, maxCluster};
        end
        
        %% 2. Calculate PCCS
        function varargout = calc_pccs(obj)
            K = size(obj,1);
            if K >= 2
                fmax    = max(obj);
                fmin    = min(obj);
                L       = ceil(K * (obj-fmin)./(fmax-fmin));
                L(L==0) = L(L==0) + 1;
                
                PCD = pdist2(L,L,'cityblock');
                PCD(logical(eye(K))) = [];
                PCD         = reshape(PCD,K-1,K)';
                PCD(PCD==0) = PCD(PCD==0) + 0.5;
                density     = sum(1./PCD.^2,2);
                varargout   = {density};
            else
                varargout   = {1};
            end
        end
        
        %% 3. Non-dominated PCCS Sort
        function [selected_population, sorted_population, sorted_SCD]= nd_pccs_sort(population, aSize)
            if nargin < 2
                aSize = 1000;
            end
            PopDec = population.decs;
            PopObj = population.objs;
            [N,~]  = size(PopDec);
            if N == 1
                selected_population = population;
                sorted_population   = population;
                sorted_SCD          = 1;
            else        
                CD_x    = normalize( -MMOEA_Utils.calc_pccs(PopDec), 'range' );
                CD_f    = normalize( -MMOEA_Utils.calc_pccs(PopObj), 'range' );
                idx_max = bitor(CD_x>mean(CD_x), CD_f>mean(CD_f));
                idx_min = ~idx_max;
                
                CD(idx_max) = CD_x(idx_max) + CD_f(idx_max);
                CD(idx_min) = min(CD_x(idx_min), CD_f(idx_min));
                [sorted_SCD, idx]   = sort(CD,'descend');
                selected_population = population(idx(1));
                
                if numel(idx) > aSize
                    sorted_population = population(idx(1:aSize));
                    sorted_SCD        = sorted_SCD(1:aSize);
                else
                    sorted_population = population(idx);
                end
            end
        end
        
        %% 4. REP Selection for PSO
        function [pbest, lbest] = rep_selection_pso(Problem, PBA, PBA_SCD, LBA, LBA_SCD, idx_APC, maxCluster_APC)
            for i = 1 : numel(PBA)
                idx1     = 1;
                pbest(i) = PBA{i}(idx1);
            end
            if numel(idx_APC)<Problem.N
                k       = ceil(Problem.N/numel(idx_APC));
                idx_APC = repmat(idx_APC,k,1);    
            end
            idx_APC = idx_APC(1:Problem.N);
            for i = 1 : maxCluster_APC
                lbest_temp = LBA{i};
                pos        = i==idx_APC;
                idx        = TournamentSelection(2,sum(pos),-LBA_SCD{i});
                lbest(pos) = lbest_temp(idx);
            end
        end
        
        %% 5. Get Original Volume
        function [orig_vol, center] = get_orig_vol(Problem, maxCluster, LBA)
            for i = 1 : maxCluster
                pop         = LBA{i};
                pop_dec     = pop.decs;
                center(i,:) = mean(pop_dec,1);
            end
            grid_num = ceil(power(maxCluster,1/Problem.D));
            len1     = (Problem.upper-Problem.lower)/grid_num;
            orig_vol = prod(len1);
        end
        
        %% 6. Get Upper and Lower bounds
        function [new_up_s, new_low_s] = get_upper_lower(Problem, pop, orig_vol)
            pop_dec          = pop.decs;
            upper_small      = max(pop_dec,[],1);
            lower_small      = min(pop_dec,[],1);
            dimensions_small = upper_small - lower_small;
            
            ratio          = dimensions_small ./ max(dimensions_small);
            adjusted_ratio = max(ratio, 0.2);
            ratio          = adjusted_ratio ./ sum(adjusted_ratio);
            
            new_volume     = orig_vol;
            new_dimensions = (new_volume / prod(ratio))^(1/length(ratio)) * ratio;
            
            center    = (upper_small + lower_small) / 2;
            new_low_s = center - new_dimensions / 2;
            new_up_s  = new_low_s + new_dimensions;
            new_up_s  = max(min(new_up_s,Problem.upper),Problem.lower);
            new_low_s = max(min(new_low_s,Problem.upper),Problem.lower);
        end
        
        %% 7. Clip Upper and Lower bounds
        function [new_new_up_s, new_new_low_s] = clip_upper_lower(new_up_s, new_low_s, center)
            inside_each_dim  = (center >= new_low_s) & (center <= new_up_s);
            inside_rectangle = all(inside_each_dim, 2);
            points_inside    = center(inside_rectangle, :);
            num_inside       = sum(inside_rectangle); 
            
            if num_inside < 2
                new_new_up_s  = new_up_s;
                new_new_low_s = new_low_s;
            else
                original_volume = prod(new_up_s - new_low_s);
                target_volume   = original_volume / num_inside;
                X               = points_inside(:, 1:end-1);
                y               = points_inside(:, end);
                mdl             = fitlm(X, y);
                
                coefficients        = round(1./abs([mdl.Coefficients.Estimate(2:end)', 1]), 2);
                coefficients(isinf(coefficients)) = 10^6;
                original_dimensions = new_up_s - new_low_s;
                syms a
                equation_terms = 1;
                for i = 1 : length(original_dimensions)
                    equation_terms = equation_terms * (original_dimensions(i) - coefficients(i) * a);
                end
                
                equation  = equation_terms == target_volume;
                solutions = double(solve(equation, a));
                valid_solution = [];
                for i = 1 : length(solutions)
                    if all(solutions(i) > 0) && all(original_dimensions ./ coefficients > solutions(i))
                        valid_solution = solutions(i);
                        break;
                    end
                end
                
                if isempty(valid_solution)
                    error('No valid solution found');
                end
                adjusted_target_side_length = original_dimensions - coefficients * valid_solution;
                centroid      = mean((new_up_s + new_low_s) / 2, 1);
                new_new_low_s = centroid - adjusted_target_side_length / 2;
                new_new_up_s  = centroid + adjusted_target_side_length / 2;
                new_new_up_s  = max(new_new_up_s, new_low_s);
                new_new_low_s = min(new_new_low_s, new_up_s);
            end
        end
    end
end
```

### `operator_CSO.m`
```matlab
function Population = operator_CSO(Problem, pop_LBA, lower_bound, upper_bound, N, maxFEs)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Generate random population
    [V, N] = UniformPoint(N, Problem.M);
    
    Dec      = unifrnd(repmat(lower_bound, N, 1), repmat(upper_bound, N, 1));
    pop_init = Problem.Evaluation(Dec);
    
    if numel(pop_LBA) < N
        Population = [pop_LBA, pop_init];
        Population = Population(1:N);
    else
        Population = pop_LBA(1:N);
    end
    
    currentFE  = length(pop_init);
    Population = EnvironmentalSelection(Population, V, (currentFE/maxFEs)^2);
    
    %% Optimization
    while currentFE < maxFEs
        Fitness = calFitness(Population.objs);
        if length(Population) >= 2
            Rank = randperm(length(Population), floor(length(Population)/2)*2);
        else
            Rank = [1, 1];
        end
        
        Loser  = Rank(1:end/2);
        Winner = Rank(end/2+1:end);
        Change = Fitness(Loser) >= Fitness(Winner);
        Temp   = Winner(Change);
        
        Winner(Change) = Loser(Change);
        Loser(Change)  = Temp;
        
        % Generate offspring using the current subspace bounds
        Offspring = Operator_niche(Problem, Population(Loser), Population(Winner), lower_bound, upper_bound);
        
        currentFE  = currentFE + length(Offspring);
        Population = EnvironmentalSelection([Population, Offspring], V, (currentFE/maxFEs)^2);
    end
end

function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density
    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end

function Population = EnvironmentalSelection(Population,V,theta)
% The environmental selection of LMOCSO

    Population = Population(NDSort(Population.objs,1)==1);
    PopObj     = Population.objs;
    [N,M]      = size(PopObj);
    NV         = size(V,1);
    
    %% Translate the population
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    
    %% Calculate the degree of violation of each solution
    CV     = sum(max(0,Population.cons),2);

    %% Calculate the smallest angle value between each vector and others
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle         = acos(1-pdist2(PopObj,V,'cosine'));
    [~,associate] = min(Angle,[],2);

    %% Select one solution for each reference vector
    Next = zeros(1,NV);
    for i = unique(associate)'
        current1 = find(associate==i & CV==0);
        current2 = find(associate==i & CV~=0);
        if ~isempty(current1)
            % Calculate the APD value of each solution
            APD      = (1+M*theta*Angle(current1,i)/gamma(i)).*sqrt(sum(PopObj(current1,:).^2,2));
            [~,best] = min(APD);
            Next(i)  = current1(best);
        elseif ~isempty(current2)
            % Select the one with the minimum CV value
            [~,best] = min(CV(current2));
            Next(i)  = current2(best);
        end
    end
    % Population for next generation
    Population = Population(Next(Next~=0));
end

function Offspring = Operator_niche(Problem, Loser, Winner, lower_bound, upper_bound)
% The competitive swarm optimizer of LMOCSO customized for local boundaries

    %% Parameter setting
    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(LoserDec);
    LoserVel  = Loser.adds(zeros(N,D));
    WinnerVel = Winner.adds(zeros(N,D));
    
    %% Competitive swarm optimizer
    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    OffVel = r1.*LoserVel + r2.*(WinnerDec-LoserDec);
    OffDec = LoserDec + OffVel + r1.*(OffVel-LoserVel);
    
    %% Add the winners
    OffDec = [OffDec;WinnerDec];
    OffVel = [OffVel;WinnerVel];
    
    %% Polynomial mutation (using the externally provided local block bounds)
    Lower = repmat(lower_bound, 2*N, 1);
    Upper = repmat(upper_bound, 2*N, 1);
    disM  = 20;
    Site  = rand(2*N,D) < 1/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    OffDec       = max(min(OffDec,Upper),Lower);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
               
    temp         = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```

### `operator_PSO.m`
```matlab
function Offspring = operator_PSO(Problem,Particle,Pbest,Gbest)
% Particle swarm optimization in SMPSO

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

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
    OffDec         = max(min(OffDec,Upper),Lower);
    
    %% Polynomial mutation
    disM  = 20;
    Site1 = repmat(rand(N,1)<0.15,1,D);
    Site2 = rand(N,D) < 1/D;
    mu    = rand(N,D);
    temp         = Site1 & Site2 & mu<=0.5;
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
                   
    temp         = Site1 & Site2 & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec,OffVel);
end
```
