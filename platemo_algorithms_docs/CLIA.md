# CLIA

**Tags**: <2019> <multi/many> <real/integer/label/binary/permutation>

## Description
Evolutionary algorithm with cascade clustering and reference point incremental learning

## Reference
H. Ge, M. Zhao, L. Sun, Z. Wang, G. Tan, Q. Zhang, and C. L. P. Chen. A many-objective evolutionary algorithm with two interacting processes: Cascade clustering and reference point incremental learning. IEEE Transactions on Evolutionary Computation, 2019, 23(4): 572-586.

## Source Code

### `CLIA.m`
```matlab
classdef CLIA < ALGORITHM
% <2019> <multi/many> <real/integer/label/binary/permutation>
% Evolutionary algorithm with cascade clustering and reference point incremental learning

%------------------------------- Reference --------------------------------
% H. Ge, M. Zhao, L. Sun, Z. Wang, G. Tan, Q. Zhang, and C. L. P. Chen. A
% many-objective evolutionary algorithm with two interacting processes:
% Cascade clustering and reference point incremental learning. IEEE
% Transactions on Evolutionary Computation, 2019, 23(4): 572-586.
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
            global stable_threshold delta crowding_pick_flag;
            [stable_threshold, delta] = Algorithm.ParameterSet([0, 0, 0], 2 * Problem.N);
            [Z, P, A, S, SVM] = initialize(Problem);
            while Algorithm.NotTerminated(S)
                MatingPool = TournamentSelection(2, Problem.N, sum(max(0, P.cons), 2)); 
                Offspring  = OperatorGA(Problem,P(MatingPool));
                A = update_archive(A, [P, Offspring], Z, ceil(0.33 * Problem.M * Problem.N), Problem);
                % SELECTION OF INDIVIDUALS
                [P, ICA, ICN] = cascade_cluster([P, Offspring], Z, 'PDM', Problem.N, Problem.FE < Problem.maxFE);
                % ADAPTATION OF REFERENCE VECTORS     
                [Z, SVM] = incremental_learn(Z, ICA, ICN, A, SVM, Problem);
                if Problem.FE >= Problem.maxFE && crowding_pick_flag
                    S = crowding_pick(update_archive(A, P, Z, [], Problem), Problem.N, 'precise');
                else
                    S = P;
                end
            end
        end
    end
end
```

### `cascade_cluster.m`
```matlab
function varargout = cascade_cluster(P, Z, MtcStr, N, cat_flag)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% split the population
    [O, ia, ~] = unique([P.objs], 'rows');
    P = P(ia);
    [index_frontier, NF_index] = find_frontiers(O, [P.cons]);
    O = normalization(O);
    %% attach frontiers to nearest reference vectors
    if strcmp(MtcStr, 'PBI')
        error_type = 'distance';
    elseif strcmp(MtcStr, 'PDM')
        error_type = 'sin';
    end
    [min_metric, allocation] = pair(O(index_frontier, :), Z, error_type);
    %% identify the active reference vectors and create clusters
    index_active_cluster = unique(allocation);
    NC = numel(index_active_cluster);
    cluster.center = []; cluster.queue_select = [];
    clusters = repmat(cluster, NC, 1);
    %% calculate metric and sort the frontier solutions
    for i = 1: NC
        I = index_active_cluster(i);
        small_index = (allocation == I);
        index_cluster_frontier = index_frontier(small_index);
        if numel(index_cluster_frontier) == 1
            index_ascend = 1;
        else
            if strcmp(MtcStr, 'PBI')
                F_metric = 5 * min_metric(small_index)' + sum(O(index_cluster_frontier, :) .* repmat(Z(I, :), numel(index_cluster_frontier), 1), 2)' ./ repmat(norm(Z(I, :)), 1, numel(index_cluster_frontier));
            elseif strcmp(MtcStr, 'PDM')
                F_metric = 5 * (min_metric(small_index) .* sqrt(sum(O(index_cluster_frontier, :) .^ 2, 2)))' + mean(O(index_cluster_frontier, :), 2)';
            end
            if NC >= N
                [~, index_ascend] = min(F_metric);
            else
                [~, index_ascend] = sort(F_metric, 'ascend');
            end
        end
        clusters(i).queue_select = index_cluster_frontier(index_ascend);
        clusters(i).center = clusters(i).queue_select(1);
    end
    %% Attach the non-frontiers and sort
    center_index = [clusters.center];
    if nargout == 2
        CF = P(center_index);
    end
    if NC > N
        P = crowding_pick(P(center_index), N, 'precise');
    elseif NC == N
        P = P(center_index);
    else
        if cat_flag || numel(index_frontier) < N
            C = O(center_index, :);
            Distance2Centers = pdist2(O(NF_index, :), C, 'euclidean');
            [min_distance, allocation] = min(Distance2Centers, [], 2);
            active_clusters_with_NF_index = unique(allocation);
            for i = 1: numel(active_clusters_with_NF_index)
                I = active_clusters_with_NF_index(i);
                Cluster_NF_index = NF_index(allocation == I);
                NF_distance = min_distance(allocation == I);
                [~, index_ascend] = sort(NF_distance, 'ascend');
                clusters(I).queue_select = [clusters(I).queue_select, Cluster_NF_index(index_ascend)];
            end
        end
        %% Round-robin Picking
        j = 1;
        next_pop_index = zeros(1, N);
        clusters = clusters(randperm(NC));
        for pointer = 1: N
            while isempty(clusters(j).queue_select)
                j = mod(j, NC) + 1;
            end
            next_pop_index(pointer) = clusters(j).queue_select(1);
            clusters(j).queue_select(1) = [];
            j = mod(j, NC) + 1;
        end
        P = P(next_pop_index);
    end
    varargout{1} = P;
    if nargout == 2
        varargout{2} = CF;
    elseif nargout == 3
        varargout{2} = index_active_cluster;
        varargout{3} = setdiff(1: size(Z, 1), index_active_cluster);
    end
end
```

### `crowding_pick.m`
```matlab
function frontiers_picked = crowding_pick(archive, picks, mode)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if numel(archive) < picks
        frontiers_picked = archive;
    else
        O = normalize([archive.objs]);
        if strcmp(mode, 'precise')
            index_pick = true(1, numel(archive));
            while numel(find(index_pick == true)) > picks % Optimizable
                CD = NaN(1, numel(archive));
                CD(index_pick) = crowding_dist(O);
                [~, index] = min(CD);
                index_pick(index) = false;
                O = normalization([archive(index_pick).objs]);
            end
            frontiers_picked = archive(index_pick);
        elseif strcmp(mode, 'fast')
            CD = crowding_dist(O);
            [~, I] = sort(CD, 'descend');
            frontiers_picked = archive(I(1: picks));
        end
    end
end

function CD = crowding_dist(O)
    [N, M] = size(O);
    CD = zeros(1, N);
    Fmax = max(O, [], 1);
    Fmin = min(O, [], 1);
    for i = 1: M
        [~, Rank] = sort(O(:, i));
        CD(Rank(1)) = inf;
        CD(Rank(end)) = inf;
        for j = 2 : N - 1
            CD(Rank(j)) = CD(Rank(j)) + (O(Rank(j + 1), i) - O(Rank(j - 1), i)) / (Fmax(i) - Fmin(i));
        end
    end
end
```

### `find_frontiers.m`
```matlab
function [F_index, NF_index] = find_frontiers(PopObj, PopCon)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    perm_index = randperm(size(PopObj, 2));
    PopObj     = PopObj(:, perm_index);
    Infeasible = any(PopCon>0, 2);
    PopObj(Infeasible,:) = repmat(max(PopObj,[],1),sum(Infeasible),1) + repmat(sum(max(0,PopCon(Infeasible,:)),2),1,size(PopObj,2));
    [PopObj, ~, Loc]     = unique(PopObj,'rows');
    [N,M]         = size(PopObj);
    [PopObj,rank] = sortrows(PopObj);
    FrontNo       = inf(1,N);
    MaxFNo        = 1;
    for i = 1 : N
        if FrontNo(i) == inf
            Dominated = false;
            for j = i-1 : -1 : 1
                if FrontNo(j) == MaxFNo
                    m = 2;
                    while m <= M && PopObj(i, m) >= PopObj(j, m)
                        m = m + 1;
                    end
                    Dominated = m > M;
                    if Dominated || M == 2
                        break;
                    end
                end
            end
            if ~Dominated
                FrontNo(i) = MaxFNo;
            end
        end
    end
    FrontNo(rank) = FrontNo;
    FrontNo  = FrontNo(Loc);
    F_index  = find(FrontNo == 1);
    NF_index = setdiff(1: size(PopObj, 1), F_index);
end
```

### `generate_reference.m`
```matlab
function [Z, active_Z] = generate_reference(PopObj, original_active_number, Global, action, mode)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    global LEARNING_DISABLE_FLAG;
    if strcmp(mode, 'strict')
        active_number = 0;
        PopObj = normalization(PopObj);
        active_Z = [];
        while active_number < original_active_number
            instructed_action = action;
            [Z, received_action] = generate_Z(Global, instructed_action, 'commit');
            Z = unique([Z; active_Z], 'rows');
            [~, Allocation] = pair(PopObj, Z, 'sin');
            active_index = unique(Allocation);
            active_Z = Z(active_index, :);
            active_number = size(active_Z, 1);
            fprintf('%d(+%d)\n', active_number, numel(active_index)); % TODO
            if received_action ~= instructed_action
                LEARNING_DISABLE_FLAG = true;
                warning('reference generation malfunctions');
                break;
            end
        end
    elseif strcmp(mode, 'normal')
        Z = generate_Z(Global, action, 'commit');
        active_Z = [];
    end
end

function varargout = generate_Z(Global, action, commit_str)
    global MAX_REFERENCE_NUM_FLAG current_density reference_generation_intial_flag;
    persistent incremental_sequence pointer;
    if reference_generation_intial_flag == true
        MAX_REFERENCE_NUM_FLAG = false;
        incremental_sequence = density_sequence(Global.M);
        pointer = 1;
        reference_generation_intial_flag = false;
        % TRIM
        TRIM_INDEX = find(incremental_sequence < Global.N);
        incremental_sequence(TRIM_INDEX(1: end)) = [];
    end
    original_pointer = pointer;
    pointer = pointer + action;
    if pointer >= numel(incremental_sequence)
        MAX_REFERENCE_NUM_FLAG = true;
    end
    pointer = min(numel(incremental_sequence), pointer);
    pointer = max(1, pointer);
    real_action = pointer - original_pointer;
    
    Z = UniformPoint(incremental_sequence(pointer), Global.M);
    current_density = size(Z, 1);
    
    if ~strcmp(commit_str, 'commit')
        pointer = original_pointer;
    end
    
    varargout{1} = Z;
    if nargout >= 2
        varargout{2} = real_action;
    end
end

function incremental_sequence = density_sequence(M)
    incremental_sequence = [];
    R_MAX = 1e6;
    step = 1;
    R = nchoosek(M + step - 1, step);
    while R < R_MAX
        incremental_sequence = [incremental_sequence, R];
        step = step + 1;
        R = nchoosek(M + step - 1, step);
    end
end
```

### `incremental_learn.m`
```matlab
function [Z, SVM] = incremental_learn(Z, active_clusters_index, index_inactive_clusters, frontiers, SVM, Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    global MAX_REFERENCE_NUM_FLAG MAX_ARCHIVE_SIZE disable_flag normalization_str delta learning_initial_flag;
    if learning_initial_flag == true
        learning_initial_flag = false;
        MAX_ARCHIVE_SIZE = ceil(0.33 * Problem.M * Problem.N);
    end
    if disable_flag && numel(frontiers) > 0.9 * MAX_ARCHIVE_SIZE || numel(active_clusters_index) > 0.95 * Problem.N && size(Z, 1) == Problem.N
        return;
    end
    status = check_status(size(Z, 1), active_clusters_index, index_inactive_clusters, Problem);
    if strcmp(status, 'unstable') || numel(frontiers) < 0.9 * MAX_ARCHIVE_SIZE
        return;
    end
    if numel(active_clusters_index) > Problem.N && numel(active_clusters_index) == size(Z, 1)
        Z = generate_reference([], 0, Problem, -1, 'normal');
    elseif numel(active_clusters_index) < 0.95 * Problem.N
        if MAX_REFERENCE_NUM_FLAG
            return;
        end
        [~, allocation] = pair([frontiers.objs], Z, 'sin');
        Z_active_old = Z(unique(allocation), :);
        [Z, ~] = generate_reference([frontiers.objs], numel(active_clusters_index), Problem, 1, 'normal');
        if strcmp(normalization_str, 'normalize')
            if size(Z, 1) > 4 * Problem.N
                normalization_str = '';
                [~, allocation] = pair([frontiers.objs], Z, 'sin');
                index_inactive_clusters = setdiff(1: size(Z, 1), unique(allocation));
                P = truncate([frontiers.objs]); N = truncate(Z(index_inactive_clusters, :));
                SVM = learn(P, N, SVM, 'svmtrain2');
                Z_next = reduce_useless(Z, delta, SVM, Problem);
            else
                Z_next = Z;
            end
        elseif size(Z, 1) > 2 * Problem.N
            picked_frontiers = crowding_pick(frontiers, 2 *  Problem.N, 'precise');
            [~, allocation] = pair([frontiers.objs], Z, 'sin');
            index_inactive_clusters = setdiff(1: size(Z, 1), unique(allocation));
            P = truncate([picked_frontiers.objs]); N = truncate(Z(index_inactive_clusters, :));
            if size(P, 1) > 0 && size(N, 1) > 0
                [~, y_active] = predict(P, SVM); p_index = find(y_active == 1);
                [~, y_inactive] = predict(N, SVM); n_index = find(y_inactive == -1);
                P = P(setdiff(1: size(P, 1), p_index), :); N = N(setdiff(1: size(N, 1), n_index), :);
                SVM = learn(P, N, SVM, 'svmtrain2');
                Z_next = reduce_useless(Z, delta, SVM, Problem);
            else
                Z_next = Z;
            end
        else
            Z_next = Z;
        end
        [~, allocation] = pair([frontiers.objs], Z, 'sin');
        Z = unique([Z_active_old; Z_next; Z(unique(allocation), :)], 'rows');
    end
end

function Z = reduce_useless(Z, threshold, SVM, Problem)
    original_size = size(Z, 1);
    if original_size <= Problem.N
        index_remain = 1: original_size;
    else
        [score, ~] = predict(Z, SVM);
        if length(score) == 1
            index_remain = 1: original_size;
        elseif isempty(threshold)
            index_remain = [];
        elseif threshold > 0 && threshold <= 1
            threshold = min(threshold, 0.5);
            index_remain = find(score >= threshold);
        elseif threshold > 1
            S = sort(score, 'descend');
            threshold = S(min(max(Problem.N, threshold), numel(S)));
            index_remain = find(score >= threshold);
        end
        if numel(index_remain) < Problem.N
            S = sort(score, 'descend');
            threshold = S(min(Problem.N, numel(S)));
            index_remain = find(score >= threshold);
        end
    end
    Z = Z(index_remain, :);
    fprintf('reference vectors reduced from %d to %d (-%d)\n', original_size, numel(index_remain), original_size - numel(index_remain));
end

function [status, index_i_active, index_i_inactive] = check_status(total_number, active_index, inactive_index, Problem)
    global normalization_str current_density status_initial_flag stable_threshold;
    persistent density history consecutive_stable_counter threshold;
    if status_initial_flag == true
        status_initial_flag = false;
        consecutive_stable_counter = 0;
        density = current_density;
        history = zeros(1, total_number);
        if Problem.M == 5
            pointer = 1;
        elseif Problem.M == 10
            pointer = 2;
        elseif Problem.M == 15
            pointer = 3;
        else
            pointer = 0;
        end
        if pointer > 0 && stable_threshold(pointer) > 0
            threshold = stable_threshold(pointer);
        else
            threshold = min(20, max(5, ceil(Problem.maxFE / 2e4)));
        end
    end
    if strcmp(normalization_str, 'normalize')
        fluctuation = 3;
    else
        fluctuation = 1e-2;
    end
    if density ~= current_density
        density = current_density;
        status_initial_flag = false;
        consecutive_stable_counter = 0;
        history = zeros(1, total_number);
    end
    current = zeros(size(history));
    current(active_index) = 1;
    if norm(history - current) <= fluctuation
        consecutive_stable_counter = consecutive_stable_counter + 1;
    else
        history = current;
        consecutive_stable_counter = 0;
    end
    if consecutive_stable_counter >= threshold
        consecutive_stable_counter = 0;
        threshold = max(5, threshold - 1);
        status = 'stable';
        index_i_active = active_index; index_i_inactive = inactive_index;
    else
        status = 'unstable';
        index_i_active = []; index_i_inactive = [];
    end
end

function SVM = learn(P, N, SVM, training_mode)
    length_P = size(P, 1); length_N = size(N, 1);
    if length_P == 0 || length_N == 0
        return;
    end
    X = project2simplex([P; N]); y = [ones(length_P, 1); -1 * ones(length_N, 1)];
    if ~isempty(SVM.y_mer) % NOT NEW
        [SVM.a, SVM.b, SVM.g, SVM.ind, SVM.uind, SVM.X_mer, SVM.y_mer, SVM.Rs, SVM.Q] = feval(training_mode, X', y, SVM.C);
    else
        [SVM.a, SVM.b, SVM.g, SVM.ind, SVM.uind, SVM.X_mer, SVM.y_mer, SVM.Rs, SVM.Q] = feval(training_mode, X', y, SVM.C, SVM.type, SVM.scale);
    end
end

function projected_points = project2simplex(O)
    global hp_initial_flag;
    persistent V Q;
    if isempty(V) || hp_initial_flag == true
        M = size(O, 2);
        Q = orth(eye(M, M) - repmat(ones(M, 1) ./ M, 1, M));
        hp_initial_flag = false;
    end
    if size(O, 1)
        projected_points = O * Q;
    else
        projected_points = [];
    end
end

function [score, class] = predict(X, SVM)
    class = nan(1, size(X, 1));
    score = class';
    if ~isempty(SVM.y_mer)
        block_size = 10000;
        X = project2simplex(X);
        blocks = floor(size(X, 1) / block_size);
        score = nan(size(size(X, 1), 1));
        B = mat2cell(X, [block_size * ones(1, blocks), size(X, 1) - block_size * blocks], size(X, 2));
        for i = 1: blocks
            score((i - 1) * block_size + 1: i * block_size) = 0.5 + 0.5 * elliotsig(svmeval((B{i})', SVM.a, SVM.b, SVM.ind, SVM.X_mer, SVM.y_mer, SVM.type, SVM.scale));
        end
        score(block_size * blocks + 1: size(X, 1)) = 0.5 + 0.5 * elliotsig(svmeval((B{blocks + 1})', SVM.a, SVM.b, SVM.ind, SVM.X_mer, SVM.y_mer, SVM.type, SVM.scale));
        class(score >= 0.5) = 1; class(score < 0.5) = -1;
    end
end

function Z = truncate(Z)
    Z = Z ./ repmat(sum(Z, 2), 1, size(Z, 2));
end
```

### `initialize.m`
```matlab
function [Z, Population, Archive, Sample, SVM] = initialize(Global)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    global incremental_sequence stable_threshold LARGE_ARCHIVE_FLAG SVM_KERNEL SVM_KERNEL_SCALE SVM_EXIST_FLAG reserved_evaluations reference_generation_mode crowding_pick_flag lb ub reference_generation_intial_flag learning_initial_flag normalization_str status_initial_flag disable_flag hp_initial_flag;
    rng('shuffle');
    SVM_EXIST_FLAG   = false;
    SVM_KERNEL       = 'gaussian';
    SVM_KERNEL_SCALE = 0.056;
    lb = zeros(1, Global.M);
    ub = ones(1, Global.M);
    crowding_pick_flag               = false;
    learning_initial_flag            = true;
    reference_generation_intial_flag = true;
    status_initial_flag              = true;
    hp_initial_flag                  = true;
    reference_generation_mode        = 'strict';
    LARGE_ARCHIVE_FLAG               = false;
    reserved_evaluations             = inf;
    disable_flag                     = false;
    SVM.a     = [];
    SVM.b     = [];
    SVM.g     = [];
    SVM.ind   = [];
    SVM.uind  = [];
    SVM.X_mer = [];
    SVM.y_mer = [];
    SVM.Rs    = [];
    SVM.Q     = [];
    SVM.scale = 0.056;
    SVM.type  = 5;
    SVM.C     = 10;
    problem   = class(Global);
    
    %% hyperparameters for MaF
    if strcmp(problem, 'MaF1')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [20, 20, 15];
        end
        if Global.M == 5
            disable_flag = false;
            crowding_pick_flag = false;
            incremental_sequence = [210,330,495,715,1001,1365,1820,2380,3060,3876,4845,8855];
        elseif Global.M == 10
            incremental_sequence = [230,275,440,16445];
            disable_flag = false;
            crowding_pick_flag = true;
        end
        normalization_str = '';
        reserved_evaluations = 150000;
    end
    if strcmp(problem, 'MaF2')
        if Global.M == 5
            incremental_sequence = [210;330;495;715;1001;1365;1820;2380];
            disable_flag = false;
            crowding_pick_flag = false;
        elseif Global.M == 10
            incremental_sequence = [230,275,440,715,725,770,935,1430,2002];
            disable_flag = false;
            crowding_pick_flag = true;
            normalization_str = '';
        end
    end
    if strcmp(problem, 'MaF3')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [inf, inf, inf];
        end
        crowding_pick_flag = false;
        disable_flag = true;
        normalization_str = '';
    end
    if strcmp(problem, 'MaF4')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [5, 10, 15];
        end
        if Global.M == 5
            disable_flag = false;
            LARGE_ARCHIVE_FLAG = true;
            crowding_pick_flag = false;
        elseif Global.M == 10
            disable_flag = false;
            crowding_pick_flag = false;
            reserved_evaluations = 150000;
        end
        normalization_str = '';
    end
    if strcmp(problem, 'MaF5')
        crowding_pick_flag = false;
        disable_flag = true;
        normalization_str = 'normalize';
    end
    if strcmp(problem, 'MaF6')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [5, 10, 15];
        end
        if Global.M == 5
            disable_flag = false;
            crowding_pick_flag = true;
        elseif Global.M == 10
            disable_flag = false;
            crowding_pick_flag = false;
        end
        normalization_str = '';
        reference_generation_mode = 'normal';
    end
    if strcmp(problem, 'MaF7')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [10, 15, 20];
        end
        if Global.M == 5
            disable_flag = false;
            crowding_pick_flag = false;
        elseif Global.M == 10
            disable_flag = false;
            crowding_pick_flag = false;
        end
        normalization_str = 'normalize';
        reserved_evaluations = 150000;
        reference_generation_mode = 'normal';
    end
    if strcmp(problem, 'MaF8')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [5, 5, 5];
        end
        if Global.M == 5
            disable_flag = false;
            crowding_pick_flag = false;
        elseif Global.M == 10
            disable_flag = false;
            crowding_pick_flag = true;
        end
        normalization_str = 'normalize';
        reference_generation_mode = 'normal';
    end
    if strcmp(problem, 'MaF9')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [5, 5, 5];
        end
        if Global.M == 5
            disable_flag = false;
            crowding_pick_flag = false;
        elseif Global.M == 10
            disable_flag = false;
            crowding_pick_flag = false;
        end
        normalization_str = '';
        reference_generation_mode = 'normal';
    end
    if strcmp(problem, 'MaF10')
        crowding_pick_flag = false;
        disable_flag = true;
        LARGE_ARCHIVE_FLAG = false;
    end
    if strcmp(problem, 'MaF11')
        crowding_pick_flag = false;
        disable_flag = true;
    end
    if strcmp(problem, 'MaF12')
        crowding_pick_flag = false;
        disable_flag = true;
    end
    if strcmp(problem, 'MaF13')
        crowding_pick_flag = false;
        disable_flag = false;
        reserved_evaluations = 100000;
        normalization_str = '';
    end
    if strcmp(problem, 'MaF14')%OK
        crowding_pick_flag = false;
        disable_flag = true;
        normalization_str = '';
    end
    if strcmp(problem, 'MaF15')
        if norm(stable_threshold) < 1e-2
            stable_threshold = [20, 20, 20];
        end
        disable_flag = false;
        crowding_pick_flag = false;
        normalization_str = '';
        if Global.M == 5
            reserved_evaluations = 300000;
        elseif Global.M == 10
            reserved_evaluations = 1850000;
        end
    end
    Z = generate_reference([], 0, Global, -inf, 'normal');
    reference_generation_intial_flag = true;
    Population = Global.Initialization(); Archive = Population; Sample = Population;
end
```

### `normalization.m`
```matlab
function varargout = normalization(PopObj)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    global normalization_str
    if strcmp(normalization_str, 'normalize')
        [N, ~] = size(PopObj);
        a = intercepts(PopObj);
        PopObj = (PopObj - repmat(min(PopObj, [], 1), size(PopObj, 1), 1)) ./ repmat(a,N,1);
    end
    varargout{1} = PopObj;
    if nargout == 2
        varargout{2} = a;
    end
end

function a = intercepts(PopObj)
    [N, M] = size(PopObj);
    % Find the extreme points
    [~, Choosed(1:M)] = min(PopObj, [], 1);
    L2NormABO = zeros(N, M);
    for i = 1 : M
        L2NormABO(:, i) = sum(PopObj(:, [1: i - 1, i + 1: M]) .^ 2, 2);
    end
    [~, Choosed(M + 1: 2 * M)] = min(L2NormABO, [], 1);
    [~, Extreme] = max(PopObj(Choosed, :), [], 1);
    Extreme = unique(Choosed(Extreme));
    % Calculate the intercepts
    if length(Extreme) < M
        a = max(PopObj, [], 1);
    else
        lastwarn('');
        Hyperplane = mldivide(PopObj(Extreme,:), ones(M, 1));
        [~, msgid] = lastwarn();
        if strcmp(msgid, 'MATLAB:nearlySingularMatrix')
            % error('error in normalize');
        end
        a = 1 ./ Hyperplane';
    end
end
```

### `pair.m`
```matlab
function [distance, Allocation] = pair(O, Z, type)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    BLOCK_SIZE = 10000;
    blocks     = floor(size(Z, 1) / BLOCK_SIZE);
    B          = mat2cell(Z, [BLOCK_SIZE * ones(1, blocks), size(Z, 1) - BLOCK_SIZE * blocks], size(Z, 2));
    dA_set     = {};
    for i = 1 : numel(B)
        z         = B{i};
        dA        = NaN(size(O, 1), 2);
        cor_index = (i - 1) * BLOCK_SIZE + 1: min(i * BLOCK_SIZE, size(Z, 1));
        Cosine    = 1 - pdist2(O, z, 'cosine');
        if strcmp(type, 'distance')
            Error2Z = repmat(sqrt(sum(O .^ 2, 2)), 1, size(z, 1)) .* sqrt(1 - Cosine .^ 2);
        elseif strcmp(type, 'sin')
            Error2Z = sqrt(1 - Cosine .^ 2);
        end
        [dA(:, 1), dA(:, 2)] = min(Error2Z, [], 2);
        dA(:, 2)  = cor_index(dA(:, 2));
        dA_set{i} = dA;
    end
    d = NaN(size(O, 1), numel(B));
    A = NaN(size(O, 1), numel(B));
    for i = 1 : numel(B)
        dA = dA_set{i};
        d(:, i) = dA(:, 1); A(:, i) = dA(:, 2);
    end
    Allocation = NaN(size(O, 1), 1);
    [distance, I] = min(d, [], 2);
    for i = 1 : size(d, 1)
        Allocation(i) = A(i, I(i));
    end
end
```

### `update_archive.m`
```matlab
function archive = update_archive(archive, P, Z, picks, Problem)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    global disable_flag;
    if disable_flag
        return;
    end
    if ~isempty(archive)
        P = [archive, P];
    end
    [O, ia, ~] = unique([P.objs], 'rows');
    P          = P(ia);
    [index_frontiers, ~] = find_frontiers(O, [P.cons]);
    archive              = P(index_frontiers);
    O                    = [archive.objs];
    if numel(archive) > picks
        O = normalization(O);
        % Associate each frontier solution with one reference point
        [min_metric, allocation] = pair(O, Z, 'sin');
        % Identify the active reference lines
        index_active_clusters = unique(allocation);
        % calculate metric and sort the frontier solutions
        cluster.center       = [];
        cluster.select_queue = [];
        cluster              = repmat(cluster, numel(index_active_clusters), 1);
        for i = 1 : numel(index_active_clusters)
            Cluster_F_index = find(allocation == index_active_clusters(i));
            if numel(Cluster_F_index) == 1
                ascend_index = 1;
            else
                F_metric = min_metric(Cluster_F_index)';
                if numel(cluster) >= Problem.N
                    [~, ascend_index] = min(F_metric);
                else
                    [~, ascend_index] = sort(F_metric, 'ascend');
                end
            end
            Cluster_F_index   = Cluster_F_index(ascend_index);
            cluster(i).center = Cluster_F_index(1);
        end
        center_index = [cluster.center];
        archive      = [archive(center_index), crowding_pick(archive, picks, 'fast')];
        [~, ia, ~]   = unique([archive.objs], 'rows');
        archive      = archive(ia);
    end
end
```
