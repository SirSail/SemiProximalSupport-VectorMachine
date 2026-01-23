function mi_spsvm()

% Load dataset
load dataset79MIL.mat;

[numInstances, dim] = size(X);
numBags = length(y);

% Identify positive and negative bags
posBags = find(y == 1);
negBags = find(y == -1);

printf("Instances: %d, Bags: %d (+:%d, -:%d)\n", numInstances, numBags, length(posBags), length(negBags));

% init J+ and J-
Jplus = [];
Jminus = [];

for j = 1:numInstances
    bagId = instanceBag(j);
    if any(posBags == bagId)
        Jplus = [Jplus; j];
    else
        Jminus = [Jminus; j];
    endif
end

printf("Init: |J+|=%d, |J-|=%d\n", length(Jplus), length(Jminus));

C = 1;
maxIter = 100;

for iteration = 0:maxIter-1
    
    % Step 1: Solve QP
    [v, gamma] = solveQP(X, Jplus, Jminus, C, dim);
    
    % Step 2: Witness selection
    Jstar = [];
    for i = 1:length(posBags)
        bagId = posBags(i);
        bagInJplus = Jplus(instanceBag(Jplus) == bagId);
        
        if isempty(bagInJplus)
            continue;
        endif
        
        scores = X(bagInJplus,:) * v - gamma;
        [maxScore, maxIdx] = max(scores);
        jstar = bagInJplus(maxIdx);
        
        if maxScore <= -1
            Jstar = [Jstar; jstar];
        endif
    end
    
    % Jbar: misclassified non-witnesses
    Jbar = [];
    for j = Jplus'
        if ~any(Jstar == j)
            score = X(j,:) * v - gamma;
            if score <= -1
                Jbar = [Jbar; j];
            endif
        endif
    end
    
    printf("Iter %d: |J*|=%d, |Jbar|=%d\n", iteration, length(Jstar), length(Jbar));
    
    % Stopping criterion
    if isempty(Jbar)
        printf("Converged at iter %d\n", iteration);
        break;
    endif
    
    % Step 3: Update J+ and J-
    Jplus = setdiff(Jplus, Jbar);
    Jminus = [Jminus; Jbar];
    
end

printf("\nv = [%.6f, %.6f]\n", v(1), v(2));
printf("gamma = %.6f\n", gamma);

% Predict bags using max-aggregation
predictions = zeros(numBags, 1);
for bagId = 1:numBags
    bagMask = (instanceBag == bagId);
    bagScores = X(bagMask,:) * v - gamma;
    if max(bagScores) > 0
        predictions(bagId) = 1;
    else
        predictions(bagId) = -1;
    endif
end

accuracy = mean(predictions == y) * 100;
printf("\nAccuracy: %.1f%%\n", accuracy);
printf("Pred: "); disp(predictions');
printf("True: "); disp(y');

% Plot results
plotResults(X, instanceBag, y, v, gamma, numBags);



end

function [v, gamma] = solveQP(X, Jplus, Jminus, C, dim)
    
    nPlus = length(Jplus);
    nMinus = length(Jminus);
    nVars = dim + 1 + nMinus;
    
    Xplus = X(Jplus, :);
    Aplus = [Xplus, -ones(nPlus, 1)];
    bplus = ones(nPlus, 1);
    
    AtA = Aplus' * Aplus;
    Atb = Aplus' * bplus;
    
    % Hessian P
    H = zeros(nVars, nVars);
    H(1:dim+1, 1:dim+1) = eye(dim+1) + C * AtA;
    
    q = zeros(nVars, 1);
    q(dim+2:end) = C;
    q(1:dim+1) = -C * Atb;
    
    Xminus = X(Jminus, :);
    
    % G1: v'x - gamma - xi <= -1
    G1 = zeros(nMinus, nVars);
    G1(:, 1:dim) = Xminus;
    G1(:, dim+1) = -1;
    G1(:, dim+2:end) = -eye(nMinus);
    h1 = -ones(nMinus, 1);
    
    G2 = zeros(nMinus, nVars);
    G2(:, dim+2:end) = -eye(nMinus);
    h2 = zeros(nMinus, 1);
    
    A_eq = [];
    b_eq = [];
    
    % Bounds
    lb = -Inf(nVars, 1);
    ub = Inf(nVars, 1);
    % lb(dim+2:end) = 0;  % handled by G2 now
    
    % Combine inequality constraints
    A_lb = -Inf(2*nMinus, 1);
    A_in = [G1; G2];
    A_ub = [h1; h2];
    
    % Solve QP using Octave's qp
    x0 = [];
    [z, obj, info] = qp(x0, H, q, A_eq, b_eq, lb, ub, A_lb, A_in, A_ub);
    
    if info.info ~= 0 && info.info ~= 3
        printf("QP Warning: info=%d\n", info.info);
    endif

    v = z(1:dim);
    gamma = z(dim+1);
    
endfunction


function plotResults(X, instanceBag, y, v, gamma, numBags)
    
    clf;
    hold on;
    % axis equal;
    
    colors = {'b', 'k', 'm', 'c', 'g', 'r', 'y'};
    
    legendHandles = [];
    legendLabels = {};
    
    for bagId = 1:numBags
        bagMask = (instanceBag == bagId);
        bagX = X(bagMask, :);
        
        if y(bagId) == 1
            % Positive: filled
            h = scatter(bagX(:,1), bagX(:,2), 100, colors{bagId}, 'filled', 'MarkerEdgeColor', 'k');
            legendLabels{end+1} = sprintf('Bag %d (+)', bagId);
        else
            % Negative: unfilled
            h = scatter(bagX(:,1), bagX(:,2), 100, colors{bagId}, 'LineWidth', 2);
            legendLabels{end+1} = sprintf('Bag %d (-)', bagId);
        endif
        legendHandles = [legendHandles, h];
    end
    
    % Hyperplanes
    xMin = min(X(:,1)) - 50;
    xMax = max(X(:,1)) + 50;
    x1 = linspace(xMin, xMax, 100);
    
    if abs(v(2)) > 1e-10
        % H: v'x = gamma
        x2_H = (gamma - v(1)*x1) / v(2);
        hH = plot(x1, x2_H, 'k-', 'LineWidth', 2);
        legendHandles = [legendHandles, hH];
        legendLabels{end+1} = 'H(v,\gamma)';
        
        % H+: v'x = gamma + 1
        x2_Hplus = (gamma + 1 - v(1)*x1) / v(2);
        hHplus = plot(x1, x2_Hplus, 'b--', 'LineWidth', 1.5);
        legendHandles = [legendHandles, hHplus];
        legendLabels{end+1} = 'H^+';
        
        % H-: v'x = gamma - 1
        x2_Hminus = (gamma - 1 - v(1)*x1) / v(2);
        hHminus = plot(x1, x2_Hminus, 'r--', 'LineWidth', 1.5);
        legendHandles = [legendHandles, hHminus];
        legendLabels{end+1} = 'H^-';
    endif
    
    legend(legendHandles, legendLabels, 'Location', 'bestoutside');
    
    xlim([xMin, xMax]);
    ylim([min(X(:,2))-50, max(X(:,2))+50]);
    xlabel('x_1');
    ylabel('x_2');
    title('mi-SPSVM Results');
    grid on;
    
    print('mi_spsvm_results_octave.png', '-dpng', '-r150');
    printf("Plot saved to mi_spsvm_results_octave.png\n");
    
endfunction
