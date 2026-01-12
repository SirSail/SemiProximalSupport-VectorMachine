"""
MI-SPSVM: Semi-Proximal Support Vector Machine for Multiple Instance Learning

This implementation follows the exact algorithm specification from project79.pdf:
- Mixed L2 loss (positive) / L1 loss (negative)
- Proper J+/J- set iteration scheme
- cvxopt QP solver

Author: Optimization for ML Course Project 79
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, cast
import re
import sys

# Type aliases
Vector = np.ndarray
Matrix = np.ndarray


def load_octave_mat(filepath: str) -> dict[str, np.ndarray]:
    """Parse Octave text format .mat file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    data: dict[str, np.ndarray] = {}
    
    # Pattern to match variable blocks
    pattern = r'# name: (\w+)\n# type: (\w+)\n(?:# rows: (\d+)\n# columns: (\d+)\n)?([\s\S]*?)(?=\n\n# name:|$)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        name, dtype, rows, cols, values = match
        values = values.strip()
        
        if dtype == 'scalar':
            data[name] = np.array([float(values)])
        elif dtype == 'matrix':
            rows_int = int(rows)
            cols_int = int(cols)
            numbers = [float(x) for x in values.split()]
            arr = np.array(numbers).reshape(rows_int, cols_int)
            data[name] = arr
    
    return data


def solve_mi_spsvm_qp(
    X: Matrix,
    J_plus: list[int],
    J_minus: list[int],
    C: float = 1.0
) -> Tuple[Vector, float]:
    """
    Solve the mi-SPSVM optimization problem:
    
    min  0.5 * ||(v, gamma)||^2 + (C/2) * sum_{j in J+} xi_j^2 + C * sum_{j in J-} xi_j
    s.t. xi_j = 1 - (v^T x_j - gamma)   for j in J+
         xi_j >= 1 + (v^T x_j - gamma)  for j in J-
         xi_j >= 0                      for j in J-
    
    This is a QP problem. We reformulate:
    - For J+: equality constraint, L2 penalty -> substitute xi into objective
    - For J-: inequality constraints, L1 penalty -> use slack formulation
    
    Returns:
        v: weight vector (n_features,)
        gamma: bias term
    """
    try:
        from cvxopt import matrix, solvers  # type: ignore[import-untyped]
        solvers.options['show_progress'] = False
    except ImportError:
        print("cvxopt not installed. Using fallback Newton solver.")
        return _solve_newton_fallback(X, J_plus, J_minus, C)
    
    n_features = X.shape[1]  # 2
    n_plus = len(J_plus)
    n_minus = len(J_minus)
    
    # Decision variables: [v (n_features), gamma (1), xi_minus (n_minus)]
    # Total: n_features + 1 + n_minus
    n_vars = n_features + 1 + n_minus
    
    # Build the QP matrices
    # Objective: 0.5 * z^T P z + q^T z
    
    # For positive instances, substitute xi_j = 1 - (v^T x_j - gamma)
    # Loss_plus = (C/2) * sum_j (1 - v^T x_j + gamma)^2
    # This adds quadratic terms in (v, gamma)
    
    # Let's expand: sum_j (1 - v^T x_j + gamma)^2
    # = sum_j (1 + gamma - v^T x_j)^2
    # = sum_j [(1+gamma)^2 - 2(1+gamma)(v^T x_j) + (v^T x_j)^2]
    
    # Build X_plus: (n_plus, n_features)
    X_plus = X[J_plus, :]  # Shape: (n_plus, n_features)
    
    # The quadratic part from positive instances:
    # (C/2) * v^T (X_plus^T X_plus) v - C * v^T X_plus^T (1+gamma) + (C/2)*n_plus*(1+gamma)^2
    # But since gamma is also a variable, we need to be careful
    
    # Let's use augmented representation for cleaner formulation
    # Define z = [v; gamma; xi_minus]
    # 
    # For positive loss (L2):
    # xi_j^+ = 1 - v^T x_j + gamma
    # Loss_plus = (C/2) * ||e - X_plus @ v + gamma * 1||^2
    #           where e = ones(n_plus)
    # 
    # Let A_plus = [X_plus, -ones(n_plus,1)]  (shape: n_plus x (n+1))
    # Then xi_plus = e - A_plus @ [v; gamma] = e + [-X_plus, ones] @ [v; gamma]
    # Wait, let me be more careful with signs.
    # xi_j = 1 - (v^T x_j - gamma) = 1 - v^T x_j + gamma
    
    # Let's define:
    # A = [X_plus, -ones]  shape (n_plus, n+1)
    # b = ones(n_plus)
    # xi_plus = b - A @ [v; gamma] with A = [X_plus, -ones]
    # = 1 - v^T x_j + gamma  ✓
    
    A_plus = np.hstack([X_plus, -np.ones((n_plus, 1))])  # (n_plus, n+1)
    b_plus = np.ones(n_plus)
    
    # Loss_plus = (C/2) * ||b_plus - A_plus @ w||^2  where w = [v; gamma]
    # = (C/2) * [w^T A_plus^T A_plus w - 2 b_plus^T A_plus w + b_plus^T b_plus]
    
    # Quadratic term from positive loss:
    AtA_plus = A_plus.T @ A_plus  # (n+1, n+1)
    Atb_plus = A_plus.T @ b_plus  # (n+1,)
    
    # Build P matrix (Hessian)
    # P = diag([1,...,1, 1, 0,...,0]) + C * [AtA_plus, 0; 0, 0]
    #     ^-- v      ^-- gamma ^-- xi_minus
    
    P = np.zeros((n_vars, n_vars))
    # Regularization: 0.5 * (v^T v + gamma^2) -> I for first (n+1) vars
    P[:n_features+1, :n_features+1] = np.eye(n_features + 1)
    # Positive loss quadratic term
    P[:n_features+1, :n_features+1] += C * AtA_plus
    
    # Build q vector (linear term)
    # q = [0,...,0] - C * [Atb_plus; 0,...,0]
    # Actually for quadratic expansion: -C * Atb_plus for (v, gamma)
    # And +C for each xi_minus (L1 penalty)
    q = np.zeros(n_vars)
    q[:n_features+1] = -C * Atb_plus
    q[n_features+1:] = C  # L1 penalty coefficients for xi_minus
    
    # Inequality constraints: Gz <= h
    # For j in J-:
    #   xi_j >= 1 + v^T x_j - gamma   =>  -xi_j + v^T x_j - gamma <= -1
    #   xi_j >= 0                      =>  -xi_j <= 0
    
    X_minus = X[J_minus, :]  # (n_minus, n_features)
    
    # First set of constraints: -xi + v^T x - gamma <= -1
    # Variables: [v, gamma, xi_minus]
    # G1 @ z = [X_minus, -ones, -I] @ [v; gamma; xi] <= -1
    G1 = np.zeros((n_minus, n_vars))
    G1[:, :n_features] = X_minus
    G1[:, n_features] = -1  # gamma coefficient
    G1[:, n_features+1:] = -np.eye(n_minus)  # xi coefficients
    h1 = -np.ones(n_minus)
    
    # Second set: -xi <= 0
    G2 = np.zeros((n_minus, n_vars))
    G2[:, n_features+1:] = -np.eye(n_minus)
    h2 = np.zeros(n_minus)
    
    G = np.vstack([G1, G2])
    h = np.hstack([h1, h2])
    
    # Convert to cvxopt format
    P_cvx = matrix(P, tc='d')
    q_cvx = matrix(q, tc='d')
    G_cvx = matrix(G, tc='d')
    h_cvx = matrix(h, tc='d')
    
    # Solve QP
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    
    if sol['status'] != 'optimal':
        print(f"Warning: QP solver status = {sol['status']}")
    
    z = np.array(sol['x']).flatten()
    v = z[:n_features]
    gamma = z[n_features]
    
    return v, gamma


def _solve_newton_fallback(
    X: Matrix,
    J_plus: list[int],
    J_minus: list[int],
    C: float
) -> Tuple[Vector, float]:
    """Fallback Newton solver if cvxopt is not available."""
    n_features = X.shape[1]
    
    # Simple gradient descent on smooth approximation
    w = np.zeros(n_features + 1)  # [v; gamma]
    lr = 0.01
    
    for _ in range(1000):
        # Positive loss gradient (L2)
        grad = np.copy(w)
        for j in J_plus:
            x_aug = np.append(X[j], -1)  # [x; -1] for v^T x - gamma
            xi = 1 - w @ x_aug
            grad -= C * xi * x_aug
        
        # Negative loss gradient (L1 -> subgradient)
        for j in J_minus:
            x_aug = np.append(X[j], -1)
            margin = w @ x_aug  # v^T x - gamma
            if margin < -1:  # xi = 1 + margin > 0
                grad += C * x_aug
        
        w -= lr * grad
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return w[:n_features], w[n_features]


class MiSPSVM:
    """
    mi-SPSVM: Semi-Proximal SVM for Multiple Instance Learning.
    
    Implements the exact algorithm from project79.pdf:
    - Step 0: Initialize J+ and J-
    - Step 1: Solve semiproximal QP
    - Step 2: Stopping criterion (witness selection)
    - Step 3: Update J+ and J-
    - Step 4: Iterate
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 100, verbose: bool = False) -> None:
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.v: Vector | None = None
        self.gamma: float = 0.0
        self.history: list[dict[str, object]] = []
    
    def fit(
        self,
        X: Matrix,
        instance_bag: Vector,
        bag_labels: Vector
    ) -> 'MiSPSVM':
        """
        Fit the mi-SPSVM model.
        
        Args:
            X: (n_instances, n_features) instance feature matrix
            instance_bag: (n_instances,) bag assignment for each instance
            bag_labels: (n_bags,) labels for each bag (+1 or -1)
        """
        n_instances = X.shape[0]
        n_bags = len(bag_labels)
        
        # Identify positive and negative bags
        pos_bags = [i+1 for i in range(n_bags) if bag_labels[i] == 1]  # 1-indexed
        neg_bags = [i+1 for i in range(n_bags) if bag_labels[i] == -1]
        
        if self.verbose:
            print(f"Positive bags: {pos_bags}")
            print(f"Negative bags: {neg_bags}")
        
        # Step 0: Initialize J+ and J-
        # J+ = all instances in positive bags
        # J- = all instances in negative bags
        J_plus: list[int] = []
        J_minus: list[int] = []
        
        for j in range(n_instances):
            bag_id = int(instance_bag[j])
            if bag_id in pos_bags:
                J_plus.append(j)
            else:
                J_minus.append(j)
        
        if self.verbose:
            print(f"Initial J+ size: {len(J_plus)}, J- size: {len(J_minus)}")
        
        # Main iteration loop
        for iteration in range(self.max_iter):
            # Step 1: Solve optimization problem
            v, gamma = solve_mi_spsvm_qp(X, J_plus, J_minus, self.C)
            
            self.history.append({
                'iter': iteration,
                'J_plus_size': len(J_plus),
                'J_minus_size': len(J_minus)
            })
            
            # Step 2: Stopping criterion
            # For each positive bag i, find witness j*_i = argmax_{j in J+_i ∩ J+} (v^T x_j - gamma)
            # Let J* = {j*_i | v^T x_{j*_i} - gamma <= -1}
            # Let J_bar = {j in J+ \ J* | v^T x_j - gamma <= -1}
            # If J_bar = empty, STOP
            
            J_star: list[int] = []
            J_bar: list[int] = []
            
            for bag_id in pos_bags:
                # Get instances in this bag that are still in J+
                bag_instances_in_Jplus = [j for j in J_plus if int(instance_bag[j]) == bag_id]
                
                if not bag_instances_in_Jplus:
                    continue
                
                # Find witness: instance with max score
                scores = [float(v @ X[j] - gamma) for j in bag_instances_in_Jplus]
                max_idx = int(np.argmax(scores))
                j_star = bag_instances_in_Jplus[max_idx]
                
                # Check if witness is misclassified
                if scores[max_idx] <= -1:
                    J_star.append(j_star)
            
            # J_bar = instances in J+ (not in J*) with score <= -1
            for j in J_plus:
                if j not in J_star:
                    score = float(v @ X[j] - gamma)
                    if score <= -1:
                        J_bar.append(j)
            
            if self.verbose:
                print(f"Iter {iteration}: |J*|={len(J_star)}, |J_bar|={len(J_bar)}")
            
            # Stopping criterion
            if len(J_bar) == 0:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Step 3: Update J+ and J-
            # J+ := J+ \ J_bar
            # J- := J- ∪ J_bar
            J_plus = [j for j in J_plus if j not in J_bar]
            J_minus = J_minus + J_bar
        
        self.v = v
        self.gamma = gamma
        
        return self
    
    def decision_function(self, X: Matrix) -> Vector:
        """Compute decision function value: v^T x - gamma."""
        if self.v is None:
            raise ValueError("Model not fitted")
        return cast(Vector, X @ self.v - self.gamma)
    
    def predict_bags(self, X: Matrix, instance_bag: Vector, n_bags: int) -> Vector:
        """Predict bag labels using max-aggregation."""
        if self.v is None:
            raise ValueError("Model not fitted")
        
        scores = self.decision_function(X)
        predictions = np.zeros(n_bags)
        
        for bag_id in range(1, n_bags + 1):
            bag_mask = instance_bag.flatten() == bag_id
            bag_scores = scores[bag_mask]
            if len(bag_scores) > 0:
                predictions[bag_id - 1] = 1 if np.max(bag_scores) > 0 else -1
            else:
                predictions[bag_id - 1] = -1
        
        return predictions


def plot_results(
    X: Matrix,
    instance_bag: Vector,
    bag_labels: Vector,
    model: MiSPSVM,
    save_path: str = 'mi_spsvm_results.png'
) -> None:
    """
    Plot the results according to project specification:
    - Instances colored by bag (filled for positive, unfilled for negative)
    - Separating hyperplane H(v, gamma): v^T x = gamma
    - Margin hyperplanes H+: v^T x = gamma + 1 and H-: v^T x = gamma - 1
    """
    if model.v is None:
        raise ValueError("Model not fitted")
    
    v = model.v
    gamma = model.gamma
    
    plt.figure(figsize=(12, 10))
    
    # Color palette for bags
    cmap = plt.colormaps.get_cmap('tab10')  # type: ignore[attr-defined]
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    
    n_bags = len(bag_labels)
    
    # Plot instances
    for bag_id in range(1, n_bags + 1):
        bag_mask = instance_bag.flatten() == bag_id
        bag_X = X[bag_mask]
        label = bag_labels[bag_id - 1]
        color = colors[bag_id - 1]
        
        if label == 1:
            # Positive bag: filled circles
            plt.scatter(
                bag_X[:, 0], bag_X[:, 1],
                c=[color], marker='o', s=100,
                edgecolors='black', linewidths=1.5,
                label=f'Bag {bag_id} (+)'
            )
        else:
            # Negative bag: unfilled circles
            plt.scatter(
                bag_X[:, 0], bag_X[:, 1],
                facecolors='none', edgecolors=color,
                marker='o', s=100, linewidths=2,
                label=f'Bag {bag_id} (-)'
            )
    
    # Plot hyperplanes
    x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
    
    # H(v, gamma): v1*x1 + v2*x2 = gamma  =>  x2 = (gamma - v1*x1) / v2
    if abs(v[1]) > 1e-10:
        x1_line = np.linspace(x_min, x_max, 100)
        
        # Separating hyperplane
        x2_sep = (gamma - v[0] * x1_line) / v[1]
        plt.plot(x1_line, x2_sep, 'k-', linewidth=2, label=r'$H(v,\gamma): v^T x = \gamma$')
        
        # H+: v^T x = gamma + 1
        x2_plus = (gamma + 1 - v[0] * x1_line) / v[1]
        plt.plot(x1_line, x2_plus, 'b--', linewidth=1.5, label=r'$H^+: v^T x = \gamma + 1$')
        
        # H-: v^T x = gamma - 1
        x2_minus = (gamma - 1 - v[0] * x1_line) / v[1]
        plt.plot(x1_line, x2_minus, 'r--', linewidth=1.5, label=r'$H^-: v^T x = \gamma - 1$')
    
    # Set axis limits
    y_min, y_max = X[:, 1].min() - 50, X[:, 1].max() + 50
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title('mi-SPSVM Results', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def compute_training_correctness(
    predictions: Vector,
    true_labels: Vector
) -> float:
    """Compute training correctness (accuracy)."""
    return float(np.mean(predictions == true_labels))


def main() -> None:
    """Main function to run mi-SPSVM on the provided dataset."""
    
    # Load dataset
    print("Loading dataset...")
    try:
        data = load_octave_mat('dataset79MIL.mat')
    except FileNotFoundError:
        print("Error: dataset79MIL.mat not found!")
        sys.exit(1)
    
    X = data['X']  # (35, 2)
    instance_bag = data['instanceBag'].flatten()  # (35,)
    y = data['y'].flatten()  # (7,)
    
    print(f"Dataset: {X.shape[0]} instances, {len(y)} bags")
    print(f"Positive bags: {np.sum(y == 1)}, Negative bags: {np.sum(y == -1)}")
    
    # Train model
    print("\nTraining mi-SPSVM...")
    model = MiSPSVM(C=1.0, verbose=True)
    model.fit(X, instance_bag, y)
    
    print("\nLearned parameters:")
    print(f"  v = {model.v}")
    print(f"  gamma = {model.gamma:.6f}")
    
    # Predict and compute training correctness
    predictions = model.predict_bags(X, instance_bag, len(y))
    accuracy = compute_training_correctness(predictions, y)
    
    print(f"\nTraining Correctness: {accuracy * 100:.2f}%")
    print(f"Predictions: {predictions.astype(int)}")
    print(f"True labels: {y.astype(int)}")
    
    # Plot results
    print("\nGenerating plot...")
    plot_results(X, instance_bag, y, model)
    
    print("\nSuccess!")


if __name__ == "__main__":
    main()
