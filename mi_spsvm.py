"""mi-SPSVM for Multiple Instance Learning."""

import numpy as np
import matplotlib.pyplot as plt
import re
import sys


def load_octave_mat(filepath):
    """Parse Octave .mat file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {}
    
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


def solve_mi_spsvm_qp(X, J_plus, J_minus, C=1.0):
    """Solve QP, returns (v, gamma)."""
    try:
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
    except ImportError:
        return _solve_newton_fallback(X, J_plus, J_minus, C)
    
    n_features = X.shape[1]
    n_plus = len(J_plus)
    n_minus = len(J_minus)
    n_vars = n_features + 1 + n_minus
    
    X_plus = X[J_plus, :]
    A_plus = np.hstack([X_plus, -np.ones((n_plus, 1))])
    b_plus = np.ones(n_plus)
    AtA_plus = A_plus.T @ A_plus
    Atb_plus = A_plus.T @ b_plus
    
    P = np.zeros((n_vars, n_vars))
    P[:n_features+1, :n_features+1] = np.eye(n_features + 1) + C * AtA_plus
    
    q = np.zeros(n_vars)
    q[:n_features+1] = -C * Atb_plus
    q[n_features+1:] = C
    X_minus = X[J_minus, :]
    G1 = np.zeros((n_minus, n_vars))
    G1[:, :n_features] = X_minus
    G1[:, n_features] = -1
    G1[:, n_features+1:] = -np.eye(n_minus)
    h1 = -np.ones(n_minus)
    
    G2 = np.zeros((n_minus, n_vars))
    G2[:, n_features+1:] = -np.eye(n_minus)
    h2 = np.zeros(n_minus)
    
    G = np.vstack([G1, G2])
    h = np.hstack([h1, h2])
    sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))
    
    if sol['status'] != 'optimal':
        print(f"QP status: {sol['status']}")
    
    z = np.array(sol['x']).flatten()
    return z[:n_features], z[n_features]


def _solve_newton_fallback(X, J_plus, J_minus, C):
    """Fallback when cvxopt unavailable."""
    n_features = X.shape[1]
    w = np.zeros(n_features + 1)
    lr = 0.01
    
    for _ in range(1000):
        grad = np.copy(w)
        for j in J_plus:
            x_aug = np.append(X[j], -1)
            grad -= C * (1 - w @ x_aug) * x_aug
        
        for j in J_minus:
            x_aug = np.append(X[j], -1)
            if w @ x_aug < -1:
                grad += C * x_aug
        
        w -= lr * grad
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return w[:n_features], w[n_features]


class MiSPSVM:
    """mi-SPSVM classifier"""
    
    def __init__(self, C=1.0, max_iter=100, verbose=False):
        self.C = C
        self.max_iter = max_iter 
        self.verbose = verbose
        self.v = None
        self.gamma = 0.0
        self.history = []
    
    def fit(self, X, instance_bag, bag_labels):
        """Fit model. X: instances, instance_bag: bag IDs, bag_labels: ±1."""
        n_instances, n_bags = X.shape[0], len(bag_labels)
        pos_bags = [i+1 for i in range(n_bags) if bag_labels[i] == 1]
        neg_bags = [i+1 for i in range(n_bags) if bag_labels[i] == -1]
        
        if self.verbose:
            print(f"+ bags: {pos_bags}, - bags: {neg_bags}")
        
        J_plus = [j for j in range(n_instances) if int(instance_bag[j]) in pos_bags]
        J_minus = [j for j in range(n_instances) if int(instance_bag[j]) in neg_bags]
        
        if self.verbose:
            print(f"Init: |J+|={len(J_plus)}, |J-|={len(J_minus)}")
        
        for iteration in range(self.max_iter):
            v, gamma = solve_mi_spsvm_qp(X, J_plus, J_minus, self.C)
            self.history.append({'iter': iteration, 'J_plus': len(J_plus), 'J_minus': len(J_minus)})
            
            J_star = []
            for bag_id in pos_bags:
                bag_in_Jplus = [j for j in J_plus if int(instance_bag[j]) == bag_id]
                if not bag_in_Jplus:
                    continue
                scores = [float(v @ X[j] - gamma) for j in bag_in_Jplus]
                j_star = bag_in_Jplus[int(np.argmax(scores))]
                if scores[int(np.argmax(scores))] <= -1:
                    J_star.append(j_star)
            
            J_bar = [j for j in J_plus if j not in J_star and float(v @ X[j] - gamma) <= -1]
            
            if self.verbose:
                print(f"Iter {iteration}: |J*|={len(J_star)}, |J̄|={len(J_bar)}")
            
            if not J_bar:
                if self.verbose:
                    print(f"Converged at iter {iteration}")
                break
            
            J_plus = [j for j in J_plus if j not in J_bar]
            J_minus = J_minus + J_bar
        
        self.v = v
        self.gamma = gamma
        
        return self
    
    def decision_function(self, X):
        if self.v is None:
            raise ValueError("Model not fitted")
        return X @ self.v - self.gamma
    
    def predict_bags(self, X, instance_bag, n_bags):
        """Bag prediction via max-aggregation."""
        if self.v is None:
            raise ValueError("Not fitted")
        
        scores = self.decision_function(X)
        preds = np.zeros(n_bags)
        for bag_id in range(1, n_bags + 1):
            bag_scores = scores[instance_bag.flatten() == bag_id]
            preds[bag_id - 1] = 1 if len(bag_scores) > 0 and np.max(bag_scores) > 0 else -1
        return preds


def plot_results(X, instance_bag, bag_labels, model, save_path='mi_spsvm_results.png'):
    """Plot results with hyperplanes."""
    if model.v is None:
        raise ValueError("Not fitted")
    
    v = model.v
    gamma = model.gamma
    
    plt.figure(figsize=(12, 10))
    colors = ['blue', 'black', 'magenta', 'cyan', 'green', 'red', 'yellow']
    
    n_bags = len(bag_labels)
    
    for bag_id in range(1, n_bags + 1):
        bag_X = X[instance_bag.flatten() == bag_id]
        label, color = bag_labels[bag_id - 1], colors[bag_id - 1]
        
        if label == 1:
            plt.scatter(bag_X[:, 0], bag_X[:, 1], c=[color], marker='o', s=100,
                       edgecolors='black', linewidths=1.5, label=f'Bag {bag_id} (+)')
        else:
            plt.scatter(bag_X[:, 0], bag_X[:, 1], facecolors='none', edgecolors=color,
                       marker='o', s=100, linewidths=2, label=f'Bag {bag_id} (-)')
    x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
    
    if abs(v[1]) > 1e-10:
        x1 = np.linspace(x_min, x_max, 100)
        plt.plot(x1, (gamma - v[0]*x1) / v[1], 'k-', lw=2, label=r'$H(v,\gamma): v^T x = \gamma$')
        plt.plot(x1, (gamma+1 - v[0]*x1) / v[1], 'b--', lw=1.5, label=r'$H^+: v^T x = \gamma + 1$')
        plt.plot(x1, (gamma-1 - v[0]*x1) / v[1], 'r--', lw=1.5, label=r'$H^-: v^T x = \gamma - 1$')
    plt.xlim(x_min, x_max)
    plt.ylim(X[:, 1].min() - 50, X[:, 1].max() + 50)
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.title('mi-SPSVM Results')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


def compute_training_correctness(predictions, true_labels):
    return float(np.mean(predictions == true_labels))


def main():
    try:
        data = load_octave_mat('dataset79MIL.mat')
    except FileNotFoundError:
        print("dataset79MIL.mat not found")
        sys.exit(1)
    
    X = data['X']
    instance_bag = data['instanceBag'].flatten()
    y = data['y'].flatten()
    
    print(f"{X.shape[0]} instances, {len(y)} bags (+:{np.sum(y==1)}, -:{np.sum(y==-1)})")
    
    model = MiSPSVM(C=1.0, verbose=True)
    model.fit(X, instance_bag, y)
    
    print(f"v={model.v}, γ={model.gamma:.4f}")
    
    preds = model.predict_bags(X, instance_bag, len(y))
    acc = compute_training_correctness(preds, y)
    
    print(f"Accuracy: {acc*100:.1f}% | pred={preds.astype(int)} true={y.astype(int)}")
    plot_results(X, instance_bag, y, model)


if __name__ == "__main__":
    main()
