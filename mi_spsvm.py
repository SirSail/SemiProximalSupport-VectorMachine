import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, cast
import sys

# Type aliases for clarity
Vector = np.ndarray
Matrix = np.ndarray

class SPSVM:
    """
    Smooth Proximal Support Vector Machine (SPSVM) - Primal solver.
    Minimizes: 0.5 * w'w + 0.5 * C * ||(1 - D(Aw - e*gamma))_+||^2
    Here we ignore bias 'b' (gamma) for simplicity or include it in weights.
    For this implementation, we will include bias in the weight vector by appending a 1 column to X.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 100, tol: float = 1e-3, verbose: bool = False) -> None:
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.w: Optional[Vector] = None

    def fit(self, X: Matrix, y: Vector) -> 'SPSVM':
        """
        X: (n_samples, n_features)
        y: (n_samples, )
        """
        n, d = X.shape
        # Augment X with 1s for bias term
        X_aug = np.hstack([X, np.ones((n, 1))])
        d_aug = d + 1
        
        # Initial weights
        w = np.zeros(d_aug)
        
        # D matrix (diagonal labeling matrix), effectively we can just multiply rows of X
        # Optimization objective function f(w)
        # Gradient grad_f(w)
        # Hessian hess_f(w)
        
        for i in range(self.max_iter):
            # Calculate margin: D(Xw) where D is diag(y)
            # Since y is {-1, 1}, we can just do y * (X @ w)
            linear_output = X_aug @ w
            margin = y * linear_output
            
            # Error vector e = (1 - margin)_+
            error_raw = 1 - margin
            error_vec = np.maximum(0, error_raw)
            
            # Objective function value (primal)
            # obj = 0.5 * w.T @ w + 0.5 * C * error_vec.T @ error_vec
            
            if np.linalg.norm(error_vec) < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i}")
                break

            # Gradient calculation
            # grad = w - C * X^T D * I_sv * (1 - D X w)
            # where I_sv is indicator of support vectors (error > 0)
            # Let's simplify:
            # The derivative of 0.5*C*||(1 - Dw^T x)||^2 is -C * sum_{sv} y_i * x_i * (1 - y_i w^T x_i)
            # which is -C * X^T * (y * error_vec) ? 
            # Actually simpler: let r = (1 - DyXw)_+. 
            # Loss = 0.5 * C * r^T r
            # Grad_Loss = - C * X^T D r
            # because d/dw (1 - y_i x_i^T w) = -y_i x_i
            
            # Support vector indices where 1 - y(wTx) > 0
            sv_indices = error_raw > 0
            
            # Gradient
            # term2 = X_aug.T @ (y * error_vec)  <-- wait, error_vec already has zeroes where not SV
            # But the sign: loss is increasing with error.
            # grad = w + C * X_aug.T @ ( -y * error_vec ) is INCORRECT derivation direction.
            # Correct derivation:
            # f(w) = 0.5 w'w + 0.5 C || (1 - D A w)_+ ||^2
            # Let v = (1 - D A w). P(v) = v_+
            # Grad = w + C * (d/dw v) * v_+
            # d/dw v = - (D A)^T
            # Grad = w - C * A^T D * v_+ 
            #      = w - C * A^T * (y * error_vec)
            
            grad = w - self.C * (X_aug.T @ (y * error_vec))
            
            # Hessian calculation (generalized Hessian)
            # H = I + C * X^T D I_sv D X
            # Since D*D = I, H = I + C * X_{sv}^T X_{sv}
            
            X_sv = X_aug[sv_indices]
            hess = np.eye(d_aug) + self.C * (X_sv.T @ X_sv)
            
            # Newton step: w_new = w - H^-1 grad
            # To avoid inversion, solve H d = -grad
            try:
                step = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if singular
                step = -0.01 * grad
            
            # Simple armijo line search could be added here, but full step usually works for SPSVM
            w = w + step
            
            if np.linalg.norm(step) < self.tol:
                if self.verbose:
                    print(f"Items converged at {i}")
                break
                
        self.w = w
        return self

    def predict(self, X: Matrix) -> Vector:
        if self.w is None:
            raise ValueError("Model not fitted")
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])
        scores = X_aug @ self.w
        return cast(Vector, np.sign(scores))

    def decision_function(self, X: Matrix) -> Vector:
        if self.w is None:
            raise ValueError("Model not fitted")
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])
        return cast(Vector, X_aug @ self.w)
    



class MISPSVM:
    """
    Multiple Instance Learning wrapper around SPSVM.
    Algorithm:
    1. Initialize: Assign y_i = Y_I for all instances in bag I.
    2. Loop:
       a. Train SPSVM on all instances with current labels.
       b. Update labels: For each positive bag, select instance with max score. 
          Set that to +1, others to -1. Negative bags all -1.
       c. Check convergence.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 10, verbose: bool = False) -> None:
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.svm = SPSVM(C=C, verbose=verbose)
        self.final_w: Optional[Vector] = None

    def fit(self, bags: List[Matrix], bag_labels: Vector) -> 'MISPSVM':
        """
        bags: List of numpy arrays, each (n_instances_in_bag, n_features)
        bag_labels: (n_bags, ) -1 or 1
        """
        # Flatten everything for easier indexing
        # But we need to keep track of which instance belongs to which bag
        
        # Initial labeling: propagate bag label to all instances
        instance_vars = []
        instance_bag_map = []
        
        current_labels = []
        
        for idx, (bag, label) in enumerate(zip(bags, bag_labels)):
            n_inst = bag.shape[0]
            instance_vars.append(bag)
            instance_bag_map.extend([idx] * n_inst)
            
            # Heuristic initialization: align with bag label
            current_labels.extend([label] * n_inst)
            
        X_all = np.vstack(instance_vars)
        y_all = np.array(current_labels, dtype=float)
        bag_map = np.array(instance_bag_map)
        
        # Indices of positive bags
        pos_bag_indices = np.where(bag_labels == 1)[0]
        
        prev_y = np.copy(y_all)
        
        for it in range(self.max_iter):
            # Step A: Train SVM
            self.svm.fit(X_all, y_all)
            
            # Step B: Re-assign labels
            # Calculate scores for all instances
            scores = self.svm.decision_function(X_all)
            
            # Reset positive bag instances to -1 first?
            # Standard mi-SVM:
            # For negative bags, all instances are -1 (fixed).
            # For positive bags, find instance with max score -> +1, others -> -1 ?
            # OR others -> remain valid?
            # "mi-SVM" formulation:
            # s.t. sum_{i in I} (y_i+1)/2 >= 1  (at least one positive)
            # Heuristic: set all y_i = sgn(w^T x_i), but ensure at least one is +1
            
            # Let's use the max-score heuristic:
            # For each positive bag:
            #   Find instance with max w^T x.
            #   Set its label to 1.
            #   Set others to sign(w^T x) ... or -1?
            #   Usually: others are determined by the classifier, but subject to the constraint.
            #   Common simple heuristic: select the 'witness' (max score) as +1. 
            #   What about the others? If we treat them as unlabeled, it's semi-supervised.
            #   If we treat them as negative, it's the "witness" approach.
            #   Let's stick to: "At least one is positive".
            #   Update rule: For pos bag, y_i = sgn(w^T x) EXCEPT if all are -1, force max to +1.
            
            new_y = np.copy(y_all)
            
            for b_idx in pos_bag_indices:
                # Get indices in X_all for this bag
                indices = np.where(bag_map == b_idx)[0]
                
                bag_scores = scores[indices]
                
                # Default update: just follow the classifier sign
                bag_inst_labels = np.sign(bag_scores)
                
                # Consistency check: At least one must be +1
                if not np.any(bag_inst_labels == 1):
                    # Force the max score instance to be positive
                    max_idx_local = np.argmax(bag_scores)
                    bag_inst_labels[max_idx_local] = 1.0
                
                new_y[indices] = bag_inst_labels
            
            # Negative bags are always all -1
            neg_indices = np.where(np.isin(bag_map, np.where(bag_labels == -1)[0]))[0]
            new_y[neg_indices] = -1.0
            
            # Check convergence
            if np.array_equal(new_y, prev_y):
                if self.verbose:
                    print(f"MI-SPSVM Converged at outer iter {it}")
                break
                
            y_all = new_y
            prev_y = np.copy(new_y)
            
        self.final_w = self.svm.w
        return self

    def predict_bag(self, bags: List[Matrix]) -> Vector:
        if self.final_w is None:
            raise ValueError("Not fitted")
            
        preds = []
        for bag in bags:
            # Max pooling assumption
            scores = self.svm.decision_function(bag)
            if np.max(scores) > 0:
                preds.append(1.0)
            else:
                preds.append(-1.0)
        return np.array(preds, dtype=np.float64)


def generate_data(n_bags: int = 50, seed: int = 42) -> Tuple[List[Matrix], Vector]:
    np.random.seed(seed)
    bags = []
    labels = []
    
    # Positive bags have at least one instance from Positive distribution
    # Negative bags have all instances from Negative distribution
    
    # Dist 1 (Neg): centered at (-2, -2)
    # Dist 2 (Pos): centered at (2, 2)
    
    for _ in range(n_bags):
        n_instances = np.random.randint(3, 10)
        label = 1 if np.random.rand() > 0.5 else -1
        
        instances = []
        if label == 1:
            # At least one positive instance
            n_pos = np.random.randint(1, n_instances + 1)
            # Some positive instances
            pos_inst = np.random.randn(n_pos, 2) + np.array([2, 2])
            instances.append(pos_inst)
            # Remaining negative
            if n_instances - n_pos > 0:
                neg_inst = np.random.randn(n_instances - n_pos, 2) + np.array([-2, -2])
                instances.append(neg_inst)
        else:
            # All negative
            instances.append(np.random.randn(n_instances, 2) + np.array([-2, -2]))
            
        bag_data = np.vstack(instances)
        bags.append(bag_data)
        labels.append(label)
        
    return bags, np.array(labels)


def main() -> None:
    # 1. Generate Data
    bags, y = generate_data(n_bags=50)
    
    # 2. Train
    print("Training MI-SPSVM...")
    model = MISPSVM(C=1.0, verbose=True)
    model.fit(bags, y)
    
    # 3. Predict / Test
    y_pred = model.predict_bag(bags)
    acc = np.mean(y_pred == y)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    
    if acc < 0.5:
        print("Error: Accuracy too low, something is wrong.")
        sys.exit(1)
        
    # 4. Plot
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    # Collect all points for background limits
    all_X = np.vstack(bags)
    x_min, x_max = all_X[:, 0].min() - 1, all_X[:, 0].max() + 1
    y_min, y_max = all_X[:, 1].min() - 1, all_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Plot decision boundary
    # We need to reshape grid for decision_function
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Use the underlying SVM to predict
    Z = model.svm.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=['red', 'blue'])
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    
    # Plot bags
    # Positive bags - plot instances
    # We color instances by their ground truth potential (heuristic based on location)
    # to see if the algorithm picked the right ones.
    
    for i, (bag, label) in enumerate(zip(bags, y)):
        if label == 1:
            plt.scatter(bag[:, 0], bag[:, 1], c='blue', marker='o', alpha=0.6, s=30, label='Pos Bag' if i == 0 else "")
            # Verify which was chosen as witness (highest score)
            scores = model.svm.decision_function(bag)
            max_idx = np.argmax(scores)
            plt.scatter(bag[max_idx, 0], bag[max_idx, 1], c='cyan', marker='*', s=100, edgecolors='k', label='Witness' if i == 0 else "")
        else:
            plt.scatter(bag[:, 0], bag[:, 1], c='red', marker='x', alpha=0.6, s=30, label='Neg Bag' if i == 0 else "")

    # Clean up legend
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title(f'MI-SPSVM Results (Acc: {acc*100:.1f}%)')
    plt.savefig('mi_spsvm_results.png')
    print("Plot saved to mi_spsvm_results.png")
    print("Success")

if __name__ == "__main__":
    main()
