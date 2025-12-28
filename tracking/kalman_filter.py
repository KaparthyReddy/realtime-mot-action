import numpy as np

class KalmanFilter:
    """
    Highly stable Kalman Filter for MOT.
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    def __init__(self, dt=1.0):
        self.dt = dt
        self.state_dim = 8
        self.meas_dim = 4

        # Transition matrix F
        self.F = np.eye(self.state_dim)
        for i in range(4):
            self.F[i, 4 + i] = self.dt

        # Measurement matrix H
        self.H = np.eye(self.meas_dim, self.state_dim)

        # Noise weights (Standard DeepSORT-inspired scaling)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self.x = None
        self.P = None

    def init_state(self, measurement):
        """Initialize state with height-dependent noise."""
        self.x = np.zeros(self.state_dim)
        self.x[:self.meas_dim] = measurement
        
        # Scale initial uncertainty by object height
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2, 
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h
        ]
        self.P = np.diag(np.square(std))

    def predict(self):
        if self.x is None: return
        
        # Adaptive Process Noise Q
        h = self.x[3]
        std_pos = self._std_weight_position * h
        std_vel = self._std_weight_velocity * h
        std = [std_pos, std_pos, 1e-2, std_pos, std_vel, std_vel, 1e-5, std_vel]
        Q = np.diag(np.square(std))

        self.x = np.dot(self.F, self.x)
        # P = FPF' + Q
        self.P = self.F @ self.P @ self.F.T + Q
        
        # --- Numerical Stability Regularization ---
        # Ensure symmetry
        self.P = (self.P + self.P.T) * 0.5
        # Add a tiny 'tick' of uncertainty to prevent collapse
        self.P += np.eye(self.state_dim) * 1e-6

        # Numerical Safety Clip
        if np.any(np.abs(self.P) > 1e6):
            self.P = np.clip(self.P, -1e6, 1e6)
            
        return self.x.copy()

    def update(self, measurement):
        if self.x is None:
            self.init_state(measurement)
            return self.x.copy()

        # Adaptive Measurement Noise R
        h = measurement[3]
        std_pos = self._std_weight_position * h
        R = np.diag(np.square([std_pos, std_pos, 1e-2, std_pos]))

        # Innovation
        y = measurement - np.dot(self.H, self.x)
        # S = HPH' + R
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman Gain using pseudo-inverse for stability
        K = self.P @ self.H.T @ np.linalg.pinv(S)

        # Update State
        self.x = self.x + np.dot(K, y)
        
        # Update Covariance (Joseph Form)
        I = np.eye(self.state_dim)
        I_KH = I - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # --- Numerical Stability Regularization ---
        # Ensure symmetry
        self.P = (self.P + self.P.T) * 0.5
        # Add a tiny 'tick' of uncertainty
        self.P += np.eye(self.state_dim) * 1e-6
        
        return self.x.copy()

    def get_predicted_bbox(self):
        return self.x[:4].copy() if self.x is not None else None