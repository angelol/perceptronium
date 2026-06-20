/// A deterministic Pseudo-Random Number Generator (PRNG) based on
/// a Linear Congruential Generator (LCG).
///
/// We write our own LCG so that we do not have to pull in external dependencies like `rand`,
/// and to guarantee that the neural network's weight initialization is 100% reproducible
/// across different platforms and runs.
pub struct Lcg {
    state: u32,
}

impl Lcg {
    /// Creates a new LCG generator with the specified seed.
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Generates the next pseudo-random 31-bit unsigned integer.
    pub fn next_u32(&mut self) -> u32 {
        // Classic POSIX LCG constants
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        self.state
    }

    /// Generates a pseudo-random f64 in the range `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / 2147483648.0
    }

    /// Generates a pseudo-random f64 in the range `[min, max)`.
    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

/// The Sigmoid activation function: \sigma(x) = 1 / (1 + e^-x)
///
/// Maps any real value into a probability range [0.0, 1.0].
/// Essential for our final classification layer.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the Sigmoid activation function.
///
/// Note: To save computation, this function expects the *already activated* value
/// `s = sigmoid(x)` rather than the raw pre-activation value `x`.
///
/// Since \sigma'(x) = \sigma(x) * (1 - \sigma(x)), passing the activated value
/// directly makes this a simple and extremely fast multiplication: `s * (1.0 - s)`.
pub fn sigmoid_derivative(activated_val: f64) -> f64 {
    activated_val * (1.0 - activated_val)
}

/// Computes the Binary Cross-Entropy (BCE) Loss.
///
/// Standard loss function for binary classification (cat vs dog).
/// BCE = -[y * ln(p) + (1 - y) * ln(1 - p)]
///
/// We clamp prediction `p` slightly to prevent `ln(0)` which would result in NaN or Inf.
pub fn binary_cross_entropy(prediction: f64, target: f64) -> f64 {
    let epsilon = 1e-15;
    let clamped_pred = prediction.clamp(epsilon, 1.0 - epsilon);
    -(target * clamped_pred.ln() + (1.0 - target) * (1.0 - clamped_pred).ln())
}

/// Computes the derivative of Binary Cross-Entropy Loss with respect to the prediction.
///
/// dLoss/dPrediction = (p - y) / (p * (1 - p))
#[allow(dead_code)]
pub fn binary_cross_entropy_derivative(prediction: f64, target: f64) -> f64 {
    let epsilon = 1e-15;
    let clamped_pred = prediction.clamp(epsilon, 1.0 - epsilon);
    (clamped_pred - target) / (clamped_pred * (1.0 - clamped_pred))
}

/// Computes the dot product of two vectors of equal length.
///
/// Sum of element-wise multiplications.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length for dot product.");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let act = sigmoid(0.0); // 0.5
        assert!((sigmoid_derivative(act) - 0.25).abs() < 1e-9); // 0.5 * (1 - 0.5) = 0.25
    }

    #[test]
    fn test_lcg_deterministic() {
        let mut lcg1 = Lcg::new(42);
        let mut lcg2 = Lcg::new(42);
        for _ in 0..10 {
            assert_eq!(lcg1.next_u32(), lcg2.next_u32());
            assert_eq!(lcg1.next_f64(), lcg2.next_f64());
        }
    }

    #[test]
    fn test_dot_product() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&v1, &v2), 32.0); // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    }
}
