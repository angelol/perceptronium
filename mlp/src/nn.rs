use crate::math::{dot_product, sigmoid, sigmoid_derivative, binary_cross_entropy, Lcg};

/// Represents our custom Multi-Layer Perceptron (MLP) neural network.
///
/// It features a single hidden layer of configurable size, designed for maximum
/// educational clarity and simplicity.
pub struct NeuralNetwork {
    /// Weights from input layer to hidden layer.
    /// `w_ih[i][j]` is the weight connecting input node `j` to hidden neuron `i`.
    w_ih: Vec<Vec<f64>>,
    /// Biases for each neuron in the hidden layer.
    b_h: Vec<f64>,
    /// Weights from hidden layer to output layer.
    /// Since we are doing binary classification (Cat vs Dog), the output layer has a
    /// single neuron. `w_ho[i]` is the weight connecting hidden neuron `i` to the output neuron.
    w_ho: Vec<f64>,
    /// Bias for the single output neuron.
    b_o: f64,
}

impl NeuralNetwork {
    /// Creates and initializes a new neural network using Xavier (Glorot) initialization.
    ///
    /// Xavier initialization is a standard deep learning best practice that initializes weights
    /// within a specific range based on the number of input and output nodes. This prevents
    /// the gradients from becoming too small (vanishing) or too large (exploding) at the start of training.
    pub fn new(input_size: usize, hidden_size: usize, seed: u32) -> Self {
        let mut prng = Lcg::new(seed);

        // Xavier Uniform Initialization limit: sqrt(6 / (fan_in + fan_out))
        // For Input -> Hidden
        let limit_ih = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let mut w_ih = vec![vec![0.0; input_size]; hidden_size];
        for i in 0..hidden_size {
            for j in 0..input_size {
                w_ih[i][j] = prng.next_range(-limit_ih, limit_ih);
            }
        }

        // Biases can start at 0.0 or small random values. We initialize them to 0.0
        let b_h = vec![0.0; hidden_size];

        // For Hidden -> Output (Output size is 1)
        let limit_ho = (6.0 / (hidden_size + 1) as f64).sqrt();
        let mut w_ho = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            w_ho[i] = prng.next_range(-limit_ho, limit_ho);
        }

        let b_o = 0.0;

        Self {
            w_ih,
            b_h,
            w_ho,
            b_o,
        }
    }

    /// Performs a forward propagation pass through the network.
    ///
    /// Takes the input vector and calculates the activations of the hidden layer,
    /// and then the activation of the final output neuron.
    /// Returns: `(hidden_activations, output_prediction)`
    pub fn forward(&self, inputs: &[f64]) -> (Vec<f64>, f64) {
        // 1. Calculate activations for the hidden layer
        let mut hidden_activations = vec![0.0; self.b_h.len()];
        for i in 0..self.b_h.len() {
            // z_h[i] = dot_product(weights, inputs) + bias
            let z_h = dot_product(&self.w_ih[i], inputs) + self.b_h[i];
            // h[i] = sigmoid(z_h[i])
            hidden_activations[i] = sigmoid(z_h);
        }

        // 2. Calculate activation for the single output neuron
        // z_o = dot_product(w_ho, hidden_activations) + b_o
        let z_o = dot_product(&self.w_ho, &hidden_activations) + self.b_o;
        // output = sigmoid(z_o)
        let output = sigmoid(z_o);

        (hidden_activations, output)
    }

    /// Performs a single step of Stochastic Gradient Descent (SGD) training.
    ///
    /// This includes:
    /// 1. Forward propagation to get predictions.
    /// 2. Backward propagation (backprop) to calculate gradients of loss w.r.t weights/biases.
    /// 3. Adjusting weights and biases using the learning rate.
    ///
    /// Returns the Binary Cross-Entropy Loss for this sample *before* the update.
    pub fn train_sample(&mut self, inputs: &[f64], target: f64, learning_rate: f64) -> f64 {
        // --- 1. Forward Propagation ---
        let (hidden_activations, output) = self.forward(inputs);

        // --- 2. Calculate Loss ---
        let loss = binary_cross_entropy(output, target);

        // --- 3. Backward Propagation (Backprop) ---

        // A. Output Layer Error Gradient (delta_o)
        // Mathematically, for Binary Cross-Entropy Loss combined with a Sigmoid activation,
        // the derivative of loss with respect to pre-activation output (z_o) simplifies beautifully:
        // delta_o = dL/dz_o = (output - target)
        // This holds true and avoids division by zero or numerical instability.
        let delta_o = output - target;

        // B. Output Layer Gradients
        // Gradient of output bias: db_o = delta_o
        let db_o = delta_o;
        // Gradient of output weights: dw_ho[i] = delta_o * hidden_activation[i]
        let mut dw_ho = vec![0.0; self.w_ho.len()];
        for i in 0..self.w_ho.len() {
            dw_ho[i] = delta_o * hidden_activations[i];
        }

        // C. Hidden Layer Error Gradients (delta_h)
        // We backpropagate the error from the output layer to the hidden layer.
        // For hidden neuron `i`:
        // delta_h[i] = delta_o * w_ho[i] * sigmoid_derivative(hidden_activations[i])
        let mut delta_h = vec![0.0; self.b_h.len()];
        for i in 0..self.b_h.len() {
            let d_activation = sigmoid_derivative(hidden_activations[i]);
            delta_h[i] = delta_o * self.w_ho[i] * d_activation;
        }

        // --- 4. Gradient Descent Weight Updates ---

        // Update output weights and bias
        self.b_o -= learning_rate * db_o;
        for i in 0..self.w_ho.len() {
            self.w_ho[i] -= learning_rate * dw_ho[i];
        }

        // Update hidden weights and biases
        for i in 0..self.b_h.len() {
            // Update bias for hidden neuron i
            self.b_h[i] -= learning_rate * delta_h[i];

            // Update weights from input nodes to hidden neuron i
            for j in 0..self.w_ih[i].len() {
                let dw_ih = delta_h[i] * inputs[j];
                self.w_ih[i][j] -= learning_rate * dw_ih;
            }
        }

        loss
    }

    /// Evaluates the network's prediction on an input, returning the final float value.
    ///
    /// 0.0 means highly confident Cat, 1.0 means highly confident Dog.
    pub fn predict(&self, inputs: &[f64]) -> f64 {
        let (_, output) = self.forward(inputs);
        output
    }

    /// Saves the current model weights and biases to a structured, human-readable text file.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), String> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create weights file: {}", e))?;

        let input_size = self.w_ih[0].len();
        let hidden_size = self.b_h.len();

        // Write dimensions
        writeln!(file, "{},{}", input_size, hidden_size).map_err(|e| e.to_string())?;
        // Write output bias
        writeln!(file, "{}", self.b_o).map_err(|e| e.to_string())?;

        // Write hidden biases
        let b_h_str: Vec<String> = self.b_h.iter().map(|f| f.to_string()).collect();
        writeln!(file, "{}", b_h_str.join(",")).map_err(|e| e.to_string())?;

        // Write output weights
        let w_ho_str: Vec<String> = self.w_ho.iter().map(|f| f.to_string()).collect();
        writeln!(file, "{}", w_ho_str.join(",")).map_err(|e| e.to_string())?;

        // Write input-to-hidden weights rows
        for row in &self.w_ih {
            let row_str: Vec<String> = row.iter().map(|f| f.to_string()).collect();
            writeln!(file, "{}", row_str.join(",")).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Loads model weights and biases from a saved structured text file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open weights file: {}", e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // 1. Parse Dimensions
        let first_line = lines.next().ok_or("Missing size metadata")?.map_err(|e| e.to_string())?;
        let sizes: Vec<usize> = first_line
            .split(',')
            .map(|s| s.parse().map_err(|e| format!("Invalid size: {}", e)))
            .collect::<Result<Vec<usize>, String>>()?;
        if sizes.len() != 2 {
            return Err("Invalid size metadata format".to_string());
        }
        let input_size = sizes[0];
        let hidden_size = sizes[1];

        // 2. Parse Output Bias
        let b_o_line = lines.next().ok_or("Missing output bias")?.map_err(|e| e.to_string())?;
        let b_o: f64 = b_o_line.trim().parse().map_err(|e| format!("Invalid output bias: {}", e))?;

        // 3. Parse Hidden Biases
        let b_h_line = lines.next().ok_or("Missing hidden biases")?.map_err(|e| e.to_string())?;
        let b_h: Vec<f64> = b_h_line
            .split(',')
            .map(|s| s.parse().map_err(|e| format!("Invalid hidden bias: {}", e)))
            .collect::<Result<Vec<f64>, String>>()?;
        if b_h.len() != hidden_size {
            return Err(format!("Mismatch in hidden biases (expected {}, got {})", hidden_size, b_h.len()));
        }

        // 4. Parse Output Weights
        let w_ho_line = lines.next().ok_or("Missing output weights")?.map_err(|e| e.to_string())?;
        let w_ho: Vec<f64> = w_ho_line
            .split(',')
            .map(|s| s.parse().map_err(|e| format!("Invalid output weight: {}", e)))
            .collect::<Result<Vec<f64>, String>>()?;
        if w_ho.len() != hidden_size {
            return Err(format!("Mismatch in output weights (expected {}, got {})", hidden_size, w_ho.len()));
        }

        // 5. Parse Input-Hidden Weights Rows
        let mut w_ih = Vec::new();
        for _ in 0..hidden_size {
            let row_line = lines.next().ok_or("Missing input-hidden weights row")?.map_err(|e| e.to_string())?;
            let row: Vec<f64> = row_line
                .split(',')
                .map(|s| s.parse().map_err(|e| format!("Invalid weight: {}", e)))
                .collect::<Result<Vec<f64>, String>>()?;
            if row.len() != input_size {
                return Err(format!("Mismatch in input weights row size (expected {}, got {})", input_size, row.len()));
            }
            w_ih.push(row);
        }

        Ok(Self {
            w_ih,
            b_h,
            w_ho,
            b_o,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_forward_dimensions() {
        let input_size = 100;
        let hidden_size = 10;
        let nn = NeuralNetwork::new(input_size, hidden_size, 1337);

        let inputs = vec![0.5; input_size];
        let (hidden, output) = nn.forward(&inputs);

        assert_eq!(hidden.len(), hidden_size);
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_backpropagation_decreases_loss() {
        // Tests that training on a single sample repeatedly decreases the loss
        let mut nn = NeuralNetwork::new(16, 4, 12345);
        let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let target = 1.0; // Say this is a dog
        let learning_rate = 0.5;

        let initial_loss = nn.train_sample(&inputs, target, learning_rate);
        let mut last_loss = initial_loss;

        for _ in 0..10 {
            let current_loss = nn.train_sample(&inputs, target, learning_rate);
            assert!(current_loss <= last_loss, "Loss did not decrease (current: {}, last: {})", current_loss, last_loss);
            last_loss = current_loss;
        }

        assert!(last_loss < initial_loss);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let input_size = 64;
        let hidden_size = 8;
        let seed = 123;
        let mut nn = NeuralNetwork::new(input_size, hidden_size, seed);
        
        // Train a bit so values are non-zero and non-trivial
        let inputs = vec![0.5; input_size];
        nn.train_sample(&inputs, 1.0, 0.1);

        // Run forward pass to get some outputs to compare later
        let (hidden_orig, output_orig) = nn.forward(&inputs);

        // Save to a temporary file
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("perceptronium_test_weights.txt");
        
        nn.save_to_file(&test_path).expect("Failed to save weights");

        // Load back into a new network
        let nn_loaded = NeuralNetwork::load_from_file(&test_path).expect("Failed to load weights");

        // Clean up the file
        let _ = std::fs::remove_file(&test_path);

        // Verify dimensions and parameters match
        assert_eq!(nn_loaded.b_o, nn.b_o);
        assert_eq!(nn_loaded.b_h, nn.b_h);
        assert_eq!(nn_loaded.w_ho, nn.w_ho);
        assert_eq!(nn_loaded.w_ih, nn.w_ih);

        // Verify that predictions match precisely
        let (hidden_loaded, output_loaded) = nn_loaded.forward(&inputs);
        assert_eq!(hidden_orig, hidden_loaded);
        assert_eq!(output_orig, output_loaded);
    }
}

