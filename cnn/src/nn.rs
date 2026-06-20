use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use crate::math::Lcg;

// Upgraded Hierarchical 2-Layer CNN Architecture Constants
pub const FILTERS1: usize = 8;        // 8 filters in the first convolutional layer
pub const FILTERS2: usize = 16;       // 16 filters in the second convolutional layer
pub const DENSE_NEURONS: usize = 64;  // 64 hidden neurons in the fully connected dense classifier
pub const FLATTENED_SIZE: usize = 30 * 30 * FILTERS2; // 30x30x16 = 14,400 flattened features
pub const L2_REG: f64 = 0.001;        // L2 Regularization/Weight Decay coefficient

pub struct ConvolutionalNetwork {
    // Conv Layer 1 parameters: FILTERS1 filters of size 3x3
    pub kernels1: Vec<Vec<f64>>,
    pub conv_biases1: Vec<f64>,

    // Conv Layer 2 parameters: FILTERS2 filters, each convolving FILTERS1 input channels with 3x3 kernels
    pub kernels2: Vec<Vec<Vec<f64>>>, // FILTERS2 x FILTERS1 x 9
    pub conv_biases2: Vec<f64>,

    // Dense hidden classifier layer parameters: DENSE_NEURONS neurons with FLATTENED_SIZE inputs each
    pub weights_dense: Vec<Vec<f64>>,
    pub biases_dense: Vec<f64>,

    // Output classifier parameters: 1 neuron with DENSE_NEURONS inputs
    pub weights_out: Vec<f64>,
    pub bias_out: f64,

    // Intermediate activations caches (populated during forward pass, used in backward pass)
    conv1_out: Vec<Vec<f64>>,       // FILTERS1 maps of size 126x126
    conv1_activated: Vec<Vec<f64>>, // FILTERS1 maps of size 126x126
    pool1_out: Vec<Vec<f64>>,       // FILTERS1 maps of size 63x63
    pool1_indices: Vec<Vec<(usize, usize)>>, // Tracks exact max value indices in 2x2 windows

    conv2_out: Vec<Vec<f64>>,       // FILTERS2 maps of size 61x61
    conv2_activated: Vec<Vec<f64>>, // FILTERS2 maps of size 61x61
    pool2_out: Vec<Vec<f64>>,       // FILTERS2 maps of size 30x30
    pool2_indices: Vec<Vec<(usize, usize)>>, // Tracks exact max value indices in 2x2 windows

    flattened: Vec<f64>,           // Size FLATTENED_SIZE (14,400)
    dense_out: Vec<f64>,           // Size DENSE_NEURONS (64)
    dense_activated: Vec<f64>,     // Size DENSE_NEURONS (64, activated using ReLU!)
    output_out: f64,
    output_activated: f64,
}

impl ConvolutionalNetwork {
    /// Creates and initializes a 2-layer CNN using Xavier Uniform initialization.
    pub fn new(seed: u32) -> Self {
        let mut prng = Lcg::new(seed);

        // 1. Initialize Conv Layer 1 (FILTERS1 filters of size 3x3)
        // fan_in = 3 * 3 = 9, fan_out = FILTERS1
        let conv1_bound = (6.0f64 / (9.0f64 + FILTERS1 as f64)).sqrt();
        let mut kernels1 = vec![vec![0.0; 9]; FILTERS1];
        for f in 0..FILTERS1 {
            for i in 0..9 {
                kernels1[f][i] = prng.next_range(-conv1_bound, conv1_bound);
            }
        }
        let conv_biases1 = vec![0.0; FILTERS1];

        // 2. Initialize Conv Layer 2 (FILTERS2 filters, each of size 3x3 x FILTERS1)
        // fan_in = 3 * 3 * FILTERS1 = 72, fan_out = FILTERS2
        let conv2_bound = (6.0f64 / (72.0f64 + FILTERS2 as f64)).sqrt();
        let mut kernels2 = vec![vec![vec![0.0; 9]; FILTERS1]; FILTERS2];
        for f_out in 0..FILTERS2 {
            for c_in in 0..FILTERS1 {
                for i in 0..9 {
                    kernels2[f_out][c_in][i] = prng.next_range(-conv2_bound, conv2_bound);
                }
            }
        }
        let conv_biases2 = vec![0.0; FILTERS2];

        // 3. Initialize Dense Layer (DENSE_NEURONS neurons, FLATTENED_SIZE inputs each)
        // fan_in = FLATTENED_SIZE, fan_out = DENSE_NEURONS
        let dense_bound = (6.0f64 / (FLATTENED_SIZE as f64 + DENSE_NEURONS as f64)).sqrt();
        let mut weights_dense = vec![vec![0.0; FLATTENED_SIZE]; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            for m in 0..FLATTENED_SIZE {
                weights_dense[k][m] = prng.next_range(-dense_bound, dense_bound);
            }
        }
        let biases_dense = vec![0.0; DENSE_NEURONS];

        // 4. Initialize Output Layer (1 neuron, DENSE_NEURONS inputs)
        // fan_in = DENSE_NEURONS, fan_out = 1
        let out_bound = (6.0f64 / (DENSE_NEURONS as f64 + 1.0f64)).sqrt();
        let mut weights_out = vec![0.0; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            weights_out[k] = prng.next_range(-out_bound, out_bound);
        }
        let bias_out = 0.0;

        Self {
            kernels1,
            conv_biases1,
            kernels2,
            conv_biases2,
            weights_dense,
            biases_dense,
            weights_out,
            bias_out,
            conv1_out: Vec::new(),
            conv1_activated: Vec::new(),
            pool1_out: Vec::new(),
            pool1_indices: Vec::new(),
            conv2_out: Vec::new(),
            conv2_activated: Vec::new(),
            pool2_out: Vec::new(),
            pool2_indices: Vec::new(),
            flattened: Vec::new(),
            dense_out: Vec::new(),
            dense_activated: Vec::new(),
            output_out: 0.0,
            output_activated: 0.0,
        }
    }

    /// Evaluates the CNN on a single input image vector of size 16384 (128x128 grayscale).
    /// Populates intermediate activation caches and returns the cat vs. dog probability in [0, 1].
    pub fn forward(&mut self, input: &[f64]) -> f64 {
        assert_eq!(input.len(), 16384, "Input image must have exactly 128x128 = 16,384 pixels.");

        let in_size = 128;
        let k_size = 3;

        // 1. Convolution Layer 1: 128x128x1 -> 126x126xFILTERS1
        let out_size1 = 126;
        self.conv1_out = vec![vec![0.0; out_size1 * out_size1]; FILTERS1];
        self.conv1_activated = vec![vec![0.0; out_size1 * out_size1]; FILTERS1];

        for f in 0..FILTERS1 {
            let kernel = &self.kernels1[f];
            let bias = self.conv_biases1[f];
            for r in 0..out_size1 {
                for c in 0..out_size1 {
                    let mut sum = 0.0;
                    for kr in 0..k_size {
                        for kc in 0..k_size {
                            let ir = r + kr;
                            let ic = c + kc;
                            sum += input[ir * in_size + ic] * kernel[kr * k_size + kc];
                        }
                    }
                    let pre_act = sum + bias;
                    self.conv1_out[f][r * out_size1 + c] = pre_act;
                    self.conv1_activated[f][r * out_size1 + c] = crate::math::relu(pre_act);
                }
            }
        }

        // 2. Max Pooling Layer 1: 126x126xFILTERS1 -> 63x63xFILTERS1
        let pool_in_size1 = 126;
        let pool_out_size1 = 63;
        self.pool1_out = vec![vec![0.0; pool_out_size1 * pool_out_size1]; FILTERS1];
        self.pool1_indices = vec![vec![(0, 0); pool_out_size1 * pool_out_size1]; FILTERS1];

        for f in 0..FILTERS1 {
            for pr in 0..pool_out_size1 {
                for pc in 0..pool_out_size1 {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = (0, 0);
                    for wr in 0..2 {
                        for wc in 0..2 {
                            let ir = pr * 2 + wr;
                            let ic = pc * 2 + wc;
                            let val = self.conv1_activated[f][ir * pool_in_size1 + ic];
                            if val > max_val {
                                max_val = val;
                                max_idx = (wr, wc);
                            }
                        }
                    }
                    self.pool1_out[f][pr * pool_out_size1 + pc] = max_val;
                    self.pool1_indices[f][pr * pool_out_size1 + pc] = max_idx;
                }
            }
        }

        // 3. Convolution Layer 2: 63x63xFILTERS1 -> 61x61xFILTERS2 (multi-channel 2D convolution)
        let in_size2 = 63;
        let out_size2 = 61;
        self.conv2_out = vec![vec![0.0; out_size2 * out_size2]; FILTERS2];
        self.conv2_activated = vec![vec![0.0; out_size2 * out_size2]; FILTERS2];

        for f_out in 0..FILTERS2 {
            let bias = self.conv_biases2[f_out];
            let mut sum_map = vec![bias; out_size2 * out_size2];
            for r in 0..out_size2 {
                for c in 0..out_size2 {
                    let mut sum = bias;
                    for c_in in 0..FILTERS1 {
                        let kernel = &self.kernels2[f_out][c_in];
                        let ch_in = &self.pool1_out[c_in];
                        for kr in 0..k_size {
                            for kc in 0..k_size {
                                let ir = r + kr;
                                let ic = c + kc;
                                sum += ch_in[ir * in_size2 + ic] * kernel[kr * k_size + kc];
                            }
                        }
                    }
                    sum_map[r * out_size2 + c] = sum;
                }
            }
            self.conv2_out[f_out] = sum_map.clone();
            for i in 0..(out_size2 * out_size2) {
                self.conv2_activated[f_out][i] = crate::math::relu(sum_map[i]);
            }
        }

        // 4. Max Pooling Layer 2: 61x61xFILTERS2 -> 30x30xFILTERS2 (crop 61st row/col, sliding 2x2 stride 2)
        let pool_in_size2 = 61;
        let pool_out_size2 = 30;
        self.pool2_out = vec![vec![0.0; pool_out_size2 * pool_out_size2]; FILTERS2];
        self.pool2_indices = vec![vec![(0, 0); pool_out_size2 * pool_out_size2]; FILTERS2];

        for f in 0..FILTERS2 {
            for pr in 0..pool_out_size2 {
                for pc in 0..pool_out_size2 {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = (0, 0);
                    for wr in 0..2 {
                        for wc in 0..2 {
                            let ir = pr * 2 + wr;
                            let ic = pc * 2 + wc;
                            let val = self.conv2_activated[f][ir * pool_in_size2 + ic];
                            if val > max_val {
                                max_val = val;
                                max_idx = (wr, wc);
                            }
                        }
                    }
                    self.pool2_out[f][pr * pool_out_size2 + pc] = max_val;
                    self.pool2_indices[f][pr * pool_out_size2 + pc] = max_idx;
                }
            }
        }

        // 5. Flatten Layer: 30x30xFILTERS2 -> FLATTENED_SIZE elements (14,400 nodes)
        self.flattened.clear();
        for f in 0..FILTERS2 {
            self.flattened.extend_from_slice(&self.pool2_out[f]);
        }

        // 6. Dense Layer: FLATTENED_SIZE -> DENSE_NEURONS hidden neurons with ReLU (prevent vanishing gradients!)
        self.dense_out = vec![0.0; DENSE_NEURONS];
        self.dense_activated = vec![0.0; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            let sum = crate::math::dot_product(&self.flattened, &self.weights_dense[k]) + self.biases_dense[k];
            self.dense_out[k] = sum;
            self.dense_activated[k] = crate::math::relu(sum); // Swapped to ReLU!
        }

        // 7. Output Layer: DENSE_NEURONS -> 1 neuron with Sigmoid
        let out_sum = crate::math::dot_product(&self.dense_activated, &self.weights_out) + self.bias_out;
        self.output_out = out_sum;
        self.output_activated = crate::math::sigmoid(out_sum);

        self.output_activated
    }

    /// Backpropagates loss gradients, updates network parameters, and returns the BCE loss.
    pub fn backprop(&mut self, input: &[f64], target: f64, lr: f64) -> f64 {
        // Step 1: Forward Pass
        let prediction = self.forward(input);

        // BCE Loss calculation
        let loss = crate::math::binary_cross_entropy(prediction, target);

        // Step 2: Output layer pre-activation gradient: \delta_{out} = Y_{pred} - Y_{target}
        let delta_out = prediction - target;

        // Step 3: Compute gradients for Output Layer Parameters
        let mut d_weights_out = vec![0.0; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            d_weights_out[k] = delta_out * self.dense_activated[k];
        }
        let d_bias_out = delta_out;

        // Step 4: Compute pre-activation gradient of dense hidden layer using ReLU derivative!
        let mut delta_dense = vec![0.0; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            let d_act = delta_out * self.weights_out[k];
            // Since activation is ReLU, we use relu_derivative on pre-activation dense_out
            delta_dense[k] = d_act * crate::math::relu_derivative(self.dense_out[k]);
        }

        // Step 5: Compute gradients for Dense Layer Parameters
        let mut d_weights_dense = vec![vec![0.0; FLATTENED_SIZE]; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            for m in 0..FLATTENED_SIZE {
                d_weights_dense[k][m] = delta_dense[k] * self.flattened[m];
            }
        }
        let d_biases_dense = delta_dense.clone();

        // Step 6: Compute gradient of loss with respect to flattened inputs (back to MaxPool2)
        let mut d_flattened = vec![0.0; FLATTENED_SIZE];
        for m in 0..FLATTENED_SIZE {
            let mut sum = 0.0;
            for k in 0..DENSE_NEURONS {
                sum += delta_dense[k] * self.weights_dense[k][m];
            }
            d_flattened[m] = sum;
        }

        // Step 7: Unflatten d_flattened back to d_pool2_out (30x30xFILTERS2)
        let pool2_out_size = 30;
        let pool2_in_size = 61;
        let pool1_size = 63;
        let pool1_in_size = 126;
        let k_size = 3;
        let img_size = 128;

        let mut d_pool2_out = vec![vec![0.0; pool2_out_size * pool2_out_size]; FILTERS2];
        let mut offset = 0;
        for f in 0..FILTERS2 {
            d_pool2_out[f].copy_from_slice(&d_flattened[offset..offset + pool2_out_size * pool2_out_size]);
            offset += pool2_out_size * pool2_out_size;
        }

        // Step 8: Route gradient back through Max Pooling 2 to d_conv2_activated (61x61xFILTERS2)
        let mut d_conv2_activated = vec![vec![0.0; pool2_in_size * pool2_in_size]; FILTERS2];
        for f in 0..FILTERS2 {
            for pr in 0..pool2_out_size {
                for pc in 0..pool2_out_size {
                    let grad = d_pool2_out[f][pr * pool2_out_size + pc];
                    let (wr, wc) = self.pool2_indices[f][pr * pool2_out_size + pc];
                    let ir = pr * 2 + wr;
                    let ic = pc * 2 + wc;
                    d_conv2_activated[f][ir * pool2_in_size + ic] = grad;
                }
            }
        }

        // Step 9: Backpropagate through Conv2 ReLU activation: d_conv2_out (61x61xFILTERS2)
        let mut d_conv2_out = vec![vec![0.0; pool2_in_size * pool2_in_size]; FILTERS2];
        for f in 0..FILTERS2 {
            for r in 0..pool2_in_size {
                for c in 0..pool2_in_size {
                    let idx = r * pool2_in_size + c;
                    let pre_act = self.conv2_out[f][idx];
                    let grad = d_conv2_activated[f][idx];
                    d_conv2_out[f][idx] = grad * crate::math::relu_derivative(pre_act);
                }
            }
        }

        // Step 10: Compute gradients for Conv Layer 2 parameters (kernels2 and biases2)
        let mut d_kernels2 = vec![vec![vec![0.0; k_size * k_size]; FILTERS1]; FILTERS2];
        let mut d_conv_biases2 = vec![0.0; FILTERS2];

        for f_out in 0..FILTERS2 {
            let d_out = &d_conv2_out[f_out];
            for c_in in 0..FILTERS1 {
                let pool1_ch = &self.pool1_out[c_in];
                for kr in 0..k_size {
                    for kc in 0..k_size {
                        let mut sum = 0.0;
                        for r in 0..pool2_in_size {
                            for c in 0..pool2_in_size {
                                let ir = r + kr;
                                let ic = c + kc;
                                sum += d_out[r * pool2_in_size + c] * pool1_ch[ir * pool1_size + ic];
                            }
                        }
                        d_kernels2[f_out][c_in][kr * k_size + kc] = sum;
                    }
                }
            }
            d_conv_biases2[f_out] = d_out.iter().sum();
        }

        // Step 11: Compute gradient of loss with respect to Conv2 inputs (to propagate back to Conv1)
        let mut d_pool1_out = vec![vec![0.0; pool1_size * pool1_size]; FILTERS1];
        for f_out in 0..FILTERS2 {
            let d_out = &d_conv2_out[f_out];
            for c_in in 0..FILTERS1 {
                let kernel = &self.kernels2[f_out][c_in];
                for r in 0..pool2_in_size {
                    for c in 0..pool2_in_size {
                        let grad = d_out[r * pool2_in_size + c];
                        for kr in 0..k_size {
                            for kc in 0..k_size {
                                let ir = r + kr;
                                let ic = c + kc;
                                d_pool1_out[c_in][ir * pool1_size + ic] += grad * kernel[kr * k_size + kc];
                            }
                        }
                    }
                }
            }
        }

        // Step 12: Route gradients back through Max Pooling 1 to d_conv1_activated (126x126xFILTERS1)
        let mut d_conv1_activated = vec![vec![0.0; pool1_in_size * pool1_in_size]; FILTERS1];
        for f in 0..FILTERS1 {
            for pr in 0..pool1_size {
                for pc in 0..pool1_size {
                    let grad = d_pool1_out[f][pr * pool1_size + pc];
                    let (wr, wc) = self.pool1_indices[f][pr * pool1_size + pc];
                    let ir = pr * 2 + wr;
                    let ic = pc * 2 + wc;
                    d_conv1_activated[f][ir * pool1_in_size + ic] = grad;
                }
            }
        }

        // Step 13: Backpropagate through Conv1 ReLU activation: d_conv1_out (126x126xFILTERS1)
        let mut d_conv1_out = vec![vec![0.0; pool1_in_size * pool1_in_size]; FILTERS1];
        for f in 0..FILTERS1 {
            for r in 0..pool1_in_size {
                for c in 0..pool1_in_size {
                    let idx = r * pool1_in_size + c;
                    let pre_act = self.conv1_out[f][idx];
                    let grad = d_conv1_activated[f][idx];
                    d_conv1_out[f][idx] = grad * crate::math::relu_derivative(pre_act);
                }
            }
        }

        // Step 14: Compute gradients for Conv Layer 1 parameters (kernels1 and biases1)
        let mut d_kernels1 = vec![vec![0.0; k_size * k_size]; FILTERS1];
        let mut d_conv_biases1 = vec![0.0; FILTERS1];

        for f in 0..FILTERS1 {
            let d_out = &d_conv1_out[f];
            for kr in 0..k_size {
                for kc in 0..k_size {
                    let mut sum = 0.0;
                    for r in 0..pool1_in_size {
                        for c in 0..pool1_in_size {
                            let ir = r + kr;
                            let ic = c + kc;
                            sum += d_out[r * pool1_in_size + c] * input[ir * img_size + ic];
                        }
                    }
                    d_kernels1[f][kr * k_size + kc] = sum;
                }
            }
            d_conv_biases1[f] = d_out.iter().sum();
        }

        // Step 15: Apply Parameter updates using SGD with L2 Weight Decay (Regularization)
        // 1. Update output weights and bias
        for k in 0..DENSE_NEURONS {
            self.weights_out[k] -= lr * (d_weights_out[k] + L2_REG * self.weights_out[k]);
        }
        self.bias_out -= lr * d_bias_out;

        // 2. Update dense layer weights and biases
        for k in 0..DENSE_NEURONS {
            for m in 0..FLATTENED_SIZE {
                self.weights_dense[k][m] -= lr * (d_weights_dense[k][m] + L2_REG * self.weights_dense[k][m]);
            }
            self.biases_dense[k] -= lr * d_biases_dense[k];
        }

        // 3. Update Conv Layer 2 kernels and biases
        for f_out in 0..FILTERS2 {
            for c_in in 0..FILTERS1 {
                for i in 0..(k_size * k_size) {
                    self.kernels2[f_out][c_in][i] -= lr * (d_kernels2[f_out][c_in][i] + L2_REG * self.kernels2[f_out][c_in][i]);
                }
            }
            self.conv_biases2[f_out] -= lr * d_conv_biases2[f_out];
        }

        // 4. Update Conv Layer 1 kernels and biases
        for f in 0..FILTERS1 {
            for i in 0..(k_size * k_size) {
                self.kernels1[f][i] -= lr * (d_kernels1[f][i] + L2_REG * self.kernels1[f][i]);
            }
            self.conv_biases1[f] -= lr * d_conv_biases1[f];
        }

        loss
    }

    /// Evaluates predictions only (no cache updating, thread-safe if needed).
    pub fn predict(&mut self, input: &[f64]) -> f64 {
        self.forward(input)
    }

    /// Serializes and saves model parameters to a text file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create weights file {}: {}", path.display(), e))?;

        // Format floats helper to conserve precision and file size
        let serialize_vec = |vec: &[f64]| -> String {
            vec.iter().map(|v| format!("{:.9}", v)).collect::<Vec<String>>().join(",")
        };

        // 1. Kernels1 (FILTERS1 rows, 9 values each)
        for f in 0..FILTERS1 {
            writeln!(file, "{}", serialize_vec(&self.kernels1[f])).map_err(|e| e.to_string())?;
        }
        // 2. Conv Biases1 (FILTERS1 values)
        writeln!(file, "{}", serialize_vec(&self.conv_biases1)).map_err(|e| e.to_string())?;

        // 3. Kernels2 (FILTERS2 rows, flat (FILTERS1 * 9) = 72 values each)
        for f_out in 0..FILTERS2 {
            let mut flat_k2 = Vec::with_capacity(FILTERS1 * 9);
            for c_in in 0..FILTERS1 {
                flat_k2.extend_from_slice(&self.kernels2[f_out][c_in]);
            }
            writeln!(file, "{}", serialize_vec(&flat_k2)).map_err(|e| e.to_string())?;
        }
        // 4. Conv Biases2 (FILTERS2 values)
        writeln!(file, "{}", serialize_vec(&self.conv_biases2)).map_err(|e| e.to_string())?;

        // 5. Dense Hidden Weights (DENSE_NEURONS rows, FLATTENED_SIZE values each)
        for k in 0..DENSE_NEURONS {
            writeln!(file, "{}", serialize_vec(&self.weights_dense[k])).map_err(|e| e.to_string())?;
        }
        // 6. Dense Biases (DENSE_NEURONS values)
        writeln!(file, "{}", serialize_vec(&self.biases_dense)).map_err(|e| e.to_string())?;

        // 7. Output Weights (DENSE_NEURONS values)
        writeln!(file, "{}", serialize_vec(&self.weights_out)).map_err(|e| e.to_string())?;

        // 8. Output Bias (1 value)
        writeln!(file, "{:.9}", self.bias_out).map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Deserializes and restores model parameters from a text file.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open weights file {}: {}", path.display(), e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let parse_vec = |line: Option<Result<String, std::io::Error>>, expected_len: usize| -> Result<Vec<f64>, String> {
            let line_str = line
                .ok_or_else(|| "Unexpected End-Of-File during deserialization.".to_string())?
                .map_err(|e| e.to_string())?;
            let parsed: Vec<f64> = line_str
                .split(',')
                .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
                .collect::<Result<Vec<f64>, String>>()?;
            if parsed.len() != expected_len {
                return Err(format!("Weights length mismatch. Expected {}, found {}.", expected_len, parsed.len()));
            }
            Ok(parsed)
        };

        // 1. Kernels1
        let mut kernels1 = vec![vec![0.0; 9]; FILTERS1];
        for f in 0..FILTERS1 {
            kernels1[f] = parse_vec(lines.next(), 9)?;
        }
        // 2. Conv Biases1
        let conv_biases1 = parse_vec(lines.next(), FILTERS1)?;

        // 3. Kernels2 (FILTERS2 rows, flat FILTERS1 * 9 = 72 values each)
        let mut kernels2 = vec![vec![vec![0.0; 9]; FILTERS1]; FILTERS2];
        for f_out in 0..FILTERS2 {
            let flat_k2 = parse_vec(lines.next(), FILTERS1 * 9)?;
            for c_in in 0..FILTERS1 {
                let start = c_in * 9;
                kernels2[f_out][c_in].copy_from_slice(&flat_k2[start..start + 9]);
            }
        }
        // 4. Conv Biases2
        let conv_biases2 = parse_vec(lines.next(), FILTERS2)?;

        // 5. Dense Hidden Weights
        let mut weights_dense = vec![vec![0.0; FLATTENED_SIZE]; DENSE_NEURONS];
        for k in 0..DENSE_NEURONS {
            weights_dense[k] = parse_vec(lines.next(), FLATTENED_SIZE)?;
        }
        // 6. Dense Biases
        let biases_dense = parse_vec(lines.next(), DENSE_NEURONS)?;

        // 7. Output Weights
        let weights_out = parse_vec(lines.next(), DENSE_NEURONS)?;

        // 8. Output Bias
        let bias_out_str = lines
            .next()
            .ok_or_else(|| "Missing output bias field.".to_string())?
            .map_err(|e| e.to_string())?;
        let bias_out = bias_out_str.parse::<f64>().map_err(|e| e.to_string())?;

        Ok(Self {
            kernels1,
            conv_biases1,
            kernels2,
            conv_biases2,
            weights_dense,
            biases_dense,
            weights_out,
            bias_out,
            conv1_out: Vec::new(),
            conv1_activated: Vec::new(),
            pool1_out: Vec::new(),
            pool1_indices: Vec::new(),
            conv2_out: Vec::new(),
            conv2_activated: Vec::new(),
            pool2_out: Vec::new(),
            pool2_indices: Vec::new(),
            flattened: Vec::new(),
            dense_out: Vec::new(),
            dense_activated: Vec::new(),
            output_out: 0.0,
            output_activated: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_forward_dimensions() {
        let mut nn = ConvolutionalNetwork::new(1234);
        let sample = vec![0.5; 16384]; // 128x128 input
        let pred = nn.forward(&sample);
        assert!(pred >= 0.0 && pred <= 1.0);
        assert_eq!(nn.kernels1.len(), FILTERS1);
        assert_eq!(nn.conv1_out.len(), FILTERS1);
        assert_eq!(nn.conv1_out[0].len(), 126 * 126);
        assert_eq!(nn.pool1_out.len(), FILTERS1);
        assert_eq!(nn.pool1_out[0].len(), 63 * 63);
        
        assert_eq!(nn.kernels2.len(), FILTERS2);
        assert_eq!(nn.kernels2[0].len(), FILTERS1);
        assert_eq!(nn.conv2_out.len(), FILTERS2);
        assert_eq!(nn.conv2_out[0].len(), 61 * 61);
        assert_eq!(nn.pool2_out.len(), FILTERS2);
        assert_eq!(nn.pool2_out[0].len(), 30 * 30);

        assert_eq!(nn.flattened.len(), FLATTENED_SIZE);
        assert_eq!(nn.dense_activated.len(), DENSE_NEURONS);
    }

    #[test]
    fn test_backpropagation_decreases_loss() {
        // Test if overfitting a single sample works on the 2-layer CNN hierarchy
        let mut nn = ConvolutionalNetwork::new(42);
        let sample = vec![0.3; 16384];
        let target = 1.0;

        let initial_loss = nn.backprop(&sample, target, 0.1);
        let mut last_loss = initial_loss;

        for _ in 0..10 {
            last_loss = nn.backprop(&sample, target, 0.1);
        }

        assert!(
            last_loss < initial_loss,
            "Loss should decrease after several backprop iterations on 2-layer hierarchy. Initial: {}, Final: {}",
            initial_loss,
            last_loss
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        use std::env;
        let mut nn = ConvolutionalNetwork::new(999);
        let sample = vec![0.1; 16384];
        let pred_orig = nn.forward(&sample);

        let temp_dir = env::temp_dir();
        let test_path = temp_dir.join("perceptronium_cnn_test_weights.txt");

        nn.save_to_file(&test_path).expect("Failed to serialize CNN");
        let mut restored_nn = ConvolutionalNetwork::load_from_file(&test_path).expect("Failed to deserialize CNN");

        let pred_restored = restored_nn.forward(&sample);
        assert!(
            (pred_orig - pred_restored).abs() < 1e-9,
            "Restored network prediction {} does not match original {}",
            pred_restored,
            pred_orig
        );

        let _ = std::fs::remove_file(test_path);
    }
}
