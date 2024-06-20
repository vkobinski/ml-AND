// src/main.rs

use std::vec;

use rand::Rng;

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    output: f64,
    delta: f64,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);
        Neuron {
            weights,
            bias,
            output: 0.0,
            delta: 0.0,
        }
    }

    fn activate(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        sum + self.bias
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(output: f64) -> f64 {
        output * (1.0 - output)
    }

    fn forward(&mut self, inputs: &[f64]) -> f64 {
        self.output = Neuron::sigmoid(self.activate(inputs));
        self.output
    }

    fn calculate_delta(&mut self, target: f64) {
        self.delta = (target - self.output) * Neuron::sigmoid_derivative(self.output);
    }

    fn update_weights(&mut self, inputs: &[f64], learning_rate: f64) {
        for i in 0..self.weights.len() {
            self.weights[i] += learning_rate * self.delta * inputs[i];
        }
        self.bias += learning_rate * self.delta;
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let neurons = (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect();
        Layer { neurons }
    }

    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i]));
        }
        NeuralNetwork {
            layers,
            learning_rate,
        }
    }

    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = inputs.to_vec();
        for layer in self.layers.iter_mut() {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    fn backward(&mut self, inputs: &[f64], targets: &[f64]) {
        // Calculate output layer delta
        let mut layer_inputs = inputs.to_vec();
        for layer in self.layers.iter_mut() {
            layer_inputs = layer.forward(&layer_inputs);
        }

        for (i, neuron) in self
            .layers
            .last_mut()
            .unwrap()
            .neurons
            .iter_mut()
            .enumerate()
        {
            neuron.calculate_delta(targets[i]);
        }

        // Calculate hidden layers delta
        for l in (0..self.layers.len() - 1).rev() {
            for i in 0..self.layers[l].neurons.len() {
                let delta: f64 = self.layers[l + 1]
                    .neurons
                    .iter()
                    .map(|n| n.weights[i] * n.delta)
                    .sum();
                self.layers[l].neurons[i].delta =
                    delta * Neuron::sigmoid_derivative(self.layers[l].neurons[i].output);
            }
        }

        // Update weights
        let mut layer_inputs = inputs.to_vec();
        for layer in self.layers.iter_mut() {
            for neuron in layer.neurons.iter_mut() {
                neuron.update_weights(&layer_inputs, self.learning_rate);
            }
            layer_inputs = layer.forward(&layer_inputs);
        }
    }

    fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.forward(input);
                self.backward(input, target);
            }
        }
    }
}

fn generate_input() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut inputs: Vec<Vec<f64>> = vec![];
    let mut outs: Vec<Vec<f64>> = vec![];

    for i in 0..1000 {
        let mut rng = rand::thread_rng();
        let mut new: Vec<f64> = vec![];
        let mut new_out: Vec<f64> = vec![];
        let random_1: f64 = rng.gen_range(0.0..1.9);
        let random_2: f64 = rng.gen_range(0.0..1.9);

        let result = random_1 as i32 & random_2 as i32;

        new.push((random_1 as i32) as f64);
        new.push((random_2 as i32) as f64);

        new_out.push(result as f64);

        outs.push(new_out);
        inputs.push(new);
    }

    (inputs, outs)
}

fn main() {
    // AND dataset

    let (inputs, targets) = generate_input();

    let mut nn = NeuralNetwork::new(&[2, 2, 1], 0.1);
    nn.train(inputs.clone(), targets.clone(), 10000);

    for input in inputs {
        let output = nn.forward(&input);

        let mut first_out = *output.first().unwrap();

        if first_out >= 0.5 {
            first_out = 1.0;
        } else {
            first_out = 0.0;
        }

        println!("{:?} -> {:?}", input, first_out);
    }
}
