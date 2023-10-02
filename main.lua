local HttpService = game:GetService("HttpService")

math.randomseed(os.clock())

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

local function sigmoid_derivative(x)
    local sig = sigmoid(x)
    return sig * (1 - sig)
end

local function relu(x)
    return math.max(0, x)
end

local function relu_derivative(x)
    return x > 0 and 1 or 0
end

local function tanh(x)
    return math.tanh(x)
end

local function tanh_derivative(x)
    local tanh_value = tanh(x)
    return 1 - tanh_value * tanh_value
end

function linear(x)
    return x
end

function linear_derivative(x)
    return 1
end

local function xavier_init(dim1, dim2)
    local bound = math.sqrt(6 / (dim1 + dim2))
    local matrix = {}
    for i = 1, dim1 do
        matrix[i] = {}
        for j = 1, dim2 do
            matrix[i][j] = math.random() * 2 * bound - bound
        end
    end
    return matrix
end

local function he_init(dim1, dim2)
    local bound = math.sqrt(2 / dim1)
    local matrix = {}
    for i = 1, dim1 do
        matrix[i] = {}
        for j = 1, dim2 do
            matrix[i][j] = math.random() * 2 * bound - bound
        end
    end
    return matrix
end

local activation_mapping = {
    sigmoid = {sigmoid, sigmoid_derivative},
    relu = {relu, relu_derivative},
    tanh = {tanh, tanh_derivative}
}

local weight_init_mapping = {
    xavier = xavier_init, 
    he = he_init
}

local MLP = {}
MLP.__index = MLP

function MLP.new(input_size, layer_sizes, activation_function_names, learning_rate, momentum, weight_decay, weight_init)
    if #layer_sizes ~= #activation_function_names then
        error("Number of layer_sizes must be equal to the number of activation_function_names")
    end

    local self = setmetatable({}, MLP)
    self.input_size = input_size
    self.layer_sizes = layer_sizes
    self.activation_function_names = activation_function_names
    self.learning_rate = learning_rate or 0.1
    self.momentum = momentum or 0.9
    self.weight_decay = weight_decay or 0.0001
    self.weight_init = weight_init or "xavier"

    self.activation_functions = {}
    for i, name in ipairs(activation_function_names) do
        self.activation_functions[i] = activation_mapping[name]
    end

    self.weights = {}
    self.biases = {}
    self.velocities = {}

    local prev_size = input_size
    for i, size in ipairs(layer_sizes) do
        self.weights[i] = weight_init_mapping[self.weight_init](size, prev_size)
        self.biases[i] = {}
        self.velocities[i] = {}
        for j = 1, size do
            self.biases[i][j] = math.random() * 2 - 1
            self.velocities[i][j] = 0
        end
        prev_size = size
    end

    return self
end

function MLP:forward(inputs)
    local activations = {inputs}
    local layer_input = inputs

    for i = 1, #self.layer_sizes do
        local layer_output = {}
        for j = 1, self.layer_sizes[i] do
            local sum = self.biases[i][j]
            for k = 1, #layer_input do
                sum = sum + self.weights[i][j][k] * layer_input[k]
            end
            layer_output[j] = self.activation_functions[i][1](sum)
        end
        layer_input = layer_output
        table.insert(activations, layer_output)
    end

    return activations
end

function MLP:train(inputs, targets)
    local activations = self:forward(inputs)
    local gradients = {}

    for i = 1, self.layer_sizes[#self.layer_sizes] do
        local error = targets[i] - activations[#activations][i]
        gradients[#self.layer_sizes] = gradients[#self.layer_sizes] or {}
        gradients[#self.layer_sizes][i] =
            error * self.activation_functions[#self.layer_sizes][2](activations[#activations][i])
    end

    for i = #self.layer_sizes - 1, 1, -1 do
        gradients[i] = {}
        for j = 1, self.layer_sizes[i] do
            local sum = 0
            for k = 1, self.layer_sizes[i + 1] do
                sum = sum + self.weights[i + 1][k][j] * gradients[i + 1][k]
            end
            gradients[i][j] = sum * self.activation_functions[i][2](activations[i + 1][j])
        end
    end

    for i = #self.layer_sizes, 1, -1 do
        for j = 1, self.layer_sizes[i] do
            for k = 1, #activations[i] do
                self.velocities[i][j] =
                    self.momentum * self.velocities[i][j] + self.learning_rate * gradients[i][j] * activations[i][k]
                self.weights[i][j][k] =
                    self.weights[i][j][k] * (1 - self.learning_rate * self.weight_decay) + self.velocities[i][j]
            end
            self.biases[i][j] = self.biases[i][j] + self.learning_rate * gradients[i][j]
        end
    end
end

function MLP:save(filename)
    local model_data = {
        input_size = self.input_size,
        layer_sizes = self.layer_sizes,
        activation_function_names = self.activation_function_names,
        learning_rate = self.learning_rate,
        momentum = self.momentum,
        weight_decay = self.weight_decay,
        weight_init = self.weight_init,
        weights = self.weights,
        biases = self.biases,
        velocities = self.velocities
    }
    writefile(filename, HttpService:JSONEncode(model_data))
end

function MLP:load(filename)
    if isfile(filename) then
        local model_data = HttpService:JSONDecode(readfile(filename))

        if
            model_data and model_data.input_size and model_data.layer_sizes and model_data.activation_function_names and
                model_data.learning_rate and
                model_data.momentum and
                model_data.weight_decay and
                model_data.weight_init and
                model_data.weights and
                model_data.biases and
                model_data.velocities
         then
            self.input_size = model_data.input_size
            self.layer_sizes = model_data.layer_sizes
            self.activation_function_names = model_data.activation_function_names
            self.learning_rate = model_data.learning_rate
            self.momentum = model_data.momentum
            self.weight_decay = model_data.weight_decay
            self.weight_init = model_data.weight_init
            self.weights = model_data.weights
            self.biases = model_data.biases
            self.velocities = model_data.velocities

            self.activation_functions = {}
            for i, name in ipairs(self.activation_function_names) do
                self.activation_functions[i] = activation_mapping[name]
            end
        else
            warn("Invalid model data in file: " .. filename)
        end
    else
        warn("File not found: " .. filename)
    end
end

return MLP
