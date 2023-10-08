local DQN = {}
DQN.__index = DQN

function DQN.new(state_size, action_size, learning_rate, gamma, epsilon, epsilon_decay, mlp_params, memory_size, target_update_freq, alpha)
	local self = setmetatable({}, DQN)
	self.state_size = state_size
	self.action_size = action_size
	self.learning_rate = learning_rate or 0.1
	self.gamma = gamma or 0.9
	self.epsilon = epsilon or 0.1
	self.epsilon_min = 0.01
	self.epsilon_decay = epsilon_decay or 0.995
	self.q_network = MLP.new(state_size, mlp_params.layer_sizes, mlp_params.activation_function_names, mlp_params.learning_rate, mlp_params.momentum, mlp_params.weight_decay, mlp_params.weight_init)
	self.target_network = MLP.new(state_size, mlp_params.layer_sizes, mlp_params.activation_function_names, mlp_params.learning_rate, mlp_params.momentum, mlp_params.weight_decay, mlp_params.weight_init)
	self.memory = {}
	self.memory_size = memory_size or 5000
	self.batch_size = 64
	self.target_update_freq = target_update_freq or 100
	self.train_step = 0
	self.alpha = alpha or 0.6
	self.beta = 0.4
	self.beta_increment = 0.001
	self.priority_eps = 1e-6
	return self
end

function DQN:get_max_priority()
	if #self.memory == 0 then
		return 1.0
	end
	local max_priority = -math.huge
	for _, experience in ipairs(self.memory) do
		max_priority = math.max(max_priority, experience[6])
	end
	return max_priority
end

function DQN:remember(state, action, reward, next_state, done)
	local max_priority = self:get_max_priority()
	if #self.memory >= self.memory_size then
		table.remove(self.memory, 1)
	end
	table.insert(self.memory, {state, action, reward, next_state, done, max_priority})
end

function DQN:sample_memory()
	local total_priority = 0
	for _, experience in ipairs(self.memory) do
		total_priority = total_priority + experience[6] ^ self.alpha
	end

	local batch = {}
	local weights = {}

	for _ = 1, self.batch_size do
		local rand_priority = math.random() * total_priority
		local accumulated_priority = 0
		local selected_experience = nil

		for _, experience in ipairs(self.memory) do
			accumulated_priority = accumulated_priority + experience[6] ^ self.alpha
			if accumulated_priority > rand_priority then
				selected_experience = experience
				break
			end
		end

		if not selected_experience then
			local max_priority_val = -math.huge
			for _, experience in ipairs(self.memory) do
				if experience[6] > max_priority_val then
					max_priority_val = experience[6]
					selected_experience = experience
				end
			end
		end

		table.insert(batch, selected_experience)
		local sampling_prob = (selected_experience[6] ^ self.alpha) / total_priority
		local weight = (1 / (sampling_prob * #self.memory)) ^ self.beta
		table.insert(weights, weight)
	end

	self.beta = math.min(1, self.beta + self.beta_increment)

	return batch, weights
end

local function max_index(t)
	local max_val = t[1]
	local max_idx = 1

	for i, v in ipairs(t) do
		if v > max_val then
			max_val = v
			max_idx = i
		end
	end

	return max_val, max_idx
end

function DQN:choose_action(state)
	if math.random() < self.epsilon then
		return math.random(self.action_size)
	else
		local q_values_all_layers = self.q_network:forward(state)
		local q_values = q_values_all_layers[#q_values_all_layers]
		local _, action = max_index(q_values)
		return action
	end
end

function DQN:update()
	local batch, weights = self:sample_memory()
	local max_weight = math.max(unpack(weights))

	for idx, experience in ipairs(batch) do
		local state, action, reward, next_state, done = unpack(experience)
	
		local q_values_all_layers = self.q_network:forward(state)
		local q_values = q_values_all_layers[#q_values_all_layers]

		local q_values_next_all_layers = self.q_network:forward(next_state)
		local q_values_next = q_values_next_all_layers[#q_values_next_all_layers]
		local _, best_action = max_index(q_values_next)

		local q_values_target_next_all_layers = self.target_network:forward(next_state)
		local q_values_target_next = q_values_target_next_all_layers[#q_values_target_next_all_layers]
		local target = reward
		if not done then
			target = reward + self.gamma * q_values_target_next[best_action]
		end
		
		local td_error = target - q_values[action]
		experience[6] = math.abs(td_error) + self.priority_eps

		local target_vector = {}
		for i = 1, self.action_size do
			if i == action then
				table.insert(target_vector, target)
			else
				table.insert(target_vector, q_values[i])
			end
		end
		
		
		local weight = weights[idx] / max_weight
		self.q_network:train(state, target_vector, weight)
	end

	if self.epsilon > self.epsilon_min then
		self.epsilon = self.epsilon * self.epsilon_decay
	end

	self.train_step = self.train_step + 1
	if self.train_step % self.target_update_freq == 0 then
		self.target_network.weights = self.q_network.weights
		self.target_network.biases = self.q_network.biases
	end
end
