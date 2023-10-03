local DQNAgent = {}
DQNAgent.__index = DQNAgent

local Memory = {}
Memory.__index = Memory

function Memory.new(max_size)
	return setmetatable({max_size = max_size or 2000, data = {}, priorities = {}, idx = 0}, Memory)
end

function Memory:add(experience, priority)
	self.idx = (self.idx % self.max_size) + 1
	self.data[self.idx] = experience
	self.priorities[self.idx] = priority or 1.0
end

function Memory:sample(batch_size)
	local indices = {}
	local total_priority = 0
	for _, p in ipairs(self.priorities) do
		total_priority = total_priority + p
	end
	for _ = 1, batch_size do
		local rand_priority = math.random() * total_priority
		local running_total = 0
		for idx, p in ipairs(self.priorities) do
			running_total = running_total + p
			if running_total >= rand_priority then
				table.insert(indices, idx)
				break
			end
		end
	end
	local batch = {}
	for _, idx in ipairs(indices) do
		table.insert(batch, self.data[idx])
	end
	return batch
end

function DQNAgent.new(params)
	local self = setmetatable({}, DQNAgent)
	self.state_size = params.state_size
	self.action_size = params.action_size
	self.model = params.mlp
	self.gamma = params.gamma or 0.99
	self.epsilon = params.epsilon_start or 1
	self.epsilon_min = params.epsilon_end or 0.1
	self.epsilon_decay = params.epsilon_decay or 0.995
	self.memory = Memory.new(params.memory_size or 2000)
	self.alpha = params.alpha or 0.6
	self.target_model = self.model:clone()
	self.update_target_freq = params.update_target_freq or 1000
	self.train_steps = 0
	return self
end

function DQNAgent:remember(state, action, reward, next_state, done)
	self.memory:add({state, action, reward, next_state, done}, math.pow(1.0, self.alpha))
end

function DQNAgent:act(state)
	if math.random() <= self.epsilon then
		return math.random(1, self.action_size)
	end
	local activations = self.model:forward(state)
	local q_values = activations[#activations]
	local best_action = table.maxn(q_values)
	return best_action
end

function DQNAgent:replay(batch_size)
	local batch = self.memory:sample(batch_size)
	for _, sample in ipairs(batch) do
		local state, action, reward, next_state, done = unpack(sample)
		local target = reward
		if not done then
			local next_activations = self.model:forward(next_state)
			local next_q_values = next_activations[#next_activations]
			local best_action = table.maxn(next_q_values)

			local target_activations = self.target_model:forward(next_state)
			local target_q_values = target_activations[#target_activations]
			target = reward + self.gamma * target_q_values[best_action]
		end
		local activations = self.model:forward(state)
		local q_values = activations[#activations]
		q_values[action] = target
		self.model:train(state, q_values)
	end
	self.train_steps = self.train_steps + 1
	if self.train_steps % self.update_target_freq == 0 then
		self.target_model = self.model:clone()
	end
	if self.epsilon > self.epsilon_min then
		self.epsilon = self.epsilon * self.epsilon_decay
	end
end
