"""
Network structure. 

    sizes::Array{Int64, 1}
    weights_hidden::Array{Float64,2}
    weights_output::Array{Float64,2}
    biases_hidden::Array{Float64,1}
    biases_output::Array{Float64,1} 

"""
struct Network_struct1
    sizes::Array{Int64, 1} # size of each layer
    # randomised default values
    weights_hidden::Array{Float64,2}
    weights_output::Array{Float64,2}
    biases_hidden::Array{Float64,1} 
    biases_output::Array{Float64,1} 
end
"""
convert an int in (0-9) to Array{Float64,1} eg.

    4 -> [0,0,0,0,4,0,0,0,0,0]

Not used for matrix determinant example
"""
function binary_rep(target::Int64)
    rep = zeros(10)
    rep[target+1]=1.0
    return rep
end
"""
sigmoid function for each "neuron"
"""
function sigmoid(z::Float64)
    return 1/(1+exp(-z))
end
"""
evaluate output of a given network configuration. Arguments:

    self::Network_struct
    input_data::Array{Float64,1}
    target::Int64

"""
function evaluate(self::Network_struct1, input_data::Array{Float64,1}, target::Float64)
    hidden_layer = Float64[]
    network_output = Float64[]
    #evaluate the hidden layer
    for i in 1:self.sizes[2]
        push!(hidden_layer, sigmoid(dot(self.weights_hidden[i,:],input_data)+self.biases_hidden[i]))
    end
    for i in 1:self.sizes[3]
        push!(network_output, sigmoid(dot(self.weights_output[i,:],hidden_layer)+self.biases_output[i]))
    end
    # pass forward to output layer
    return network_output
end 
"""
Calculate mean squared error of output and target
    self::Network_struct1
    network_output::Array{Float64,1}
    target::Float64
"""
function cost(self::Network_struct1, network_output::Array{Float64,1}, target::Float64)
    return (1/self.sizes[3]) * sum(abs2.(network_output .- target))
end
"""
Function to update weights and biases by gradient descent. Arguments:
    
    self::Network_struct
    learning_rate::Float64
    step_size::Float64
    parameter_space_size::Int64
    input_data::Array{Float64,1}
    tolerance::Float64
    target::Int64

"""
function train(self::Network_struct1, learning_rate::Float64, step_size::Float64, 
        parameter_space_size::Int64, input_data::Array{Float64,1}, tolerance::Float64, target::Float64)
    
    output = evaluate(self, input_data, target)
    C_0 = cost(self, output, target)
    
    print("\ntraining...\n")
    iter = 0
    while iter < 50 #C_0 > 0.01
        output = evaluate(self, input_data, target)
        @show C_0 = cost(self, output, target)
        w2_0 = copy(self.weights_hidden)
        w3_0 = copy(self.weights_output)
        b2_0 = copy(self.biases_hidden)
        b3_0 = copy(self.biases_output)
        
        dw1 = [];
        dw2 = [];
        db1 = [];
        db2 = [];
        for i in 1:length(self.weights_hidden)

            update_set = copy(w2_0)
            update_set[i] += step_size

            updated_network = 
            Network_struct1(self.sizes, update_set, self.weights_output, 
                self.biases_hidden, self.biases_output)

            updated_output = evaluate(updated_network, input_data, target)
            updated_eval = cost(updated_network, updated_output, target)
            derivative = (1.0/step_size)*(updated_eval-C_0)
            push!(dw1, derivative)
            
        end
        
        for i in 1:length(self.biases_hidden)

            update_set = copy(b2_0)
            update_set[i] += step_size

            updated_network = 
            Network_struct1(self.sizes, self.weights_hidden, self.weights_output,
                update_set, self.biases_output )

            updated_output = evaluate(updated_network, input_data, target)
            updated_eval = cost(updated_network, updated_output, target)

            derivative = (1.0/step_size)*(updated_eval-C_0)
            push!(db1, derivative)
        end
        
        for i in 1:length(self.weights_output)

            update_set = copy(w3_0)
            update_set[i] += step_size

            updated_network = 
            Network_struct1(self.sizes, self.weights_hidden, update_set,
                self.biases_hidden, self.biases_output )

            updated_output = evaluate(updated_network, input_data, target)
            updated_eval = cost(updated_network, updated_output, target)

            derivative = (1.0/step_size)*(updated_eval-C_0)
            push!(dw2, derivative)
        end
        
        for i in 1:length(self.biases_output)
            update_set = copy(b3_0)
            update_set[i] += step_size
            updated_network = 
            Network_struct1(self.sizes, self.weights_hidden, self.weights_output,
                self.biases_hidden, update_set)
            
            updated_output = evaluate(updated_network, input_data, target)
            updated_eval = cost(updated_network, updated_output, target)

            derivative = (1.0/step_size)*(updated_eval-C_0)
            push!(db2, derivative)
        end
        
        for (i, gi) in enumerate(dw1) self.weights_hidden[i] -= learning_rate*gi end
        for (i, gi) in enumerate(db1) self.biases_hidden[i] -= learning_rate*gi end
        for (i, gi) in enumerate(dw2) self.weights_output[i] -= learning_rate*gi end
        for (i, gi) in enumerate(db2) self.biases_output[i] -= learning_rate*gi end
        output = evaluate(self, input_data, target)
        C_0 = cost(self, output, target)
        iter = iter + 1
    end
end

# random matrix to test network
matrix_rand = rand(3,3)
network_size = [length(matrix_rand),6,1]

hidden_weight = rand(network_size[2],network_size[1])
output_weight = rand(network_size[3],network_size[2])
hidden_biases = rand(network_size[2])
output_biases = rand(network_size[3])

network = 
Network_struct1(network_size, hidden_weight, output_weight, hidden_biases, output_biases)

parameter_space_size =  length(network.weights_hidden)+
      length(network.weights_output)+
      length(network.biases_hidden)+
      length(network.biases_hidden)

learning_rate = 0.1; step_size = 0.00001; tolerance = 0.1;


for i in 1:25
    # random matrix
    matrix_rand = 5.0 .* rand(3,3)#[1.0 1.0 ; 0.0 1.5]
    # target result
    target = det(matrix_rand)
    
    # train network on sample data
    train(network, learning_rate, step_size, parameter_space_size, 
        vcat(matrix_rand...), tolerance, target)
    
    print("\n ___________________________________________________\n")
    print("Determinant (Prediction) = $(evaluate(network, vcat(matrix_rand...), target))\n")
    print("Determinant (True) = $(target)")
    print("\n ___________________________________________________\n")
    print("Error = $(evaluate(network, vcat(matrix_rand...), target) - target)\n")
    print("% Error = $(((evaluate(network, vcat(matrix_rand...), target) / target)*100.0) - 100.0)\n")
end
