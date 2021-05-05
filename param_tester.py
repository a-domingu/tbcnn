from main import main


# TODO asignar los valores que queramos para cada caso

vector_size_ls = [20]

learning_rate_ls = [0.1]

momentum_ls = [0.01]

# Second neural network parameters
feature_size_ls = [4]
epoch = 10
pooling = 'one-way pooling'

for vector_size in vector_size_ls:
    for learning_rate in learning_rate_ls:
        for momentum in momentum_ls:
            for learning_rate2 in [0.1]:
                for feature_size in feature_size_ls:
                    main(vector_size, learning_rate, momentum, learning_rate2,\
                         feature_size, epoch, pooling)




