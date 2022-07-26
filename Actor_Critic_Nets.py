
import tensorflow as tf 
import numpy as np 

class Critic(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        initializer_weights = tf.keras.initializers.Orthogonal() # based on the original PPO paper
        initializer_bais = tf.keras.initializers.Constant(value=0.0)
        self.d1 = tf.keras.layers.Dense(64,  activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        self.d2 = tf.keras.layers.Dense(64,  activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        self.v = tf.keras.layers.Dense(1,  activation = None, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        
        # activate layers
        x = tf.random.normal(shape = input_dims)
        self.call(x)
        self.trainable_param = []
        
        # collect all trainable params 
        for layer in self.layers:
            # layers object is already in the parents 
            self.trainable_param += layer.trainable_variables
        
    def call(self, x):
        x = np.asarray(x).astype(np.float32)
        for layer in self.layers:
            x = layer(x)
            
        return x
    

class Actor(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        initializer_weights = tf.keras.initializers.Orthogonal() # based on the original PPO paper
        initializer_bais = tf.keras.initializers.Constant(value=0.0)
        
        self.d1 = tf.keras.layers.Dense(64, activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        self.d2 = tf.keras.layers.Dense(64, activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        self.a = tf.keras.layers.Dense(output_dims,activation = None, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        
        x = tf.random.normal(shape = input_dims)
        self.call(x)
        
        # collect trainable params 
        self.trainable_param = []
        for layer in self.layers:
            self.trainable_param += layer.trainable_variables
            
    def call(self,x):
        x = np.asarray(x, dtype=np.float32)
        for layer in self.layers:
            x = layer(x)
        return x
    
class ActorCritic(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        initializer_weights = tf.keras.initializers.Orthogonal() # based on the original PPO paper
        initializer_bais = tf.keras.initializers.Constant(value=0.0)
        
        self.d1 = tf.keras.layers.Dense(64, activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        self.d2 = tf.keras.layers.Dense(64, activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        self.a = tf.keras.layers.Dense(output_dims,activation = tf.nn.tanh, 
                                        kernel_initializer = initializer_weights,
                                        bias_initializer=initializer_bais)
        
        
        x = tf.random.normal(shape = input_dims)
        self.call(x)
        
        # collect trainable params 
        self.trainable_param = []
        for layer in self.layers:
            self.trainable_param += layer.trainable_variables
            
    def call(self,x):
        x = np.asarray(x, dtype=np.float32)
        for layer in self.layers:
            x = layer(x)
        return x  
# x = tf.random.normal(shape = (1,3))
# critic = Critic(input_dims= (1,3), output_dims= 1)
# actor = Actor(input_dims= (1,3), output_dims= 4)


# print(critic.call(x))
# print(actor(x))