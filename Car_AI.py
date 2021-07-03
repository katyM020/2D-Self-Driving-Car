# AI for 2D Self Driving Car

"""
Torch, a script language based on Lua, is an open-source ML library that
provides wide range of algorithms for Artificial intelligence.
    - torch.nn: used for neural networks creation
    - torch.nn.functional: used for activation and loss functions
    - torch.optim: used for the optimizers
"""
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optm
from torch.autograd import Variable

import random
import os


# Creating the of the Neural Network
class NeuralNetwork(nn.Module):
    """
    The NeuralNetwork class is used to create the architecture our solution
    based on neural networks (deep Q-learning). Here our class inherit from the Module class.
    """
    def __init__(self, Size_of_inputs, OutputAction):
        """
        The initialization function takes the following parameters:
            :param Size_of_inputs: the size of the input layer in the NN (the given inputs)
            :param OutputAction: the size of the output layer in the NN (the possible output actions),
                in this scope we have only 3 possible actions rotate (0, 10, -10) degrees.
        """
        super(NeuralNetwork, self).__init__()

        # getting values from the given parameters
        self.Size_of_inputs = Size_of_inputs
        self.OutputAction = OutputAction

        # First neural connection between the given input layer to a hidden layer (with 30 neurons)
        self.connectionOne = nn.Linear(Size_of_inputs, 30)
        # Second neural connection from the hidden layer, having 30 neurons, to the output layer
        self.connectionTwo = nn.Linear(30, OutputAction)

    def forwardPropagation(self, state):
        """
        This function is used to forward the signals from the input layer, to the hidden layer, to the output layer.
        It uses Rectified Linear Unit (ReLU) as an activation function that is responsible of activating nodes on
        the proceeding layer. ReLU --> returns max(0, input).
            :param state: input list that contains 4 elements, the car orientations and the 3 sensors values.
            :return: it returns the Q-values (predicted)
        """
        x = Func.relu(self.connectionOne(state))
        Qvalues = self.connectionTwo(x)

        # No activation function used in the qValues as softmax is used later
        return Qvalues


# Building the brain's Memory
class ExperienceMem(object):

    def __init__(self, MaxSize):
        """
        :param MaxSize: defines the max capacity of the program memory
        """
        self.MaxSize = MaxSize

        # The memory list which is used to make future decisions.
        self.recollection = []

    def stack(self, event):
        """
        it is used to add events to the memory "recollection", if the addition exceeds the size of the memory,
        then free the first index of the memory list
        :param event: the given tranisiotn to be inserted to the recollection list
        """
        self.recollection.append(event)
        MemLength = len(self.recollection)
        if MemLength > self.MaxSize:
            del self.recollection[0]

    def getRandomSample(self, batch_size):
        """
        This is used to get a random sample from the memory for training the NN.
        :param batch_size: the size of the training data
        """

        # Pytorch uses the Variable format that wraps tensors and gradients
        # cat(n, 0) is a concatenation method used to make the data acceptable by the Variable
        x = lambda n: Variable(torch.cat(n, 0))

        # Given the batch size, separate each group of L-states, actions, rewards and next states into batches.
        return map(x, zip(*random.sample(self.recollection, batch_size)))


# Deep Q Learning implementation
class Dqn():
    """
    This is the class where the deep learning takes place
    """

    def __init__(self, Size_of_inputs, OutputAction, gamma):
        """
        The initialization function
        :param Size_of_inputs: the input layer size which is 4 (orientation, 3 sensors)
        :param OutputAction: the output layer size which is 3, move 20/0/-20 degrees.
        :param gamma: used to identify the importance of the future rewards to the agent.
            It is a value between 0 and 1. Getting closer to zero, means that the agent
            cares about the current rewards and does not consider future rewards. Closer
            to one, is the opposite.
        """
        self.gamma = gamma
        self.rewardsList = []

        # Calling the previously defined NeuralNetwork method
        self.model = NeuralNetwork(Size_of_inputs, OutputAction)

        # Setting the agent's memory to hold up to 100,000 happened events
        self.recollection = ExperienceMem(100000)

        # Adam optimizer is used for weights update in backpropagation
        self.optimizer = optm.Adam(self.model.parameters(), lr=0.001)
        self.L_state = torch.Tensor(Size_of_inputs).unsqueeze(0)
        self.L_action = 0
        self.L_reward = 0

    def ActionSelection(self, state):
        """
        This function is used to select one of the output actions using softmax function
        :param state: the inputs, which are the orientation and the 3 sensors signals
        :return:
        """
        with torch.no_grad():
            # The softmax, given the inputs, returns probabilities for each action
            probabilities = Func.softmax(self.model.forwardPropagation(Variable(state)) * 300, dim = 1)
        output = probabilities.multinomial(num_samples=1)
        return output.data[0, 0]

    def Learning(self, current_states, next_states, rewards, output_actions):
        """
        The function used to improve the agent performance, it learns from the data, update the weights and
            decrease the loss. All the parameters all used to substitute in loss function.
        :param current_states: sample of input states
        :param next_states: sample of the reached next states
        :param rewards: rewards batch
        :param output_actions: actions batch
        """
        All_outputs = self.model.forwardPropagation(current_states).gather(1, output_actions.unsqueeze(1)).squeeze(1)
        Chosen_output = self.model.forwardPropagation(next_states).detach().max(1)[0]
        targets_batch = self.gamma * Chosen_output + rewards

        # Computing the loss
        loss = Func.smooth_l1_loss(All_outputs, targets_batch)

        # Setting gradient weights to zero
        self.optimizer.zero_grad()

        # BackPropagation
        loss.backward(retain_graph=True)

        # Updating weights
        self.optimizer.step()


    def weightsUpdate(self, updatedReward, newState):
        """
        This function is used to select an action and update the weights. It gathers all what is done till now.
        """

        # Putting the newState in the tensor form
        State_new = torch.Tensor(newState).float().unsqueeze(0)

        # Pushing into the memory, it pushes the last states, actions, rewards, and the newly got state.
        self.recollection.stack(
            (self.L_state, State_new, torch.LongTensor([int(self.L_action)]), torch.Tensor([self.L_reward])))

        # Calling the ActionSelection function to select the new output action
        output = self.ActionSelection(State_new)

        # Checks if the memory size exceeds 100, sample only a batch of 100
        recollectionLength = len(self.recollection.recollection)
        if recollectionLength > 100:
            statesBatch, nextStatesBatch, actionsBatch, rewardsBatch = self.recollection.getRandomSample(100)
            self.Learning(statesBatch, nextStatesBatch, rewardsBatch, actionsBatch)

        # Update L_action, L_state, L_reward, rewardsList with the new values
        self.L_action = output
        self.L_state = State_new
        self.L_reward = updatedReward
        self.rewardsList.append(updatedReward)

        # Maintains the rewards list updated with the most recent values
        rewardsListLen = len(self.rewardsList)
        if rewardsListLen > 1000:
            del self.rewardsList[0]

        # Return the selected Action
        return output


    def avg_score(self):
        """
        Used to get the average of the elements in the rewards list
        """
        total = sum(self.rewardsList)
        count = len(self.rewardsList) + 1.
        return total/count

    def SaveState(self):
        """
        Saves the final weights reached, so as to load it back from file, if needed. This eases the process,
        as there is no need to wait for the agent to learn from the beginning
        """
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'TrainedCarData.pth')

    def loadLastState(self):
        """
        Retreive the last saved file "TrainedCarData.pth" of the AI's weights.
        """
        if os.path.isfile('TrainedCarData.pth'):
            print("=> In Progress ... ")
            chkpnt = torch.load('TrainedCarData.pth')
            self.model.load_state_dict(chkpnt['state_dict'])
            self.optimizer.load_state_dict(chkpnt['optimizer'])
            print("Successfully loaded !")
        else:
            print("No Saved data is found...")