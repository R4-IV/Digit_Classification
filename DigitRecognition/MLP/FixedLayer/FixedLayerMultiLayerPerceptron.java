package DigitRecognition.MLP.FixedLayer;

//Class contains a hard coded single hidden layer MLP
public class FixedLayerMultiLayerPerceptron {

    //Network Layer constants.
    private int INPUT_LAYER_SIZE = 64;
    private int HIDDEN_LAYER_SIZE = 200;
    private int OUTPUT_LAYER_SIZE = 10;
    private int NUM_OF_SAMPLES = 2810;


    //Dropout rate of the hidden layer for can help with over fitting by simulating parallel training of several layer configurations in parallel.
    private double HIDDEN_LAYER_DROPOUT = 0.0;

    //Constants for min max scalar input data constraints.
    private int MAX_INPUT_VALUE = 16;
    private int MIN_INPUT_VALUE = 0;

    //Network layer neuron containers.
    private FixedLayerNeuron[] inputLayer;
    private FixedLayerNeuron[] hiddenLayer;
    private FixedLayerNeuron[] outputLayer;

    //Network weight matrices.
    private double[][] inputHiddenWeights;
    private double[][] hiddenOutputWeights;

    //Rate at which gradient descent is updated. This network has some sort of bug that result in math errors if the learning rate is too high.
    private double LEARNING_RATE = 0.001;

    //Train and test sets global access containers.
    private int[][] trainSet;
    private int[][] testSet;

    //Arrays containing the errors on each of the network layer parameters.
    private double[] deltaOutputBiases;
    private double[][] deltaHiddenOutputWeights;
    private double[] deltaHiddenBiases;
    private double[][] deltaInputHiddenWeights;

    //Constructor initialises the network layers and weight connections.
    public FixedLayerMultiLayerPerceptron(int[][] trainSet , int[][] testSet){

        this.trainSet = trainSet;
        this.testSet = testSet;
        //Network layers.
        inputLayer = initialiseLayerWithEmptyNeurons(INPUT_LAYER_SIZE);
        hiddenLayer = initialiseLayerWithEmptyNeurons(HIDDEN_LAYER_SIZE);
        outputLayer = initialiseLayerWithEmptyNeurons(OUTPUT_LAYER_SIZE);
        //Weight connections.
        inputHiddenWeights = initialiseWeightMatrix(INPUT_LAYER_SIZE , HIDDEN_LAYER_SIZE);
        hiddenOutputWeights = initialiseWeightMatrix(HIDDEN_LAYER_SIZE , OUTPUT_LAYER_SIZE);

    }


    //Creating an array of Neurons will result in an array of null objects. This function populates them with empty values that can be set during the network forward pass.
    private FixedLayerNeuron[] initialiseLayerWithEmptyNeurons(int NeuronArrSize){
        FixedLayerNeuron[] neuronArr = new FixedLayerNeuron[NeuronArrSize];
        for(int neuron = 0; neuron < NeuronArrSize; neuron++){
            neuronArr[neuron] = new FixedLayerNeuron();
        }
        return neuronArr;
    }


    //Initialises a matrix of weights given the dimensions between in the range of -1 and 1
    private double[][] initialiseWeightMatrix(int matrixDimensionOne , int matrixDimensionTwo){
        double[][] weightMatrix = new double[matrixDimensionOne][matrixDimensionTwo];

        for(int firstDimension = 0; firstDimension < matrixDimensionOne; firstDimension++){
            for(int secondDimension = 0; secondDimension < matrixDimensionTwo; secondDimension++){
                weightMatrix[firstDimension][secondDimension] = ((Math.random() * 2)-1);
            }
        }
        return weightMatrix;
    }

    //Method used to train the network on the training set for a fixed number of epochs.
     public void multiEpochTraining(int numOfEpochs){
        double lowestEpochError = Double.MAX_VALUE;
        for(int epoch = 0; epoch < numOfEpochs; epoch++){
            System.out.println("Epoch Num: " + (epoch+1) + "/" + numOfEpochs);
            double errorOnCurrentSample = trainForOneEpoch();
        }
     }

     //Method used for in the process of a single epoch training.
     public double trainForOneEpoch(){
        //Reset the gradients for each parameter at the beginning of the epoch
        deltaHiddenOutputWeights = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];
        deltaInputHiddenWeights = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
        deltaOutputBiases = new double[OUTPUT_LAYER_SIZE];
        deltaHiddenBiases = new double[HIDDEN_LAYER_SIZE];

        //Variable keeps track of the mean cross entropy error across the network.
        double totalNetworkCost = 0;

        for(int numTrainSamples = 0; numTrainSamples < trainSet.length; numTrainSamples++){
            //variable holds the classification for the current sample.
            int classification = trainSet[numTrainSamples][INPUT_LAYER_SIZE];
            forwardTrainingPass(trainSet[numTrainSamples] , true);
            totalNetworkCost += calculateCostOnSample(classification);
        }

        //Returns the average error on the network.
        System.out.println("Average error on Network: " + (totalNetworkCost/NUM_OF_SAMPLES));
        return (totalNetworkCost/NUM_OF_SAMPLES);
    }

    //Method performs the forward pass in network if training mode true is provided it will also perform backpropagation.
    private void forwardTrainingPass(int[] trainSample , boolean trainingMode){
        //initialise input layer with normalised vector input.
        for(int inputNeuron = 0; inputNeuron < INPUT_LAYER_SIZE; inputNeuron++){
            inputLayer[inputNeuron].setActivation(minMaxScalar(trainSample[inputNeuron]));
        }
        //populate hidden layer.
        for(int hiddenLayerNode = 0; hiddenLayerNode < HIDDEN_LAYER_SIZE; hiddenLayerNode++){
            if(isNodeDropped(HIDDEN_LAYER_DROPOUT)){
                hiddenLayer[hiddenLayerNode].setRawActivation(0);
            }
            else{
                hiddenLayer[hiddenLayerNode].setRawActivation(hiddenLayer[hiddenLayerNode].getBias() + nextNeuronActivation(hiddenLayerNode ,inputHiddenWeights, inputLayer));
            }
            hiddenLayer[hiddenLayerNode].setActivation(modifiedLogisticFunction(hiddenLayer[hiddenLayerNode].getRawActivation()));
        }
        //populate outputLayer.
        for(int outputLayerNode = 0; outputLayerNode < OUTPUT_LAYER_SIZE; outputLayerNode++){
            outputLayer[outputLayerNode].setRawActivation(outputLayer[outputLayerNode].getBias() + nextNeuronActivation(outputLayerNode , hiddenOutputWeights , hiddenLayer));
        }
        softMaxOutputLayer(outputLayer);
        //Training mode calls the backpropagation method.
        if(trainingMode){
            backpropagationOfError(trainSample[INPUT_LAYER_SIZE]);
            //Gradient descent output biases.
            for(int outputLayerBiases = 0; outputLayerBiases < OUTPUT_LAYER_SIZE; outputLayerBiases++){
                outputLayer[outputLayerBiases].setBias(outputLayer[outputLayerBiases].getBias() - ((deltaOutputBiases[outputLayerBiases]/NUM_OF_SAMPLES) * LEARNING_RATE));
            }
            //Gradient descent on the output hidden weights.
            for(int hiddenLayerComponent = 0; hiddenLayerComponent < HIDDEN_LAYER_SIZE; hiddenLayerComponent++){
                for(int outputLayerComponent = 0; outputLayerComponent < OUTPUT_LAYER_SIZE; outputLayerComponent++){
                    hiddenOutputWeights[hiddenLayerComponent][outputLayerComponent] -= (deltaHiddenOutputWeights[hiddenLayerComponent][outputLayerComponent]/NUM_OF_SAMPLES) * LEARNING_RATE;
                }
            }
            //Gradient descent on hidden layer biases.
            for(int hiddenBias = 0; hiddenBias < HIDDEN_LAYER_SIZE; hiddenBias++){
                hiddenLayer[hiddenBias].setBias(hiddenLayer[hiddenBias].getBias() - (deltaHiddenBiases[hiddenBias] * LEARNING_RATE));
            }
            //Gradient descent on input hidden weights.
            for(int inputComponent = 0; inputComponent < INPUT_LAYER_SIZE; inputComponent++){
                for(int hiddenComponent = 0; hiddenComponent < HIDDEN_LAYER_SIZE; hiddenComponent++){
                    inputHiddenWeights[inputComponent][hiddenComponent] -= (deltaInputHiddenWeights[inputComponent][hiddenComponent] * LEARNING_RATE);
                }
            }
        }
    }

    //Method uses chain rule derivatives starting at the last layer and moving to the input layer.
    private void backpropagationOfError(int classification){

        //backpropagation on the output layer biases
        for(int outputBias = 0; outputBias < OUTPUT_LAYER_SIZE; outputBias++){
            double dCEdRaw = (outputBias == classification) ? (outputLayer[outputBias].getActivation() -1) : outputLayer[outputBias].getRawActivation();
            //backpropagation of error to the output layer biases.
            deltaOutputBiases[outputBias] += dCEdRaw;
            //backpropagation of error onto the hidden layer weights.
            for(int outputHiddenWeights = 0; outputHiddenWeights < HIDDEN_LAYER_SIZE; outputHiddenWeights++){
                deltaHiddenOutputWeights[outputHiddenWeights][outputBias] += dCEdRaw * hiddenLayer[outputHiddenWeights].getActivation();
                //hidden layer biases
                deltaHiddenBiases[outputHiddenWeights] += dCEdRaw * hiddenOutputWeights[outputHiddenWeights][outputBias] * sigmoidDerivative(hiddenLayer[outputHiddenWeights].getActivation());
                for(int inputWeights = 0; inputWeights < INPUT_LAYER_SIZE; inputWeights++){
                    //inputLayerWeights
                    deltaInputHiddenWeights[inputWeights][outputHiddenWeights] += dCEdRaw * hiddenOutputWeights[outputHiddenWeights][outputBias] * sigmoidDerivative(hiddenLayer[outputHiddenWeights].getActivation()) * inputLayer[inputWeights].getActivation();
                }
            }
        }
    }

    //Returns the dot product from the previous layer and the corresponding weight matrix.
    private double nextNeuronActivation(int neuronNum, double[][] weightMatrix , FixedLayerNeuron[] previousNeuronLayer){
        double dotProduct = 0;
        for(int neuronWeightPair = 0; neuronWeightPair < previousNeuronLayer.length; neuronWeightPair++){
            dotProduct += (previousNeuronLayer[neuronWeightPair].getActivation() * weightMatrix[neuronWeightPair][neuronNum]);
        }
        return dotProduct;
    }


    //Uses cross entropy of the target prediction for the measure of network error.
    private double calculateCostOnSample(int expectedClassification){
        return -(Math.log(outputLayer[expectedClassification].getActivation()));
    }



    //Logistic activation function constrained in the range of 1 and -1. This is done so that negative values can be achieved in the output layer.
    private double modifiedLogisticFunction(double rawNeuronInput){
        return (2.0/(1.0 + Math.exp(-rawNeuronInput))) -1;
    }



    //Min max scalar used to constrain the input data between 0 and 1 this will normalise the outputs from the input layers.
    private double minMaxScalar(int inputVal){
        return (inputVal - MIN_INPUT_VALUE + 0.0)/(MAX_INPUT_VALUE - MIN_INPUT_VALUE + 0.0);
    }


    //Soft max function for output layer probabilistic distribution.
    private void softMaxOutputLayer(FixedLayerNeuron[] outputLayer){
        double totalProbability = 0;
        //sets the total probability for the softmax distribution
        for(int rawOutputActivation = 0; rawOutputActivation < OUTPUT_LAYER_SIZE; rawOutputActivation++){
            totalProbability += Math.exp(outputLayer[rawOutputActivation].getRawActivation());
        }
        for(int softMaxProbability = 0; softMaxProbability < OUTPUT_LAYER_SIZE; softMaxProbability++){
            outputLayer[softMaxProbability].setActivation(Math.exp(outputLayer[softMaxProbability].getRawActivation())/totalProbability);
        }
    }

    //Method returns the maximum value classification from its softmax input.
    private int argMax(){
        double softMaxThreshold = 0;
        int classification = -1;

        for(int outputLayerNode = 0; outputLayerNode < OUTPUT_LAYER_SIZE; outputLayerNode++){
            if(outputLayer[outputLayerNode].getActivation() > softMaxThreshold){
                softMaxThreshold = outputLayer[outputLayerNode].getActivation();
                classification = outputLayerNode;
            }
        }
        return classification;
    }

    //Method used to test the network after it has been trained.
    public void runTestData(int[][] set){
        int correctClassifications = 0;
        for(int testSample = 0; testSample < NUM_OF_SAMPLES; testSample++){
            forwardTrainingPass(set[testSample],false);
            if(argMax() == set[testSample][INPUT_LAYER_SIZE]){
                correctClassifications++;
            }
        }
        System.out.println(correctClassifications);
        System.out.println("Test Results: " + ((0.0 + correctClassifications)/(0.0 +NUM_OF_SAMPLES) * 100));
    };

    //The derivative of the logistic function used for hidden layer activation.
    private double sigmoidDerivative(double input){
        return ((2 * Math.pow(Math.E, input))/(Math.pow(Math.E , input)) +1);
    }

    //Method used to probabilistically determine whether the current node should be ignored in forward progation.
    //This method was supposed to be used in dropout normalisation.
    private boolean isNodeDropped(double dropoutProbability){
        if(dropoutProbability == 0){
            return false;
        }
        double randomNum = Math.random();
        if(dropoutProbability > randomNum){
            return true;
        }
        return false;
    }

}
