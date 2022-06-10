package DigitRecognition.MLP;

import java.text.DecimalFormat;

//
public class MultilayerPerceptron {
    //Pointer to the input layer of the network.
    private NetworkLayer inputLayerPointer;
    private NetworkLayer outputLayerPointer;

    //Globals containing both sets required for the network
    private int[][] trainSet;
    private int[][] testSet;

    //Constant contains the size of both sets.
    private int NUM_OF_SAMPLES = 2810;

    //Ratio of the error gradient that gets applied to the params in gradient descent.
    private double LEARNING_RATE = 1;
    //Amount of training passes during training.
    private int NUM_OF_EPOCHS = 2000;

    //Constant used in hidden layer dropout that determines the ratio of neurons that are retained.
    //because the testing forward pass reuses code from the train pass this variable must be modified to 1.0 prior to running the test.
    private double HIDDEN_NEURON_RETENTION_RATE = 0.5;

    //Constants for min max scalar input data constraints.
    private int MAX_INPUT_VALUE = 16;
    private int MIN_INPUT_VALUE = 0;

    DecimalFormat percentageFormat = new DecimalFormat("##.##");

    //Network constructor.
    public MultilayerPerceptron(int[] layerSizes , int[][] trainSet , int[][] testSet){
        //Populates global variables.
        this.trainSet = trainSet;
        this.testSet = testSet;
        //Creates the NN structure.
        inputLayerPointer = createNeuralNetwork(layerSizes);

        //Prints the neural network structure to the command line.
        NetworkLayer currentLayer = inputLayerPointer;
        System.out.println("Algorithm: MultiLayer Perceptron");
        System.out.println("Network Structure");
        while(currentLayer != null){
            System.out.print(currentLayer.layerLabel);
            System.out.println(" " +currentLayer.getLayerNeurons().length + " Nodes");
            currentLayer = currentLayer.nextLayer;
        }
        System.out.println();
        //Calls the function for network weight initialisation.
        initialiseNetworkWeights();
    }

    private void updateAllNeuronLearningRate(double newLearningRate){
        NetworkLayer currentLayer = inputLayerPointer;
        while(currentLayer != null){
            for(Neuron neuron : currentLayer.getLayerNeurons()){
                neuron.setLEARNING_RATE(newLearningRate);
            }
            currentLayer = currentLayer.nextLayer;
        }
    }

    //Method takes an array of layer sizes and initialises a network structure with empty neurons of that size.
    private NetworkLayer createNeuralNetwork(int[] layerSizes) {
        NetworkLayer inputLayerPointer = null;
        NetworkLayer previousLayer = null;
        for (int layerNum = 0; layerNum < layerSizes.length; layerNum++) {
            NetworkLayer currentLayer = null;
            //Creates an input layer which sets the previous layer to null and the label to input.
            if (layerNum == 0) {
                currentLayer = new NetworkLayer("input" , null , layerSizes[layerNum]);
                inputLayerPointer = currentLayer;
                previousLayer = inputLayerPointer;
            }
            //Creates a hidden layer setting the label to hidden, and sets the previous layers (next) pointer to this layer.
            else if(layerNum == layerSizes.length -1){
                currentLayer = new NetworkLayer("output" , previousLayer , layerSizes[layerNum]);
                previousLayer.setNextLayer(currentLayer);
            }
            //Creates an output layer.
            else{
                currentLayer = new NetworkLayer("hidden", previousLayer , layerSizes[layerNum]);
                previousLayer.setNextLayer(currentLayer);
                previousLayer = currentLayer;
                outputLayerPointer = currentLayer;
            }
        }
        //Returns a layer pointer to the input layer of the network.
        return inputLayerPointer;
    }


    //Method initialises network layers weight values layer by layer using a normal distribution of doubles in the range of 1 and -1.
    private void initialiseNetworkWeights(){
        //initialises weight vectors layer by layer.
        NetworkLayer currentLayer = inputLayerPointer;
        while(currentLayer != null){
            //Variable holds the num of neurons in the next layer so that a correct length weight vector can be assigned to them.
            int numOfNeuronsInNextLayer = (currentLayer.nextLayer == null) ? 0 : currentLayer.nextLayer.getLayerNeurons().length;
            for(int neuron = 0; neuron < currentLayer.getLayerNeurons().length; neuron++){
                //Initialises weight vectors inside each neuron based on the length of the following layer.
                currentLayer.getLayerNeurons()[neuron].initialiseNeuronWeights(numOfNeuronsInNextLayer);
                //gives each neuron the global learning rate.
                currentLayer.getLayerNeurons()[neuron].setLEARNING_RATE(LEARNING_RATE);
            }
            //Moves on to the next layer in the subsequent loop.
            currentLayer = currentLayer.nextLayer;
        }
    }

    //Method used the cross entropy loss on the predicted softmax activation output in order to receive a value representing the degree of error on the current sample prediction.
    private double calculateErrorOnSample(int expectedClassification , NetworkLayer outputLayer){
        return -Math.log(outputLayer.getLayerNeurons()[expectedClassification].getNeuronActivation());
    }

    //Method trains the network for set number of epochs.
    public void trainNetwork(){
        for(int epoch = 0; epoch < NUM_OF_EPOCHS; epoch++){
            //keeps track of the training progress.
            //System.out.println("Epoch num: " + epoch + "/" + NUM_OF_EPOCHS);

            /*
            if((epoch +1)%10000 == 0){
                updateAllNeuronLearningRate((LEARNING_RATE * 0.5));
            }

             */
            //Indicates training progress without cluttering the command line.
            if(epoch%20 ==0){
                System.out.print(".");
            }

            oneEpochTrainingPass();
            //testNetwork(testSet);
        }
    }

    //Method performs both a forward and a backpropagation pass on all samples in the train set.
    public void oneEpochTrainingPass(){
        double totalNetworkError = 0;
        for(int sample = 0; sample < trainSet.length; sample++){
            //Forward pass.
            NetworkLayer out = forwardPass(trainSet[sample]);
            //Error calculation.
            totalNetworkError += calculateErrorOnSample(trainSet[sample][trainSet[sample].length -1] , out);
            //Backpropagation.
            backpropagationOfError(out,trainSet[sample][trainSet[sample].length -1]);
        }
        //Backpropagation of the average network error to each one of the network parameters.
        updateParams(outputLayerPointer);
        //Returns the average network error on the epoch, This helps perceive improvement between epochs in the command line.
        //System.out.println("Average error on network: " + totalNetworkError/ trainSet.length);
    }

    //Method used to perform the forward pass of the network. It's responsible for populating all neuron inputs and activations.
    private NetworkLayer forwardPass(int[] trainSample){
        NetworkLayer currentLayer = inputLayerPointer;

        loop:
        while(currentLayer!= null){
            //Forward pass in the input layer.
            if(currentLayer.layerLabel.equals("input")){
                for(int inputNeuron = 0; inputNeuron < currentLayer.getLayerNeurons().length; inputNeuron++){
                    currentLayer.getLayerNeurons()[inputNeuron].setNeuronInput(trainSample[inputNeuron]);
                    //Currently, directly translates the raw input into activation as min max scalar resulted in worse performance.
                    currentLayer.getLayerNeurons()[inputNeuron].setNeuronActivation(currentLayer.getLayerNeurons()[inputNeuron].getNeuronInput());
                }
            }
            //Hidden layer forward pass.
            else{
                for(int hiddenNeuron = 0; hiddenNeuron < currentLayer.getLayerNeurons().length; hiddenNeuron++){
                    double dotProduct = 0;
                    for(int previousLayerNeuron = 0; previousLayerNeuron < currentLayer.previousLayer.getLayerNeurons().length; previousLayerNeuron++){
                        Neuron targetNeuron = currentLayer.previousLayer.getLayerNeurons()[previousLayerNeuron];
                        //Code required for dropout normalisation. It imposes a large performance penalty hence its been commented
                        /*
                        if(currentLayer.layerLabel.equals("hidden")){
                            double randomNum = Math.random();
                            if(randomNum < HIDDEN_NEURON_RETENTION_RATE){
                                dotProduct += targetNeuron.getNeuronActivation() * targetNeuron.getNeuronWeights()[hiddenNeuron];
                            }
                        }
                        else{
                            dotProduct += targetNeuron.getNeuronActivation() * targetNeuron.getNeuronWeights()[hiddenNeuron];
                        }

                         */
                        //Comment this if you want to use dropout.
                        dotProduct += targetNeuron.getNeuronActivation() * targetNeuron.getNeuronWeights()[hiddenNeuron];
                    }
                    currentLayer.getLayerNeurons()[hiddenNeuron].setNeuronInput(dotProduct + currentLayer.getLayerNeurons()[hiddenNeuron].getNeuronActivation());
                    //breaks if the layer is the output layer and all its neuron inputs were calculated.
                    if(currentLayer.layerLabel.equals("output")){
                        if(hiddenNeuron == currentLayer.getLayerNeurons().length-1){
                            break loop;
                        }
                    }
                    //if the neuron doesn't belong to the output layer then calculate its activation using the logistic function.
                    else{
                        currentLayer.getLayerNeurons()[hiddenNeuron].setNeuronActivation(modifiedLogisticFunction(currentLayer.getLayerNeurons()[hiddenNeuron].getNeuronInput()));
                    }
                }
            }
            //Goes to the next layer.
            currentLayer = currentLayer.nextLayer;
        }
        //Transforms the raw input of the output layer into activation values using the softmax function.
        double totalProbability = 0;
        //Loop contains total probability of all raw exponent values.
        for(int outputNeuron = 0; outputNeuron < currentLayer.getLayerNeurons().length; outputNeuron++){
            totalProbability += Math.pow(Math.E , currentLayer.getLayerNeurons()[outputNeuron].getNeuronInput());
        }
        //Loop gets the ratio of individual exponent raw values to the previously calculated total probability.
        for(int outputNeuron = 0; outputNeuron < currentLayer.getLayerNeurons().length; outputNeuron++){
            double softmaxOutput = Math.pow(Math.E , currentLayer.getLayerNeurons()[outputNeuron].getNeuronInput())/totalProbability;
            currentLayer.getLayerNeurons()[outputNeuron].setNeuronActivation(softmaxOutput);
        }
        //The current layer returned is the output layer which can be used in backpropagation to update neuron error params.
        return currentLayer;
    }



    //Method used to update neuron error values based on the current sample error.
    private void backpropagationOfError(NetworkLayer outputLayer , int classification){
        //Copy of the output layer pointer.
        NetworkLayer currentLayer = outputLayer;

        while(currentLayer != null){
            if(currentLayer.layerLabel.equals("input")){
                break;
            }
            if(currentLayer.layerLabel.equals("output")){
                //Variable holds differentiated value of error with respect to raw output layer input.
                for(int outputNeuron = 0; outputNeuron < currentLayer.getLayerNeurons().length; outputNeuron++){
                    //Handles output layer biases.
                    double costEntropyDerivative = (outputNeuron == classification)? outputLayer.getLayerNeurons()[outputNeuron].getNeuronActivation() - 1 : outputLayer.getLayerNeurons()[outputNeuron].getNeuronActivation();
                    currentLayer.getLayerNeurons()[outputNeuron].addNeuronBias(costEntropyDerivative);
                    for(int outputWeights = 0; outputWeights < currentLayer.previousLayer.getLayerNeurons().length; outputWeights++){
                        currentLayer.previousLayer.getLayerNeurons()[outputWeights].addNeuronWeightError( currentLayer.previousLayer.getLayerNeurons()[outputWeights].getNeuronActivation() * costEntropyDerivative , outputNeuron);
                        double previousLayerError = costEntropyDerivative *  currentLayer.previousLayer.getLayerNeurons()[outputWeights].getNeuronWeights()[outputNeuron];
                        currentLayer.previousLayer.getLayerNeurons()[outputWeights].addNeuronError(previousLayerError);
                    }
                }
            }
            else{
                for(int hiddenNeuron = 0; hiddenNeuron < currentLayer.getLayerNeurons().length; hiddenNeuron++){
                    //Handles output layer biases.
                    double neuronError = currentLayer.getLayerNeurons()[hiddenNeuron].getNeuronError();
                    currentLayer.getLayerNeurons()[hiddenNeuron].addNeuronBias(neuronError);
                    for(int hiddenWeights = 0; hiddenWeights < currentLayer.previousLayer.getLayerNeurons().length; hiddenWeights++){
                        currentLayer.previousLayer.getLayerNeurons()[hiddenWeights].addNeuronWeightError( currentLayer.previousLayer.getLayerNeurons()[hiddenWeights].getNeuronActivation() * neuronError , hiddenNeuron);
                        double previousLayerError = neuronError *  currentLayer.previousLayer.getLayerNeurons()[hiddenWeights].getNeuronWeights()[hiddenNeuron];
                        currentLayer.previousLayer.getLayerNeurons()[hiddenWeights].addNeuronError(previousLayerError);
                    }
                }
            }
            currentLayer = currentLayer.previousLayer;
        }
    }


    //Method that causes every neuron in all layers except the input layer to perform gradient descent on the average error derivative.
    private void updateParams(NetworkLayer inLayer){
        NetworkLayer currentLayer = inLayer;

        while(currentLayer != null){
            if(currentLayer.layerLabel.equals("input")){
                break;
            }
            for(int neuron = 0; neuron < currentLayer.getLayerNeurons().length; neuron++){
                currentLayer.getLayerNeurons()[neuron].backPropagateNeuronErrors();
            }
            currentLayer = currentLayer.nextLayer;
        }
    }

    //Min max scalar used to constrain the input data between 0 and 1 this will normalise the outputs from the input layers.
    private double minMaxScalar(int inputVal){
        return (inputVal - MIN_INPUT_VALUE + 0.0)/(MAX_INPUT_VALUE - MIN_INPUT_VALUE + 0.0);
    }

    //Returns the derivative of the modified logistic used in this algorithm provided a neuron raw input.
    private double modifiedLogisticDerivative(double input){
        double top = 2 * Math.exp(input);
        double bottom = Math.pow((Math.exp(input) + 1) , 2);
        if(bottom == 0){
            System.out.println("fire");
        }
        return top/bottom;
    }

    //Logistic activation function constrained in the range of 1 and -1. This is done so that negative values can be achieved in the output layer.
    private double modifiedLogisticFunction(double rawNeuronInput){
        return (2.0/(1.0 + Math.exp(-rawNeuronInput))) -1;
    }

    //Method contains the testing forward pass that does not include backpropagation.
    public double testNetwork(int[][] set){
        HIDDEN_NEURON_RETENTION_RATE = 1.0;
        //variable contains the number of correctly classified samples in the test set.
        int correctClassifications = 0;

        for(int testSample = 0; testSample < NUM_OF_SAMPLES; testSample++){
            //Forward pass.
            NetworkLayer output = forwardPass(set[testSample]);
            //The classification selected has the greatest softmax probability, This is essentially an implementation of argMax.
            double softmaxProbability = 0;
            int indexOfMaxProbability = -1;

            //Iterates through the output layer and selects the greatest softmax probability for classification.
            for(int outLayer = 0; outLayer < output.getLayerNeurons().length; outLayer++){
                if(output.getLayerNeurons()[outLayer].getNeuronActivation() > softmaxProbability){
                    softmaxProbability = output.getLayerNeurons()[outLayer].getNeuronActivation();
                    indexOfMaxProbability = outLayer;
                }
            }
            //Verifies whether the classification was correct.
            if(indexOfMaxProbability == set[testSample][set[testSample].length -1]){
                correctClassifications++;
            }
        }
        //Returns the percentage of correct classifications on the test set.
        double percentageCorrect = (0.0 + correctClassifications/(NUM_OF_SAMPLES + 0.0))*100;
        System.out.println("\nTest Accuracy: " + percentageFormat.format(percentageCorrect) + "% " + "Correct Classifications: " + correctClassifications);
        return percentageCorrect;
    }




}
