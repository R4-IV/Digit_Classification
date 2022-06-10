package DigitRecognition.MLP;

//This class holds all network neuron operations.
public class Neuron {
    //constant allows for the neuron to perform mean error loss backpropagation.
    private int NUM_OF_SAMPLES = 2810;
    //Initial values for learning rate, allows for each layer to lear at a different rate (this is not utilised).
    private double LEARNING_RATE = 0;
    //Holds the raw value entering the neuron.
    private double neuronInput;

    //Holds the value once an activation function transforms the input for example (minMax scalar, softmax, sigmoid).
    private double neuronActivation;

    //Holds the values between this neuron and all the next layers neurons (dense layers).
    private double[] neuronWeights;

    //Bias on the neuron initialised to 0, mostly because the input layer does not use it in the calculations.
    private double neuronBias = 0;

    //Params to update weights and biases in backpropagation.
    private double[] neuronWeightErrors;
    private double neuronBiasError = 0;
    private double neuronError = 0;

    //Neuron constructor.
    public Neuron(){}

    //Method creates initial weight values for the network. They take values between 1 and -1.
    public void initialiseNeuronWeights(int numOfWeights){
        neuronWeights = new double[numOfWeights];
        for(int neuronWeightNum = 0; neuronWeightNum < neuronWeights.length; neuronWeightNum++){
            neuronWeights[neuronWeightNum] = ((Math.random() * 2)-1);
            //Alternate weight configuration to constrain the between 0 and 1. This was used when exploring activation functions that did not like negative values.
            //neuronWeights[neuronWeightNum] = Math.random();
        }
        //Initialises an array of errors that are the same size as weight vector. This array is used in gradient descent
        neuronWeightErrors = new double[numOfWeights];
    }

    //Method sets the neuron learning rate, This isn't in the constructor because It's useful to dynamically change the learning rate throughout a training session.
    public void setLEARNING_RATE(double LEARNING_RATE) {
        this.LEARNING_RATE = LEARNING_RATE;
    }

    //This method will be used to load weights from a file provided they are converted to a double array.
    public void setNeuronWeights(double[] neuronWeights) {
        this.neuronWeights = neuronWeights;
    }

    //This method will be used to set bias values when loaded from a file or updated in backpropagation.
    public void setNeuronBias(double neuronBias) {
        this.neuronBias = neuronBias;
    }

    //Sets the raw input of this neuron, In the input layer this will be the raw data. Other layers will have neuron activation . weight vector + bias.
    public void setNeuronInput(double neuronInput) {
        this.neuronInput = neuronInput;
    }

    //Sets the neuron activation after a transformative function is applied to the neuron input.
    public void setNeuronActivation(double neuronActivation) {
        this.neuronActivation = neuronActivation;
    }

    //Getter methods for forward pass parameters.
    public double getNeuronBias() {
        return neuronBias;
    }

    public double getNeuronInput() {
        return neuronInput;
    }

    public double getNeuronActivation() {
        return neuronActivation;
    }

    public double[] getNeuronWeights() {
        return neuronWeights;
    }

    //Methods used in the backpropagation of bias error
    public void addNeuronBias(double neuronBiasAddition){
        neuronBiasError+= neuronBiasAddition;
    }

    public void addNeuronWeightError(double neuronErrorWeightAddition, int weightIndex){
        neuronWeightErrors[weightIndex] += neuronErrorWeightAddition;
    }

    //The cumulative error from the following layer. Having this parameter permits me to the backpropagation layer by layer.
    public void addNeuronError(double neuronErrorAddition){
        neuronError += neuronErrorAddition;
    }

    public double getNeuronError() {
        return neuronError;
    }


    //This method takes the error metrics and averages them over an epoch 2810 samples. The results are then plugged into the gradient descent algorithm.
    public void backPropagateNeuronErrors(){
        //Updates the neuron bias and resets the error metric so that subsequent epochs are not affected by its value.
        neuronBias -= (neuronBiasError/NUM_OF_SAMPLES) * LEARNING_RATE;
        neuronBiasError = 0;

        //Updates the weight vector with gradient descent and resets the error vector for the next epoch.
        for(int weight = 0; weight < neuronWeights.length; weight++){
            neuronWeights[weight] -= (neuronWeightErrors[weight]/NUM_OF_SAMPLES) * LEARNING_RATE;
        }
        neuronWeightErrors = new double[neuronWeights.length];
        //Resets the following layer neuron error so that subsequent epochs are unaffected.
        neuronError = 0;
    }

}
