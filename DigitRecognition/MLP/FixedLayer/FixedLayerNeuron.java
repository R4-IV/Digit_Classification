package DigitRecognition.MLP.FixedLayer;

//Class contains the Neuron implementation for the Fixed Layer MLP prototype.

public class FixedLayerNeuron {
    //Raw activation stands for input into the neuron prior to the application of the logistic function. This is needed as the derivative of the logistic requires this input.
    private double rawActivation;
    //Neuron activation after running it though a logistic function.
    private double activation;
    //Neuron bias.
    private double bias = 0;

    //Setter Methods.
    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public void setRawActivation(double rawActivation) {
        this.rawActivation = rawActivation;
    }

    //Getter Methods.
    public double getBias() {
        return bias;
    }

    public double getActivation() {
        return activation;
    }

    public double getRawActivation() {
        return rawActivation;
    }
}
