package DigitRecognition.MLP;

//This class stores a network layer object which corresponds to one of the MLP layers (input , hidden , output)
public class NetworkLayer {
    //Layer label used to identify what type of layer it is in certain operations inside the MultilayerPerceptron.
    public String layerLabel;

    //Pointer to the previous layer, if the previous layer is null the current layer is the input layer.
    public NetworkLayer previousLayer = null;
    //Pointer to the next layer, if the next layer is null the current layer is the output layer.
    public NetworkLayer nextLayer = null;

    //Variable holds the Neuron array of all neurons that are members of this layer.
    private Neuron[] layerNeurons;

    //Constructor attaches a label, previous layer, and initialises the current layer to be a given size.
    public NetworkLayer( String layerLabel, NetworkLayer previousLayer , int layerSize){
        this.layerLabel = layerLabel;
        this.previousLayer = previousLayer;
        initLayerNeurons(layerSize);
    }

    //Getter method to retrieve layer Neurons so that operations can be done on it.
    public Neuron[] getLayerNeurons() {
        return layerNeurons;
    }

    //Method to set this layers next layer. This can't be done in the constructor as the layer does not exist yet.
    public void setNextLayer(NetworkLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    //Method creates new neuron objects inside the layer's neuron array of the provided size in the constructor.
    private void initLayerNeurons(int layerSize){
        layerNeurons = new Neuron[layerSize];
        for(int neuron = 0; neuron < layerSize; neuron++){
            layerNeurons[neuron] = new Neuron();
        }
    }

}
