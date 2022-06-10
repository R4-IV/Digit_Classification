package DigitRecognition;

import DigitRecognition.MLP.MultilayerPerceptron;
import DigitRecognition.NearestAlgorithms.NearestCentroid;
import DigitRecognition.NearestAlgorithms.NearestNeighbour;

import java.text.DecimalFormat;

public class Main {

    public static void main(String[] args) {
        DataLoader dt = new DataLoader("setOne.txt" , "setTwo.txt", 2810);
        DataSets sets = dt.returnSets();
        //Formats average percentage for mlp and nearest centroid.
        DecimalFormat percentageFormat = new DecimalFormat("##.##");

        //Nearest Neighbour algorithm has both folds inbuilt.
        NearestNeighbour nn = new NearestNeighbour(sets);

        //Nearest Centroid algorithm.
        System.out.println();
        System.out.println("Algorithm: Nearest Centroid");
        System.out.print("First");
        NearestCentroid nearestCentroid = new NearestCentroid(dt.returnSets().getSetOne(), dt.returnSets().getSetTwo());
        System.out.print("Second");
        NearestCentroid nearestCentroidTwoFold = new NearestCentroid(dt.returnSets().getSetTwo(), dt.returnSets().getSetOne());
        System.out.println("Average Accuracy: " + percentageFormat.format((nearestCentroid.getPercentageClassified() + nearestCentroidTwoFold.getPercentageClassified())/2) +"%");


        //Modify the structure of the mlp using this array. The first and last layer should remain constant. Additional layers can be inserted.
        int[] layerStructure = new int[]{64,300,10};
        System.out.println();
        System.out.println("Fold One Test");
        MultilayerPerceptron varianceMultilayerPerceptron = new MultilayerPerceptron(layerStructure , sets.getSetOne(), sets.getSetTwo());
        System.out.println(".................................................................................................... Total Progress");
        varianceMultilayerPerceptron.trainNetwork();
        double test1 = varianceMultilayerPerceptron.testNetwork(sets.getSetTwo());
        System.out.println();
        System.out.println("Fold Two Test");
        MultilayerPerceptron varianceMultilayerPerceptronSecondFold = new MultilayerPerceptron(layerStructure , sets.getSetTwo() , sets.getSetOne());
        System.out.println(".................................................................................................... Total Progress");
        varianceMultilayerPerceptronSecondFold.trainNetwork();
        double test2 = varianceMultilayerPerceptronSecondFold.testNetwork(sets.getSetOne());

        System.out.println("Average Accuracy: " + percentageFormat.format((test1 + test2)/2) +"%");
    }
}
