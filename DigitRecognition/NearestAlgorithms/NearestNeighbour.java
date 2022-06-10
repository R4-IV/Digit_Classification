package DigitRecognition.NearestAlgorithms;

import DigitRecognition.DataSets;

import java.text.DecimalFormat;

//Nearest Neighbour algorithm classifies an image of hand drawn digit from the test set by figuring out what the closest train example to it is and assuming its classification.
public class NearestNeighbour {

    //Contains the train set matrix.
    int[][] trainSet;
    //Contains the test set matrix.
    int[][] testSet;
    //Used entirely to trim the percentage double to a format suitable for percentage display.
    DecimalFormat percentageFormat = new DecimalFormat("##.##");

    //Constructor
    public NearestNeighbour(DataSets dataSets){
        System.out.println("Algorithm: Nearest Neighbour");
        System.out.print("First Test Accuracy: ");
        trainSet = dataSets.getSetOne();
        testSet = dataSets.getSetTwo();
        double testOne = classify();
        //Swaps the sets around for 2-fold cross validation.
        System.out.print("Second Test Accuracy: ");
        trainSet = dataSets.getSetTwo();
        testSet = dataSets.getSetOne();
        double testTwo = classify();
        //Averages the performance of both tests.
        System.out.print("Average Accuracy: ");
        System.out.println(percentageFormat.format(((testOne + testTwo)/2)) + "%");
    }


    //Method that checks the distance between all test examples with the entirety of the train set and assumes the classification of the nearest train point.
    public double classify(){
        //Variable holds the amount of correct classifications in the current test.
        int classifiedCorrectly = 0;
        //Constant that holds the index position of the classifier in train and test vectors.
        int CLASSIFIER = 64;

        //2d for loop test all test vector against the entirety of the train set for every test vector.
        for(int testExample = 0; testExample < testSet.length; testExample++){
            //Variable holds the shortest distance between the current test vector and a train vector.
            double distanceToNearestPoint = Double.MAX_VALUE;
            //Variable holds the classification of the closest train vector.
            int nearestPointClassification = -1;
            //Train set loop.
            for(int trainExample = 0; trainExample < trainSet.length; trainExample++){
                //Calculates the 64d euclidean distance between the current test vector and the current train vector.
                double distanceToCurrentPoint = xDimensionalEuclideanDistance(trainSet[trainExample], testSet[testExample]);
                //Checks if the distance between the current points is shorter than currently the shortest reported value.
                if( distanceToCurrentPoint <= distanceToNearestPoint){
                    //Updates the shortest values.
                    distanceToNearestPoint = distanceToCurrentPoint;
                    nearestPointClassification = trainSet[trainExample][CLASSIFIER];
                }
            }
            //Updates the amount of correct classification if the predicted classification matches the test vector classification.
            if(nearestPointClassification == testSet[testExample][CLASSIFIER]){
                classifiedCorrectly+= 1;
            }
        }
        //Calculates the double value of the ratio between correct classifications and total classifications.
        double percentage = ((0.0 + classifiedCorrectly)/testSet.length) * 100;
        //Prints the percentage accuracy of the current test.
        System.out.println(percentageFormat.format(percentage) + "%   Correct Classifications: " + classifiedCorrectly);
        //Returns the double percentage value so that the average accuracy can be calculated in the constructor.
        return percentage;
    }

    //Method calculates the euclidean distance between two 64d vectors.
    private double xDimensionalEuclideanDistance(int[] trainSample , int[] testSample){

        double accumulator = 0;
        //Vector size ensure that only the first 64 values are calculated in the euclidean distance.
        int INPUT_SIZE = 64;
        //Iterates through all 64 coordinate pairs and calculates the difference squared.
        for(int coordinatePosition = 0; coordinatePosition < INPUT_SIZE; coordinatePosition++){
            accumulator += Math.pow((trainSample[coordinatePosition] - testSample[coordinatePosition]),2);
        }
        //Returns the euclidean distance satisfying the equation sqrt(Δx^2 + Δy^2).
        return Math.sqrt(accumulator);
    }
}
