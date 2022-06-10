package DigitRecognition.NearestAlgorithms;

import java.text.DecimalFormat;

//Class calculates train set category centroids and uses the distance to the nearest one for classification.
public class NearestCentroid {

    //Variable holds reference to centroid 2d array.
    double[][] centroidCoordinates;
    //Formats percentage output.
    DecimalFormat percentageFormat = new DecimalFormat("##.##");

    private double percentageClassified = 0;

    //Constructor calculates the centroids and passes them onto the classification function.
    public NearestCentroid(int[][] trainSet , int[][] testSet){
        this.centroidCoordinates = calculateCentroids(trainSet);
        classifyViaNearestCentroid(testSet);
    }

    //Method used to verify the accuracy of the nearest centroid classifier.
    public void classifyViaNearestCentroid(int[][] set){
        //Keeps track of the number of correct classifications made.
        int correctClassification = 0;
        //Constant tracks the position of the sample classification in the test sample.
        int INDEX_OF_CLASSIFICATION = 64;
        //For loop iterates through all 2810 test samples.
        for(int testSample = 0; testSample < set.length; testSample++){
            //Initialises the closest centroid distance to the max double value which guarantees that the next centroid will have a distance less than this value.
            double closestDistance = Double.MAX_VALUE;
            int predictedClassification = -1;
            //For loop iterates over all 10 centroid coordinates and compares them by euclidean distance to current sample.
            for(int centroid = 0; centroid < 10; centroid++){
                if(euclideanDistance(set[testSample] , centroidCoordinates[centroid]) < closestDistance){
                    closestDistance = euclideanDistance(set[testSample] , centroidCoordinates[centroid]);
                    predictedClassification = centroid;
                }
            }
            //If the closest centroid matches the classification in the Index of classification then correct classifications are incremented.
            if(predictedClassification == set[testSample][INDEX_OF_CLASSIFICATION]){
                correctClassification++;
            }
        }
        //Returns num and percentage of correct classifications.
        System.out.println("Test Accuracy: " + percentageFormat.format(((correctClassification + 0.0)/2810) *100) + "% " + "Correct Classifications: " + correctClassification);
        percentageClassified = (((correctClassification + 0.0)/2810) *100);
    }


    public double getPercentageClassified() {
        return percentageClassified;
    }

    //Method accepts a train set and return the coordinates of all 10 category centroids.
    private double[][] calculateCentroids(int[][] trainSet){
        //Constants used for array initialisation.
        int NUM_OF_CLASSIFICATIONS = 10;
        int NUM_OF_COORDINATES = 64;

        //Array that holds the raw centroid coordinate values.
        double[][] centroidCoordinates = new double[NUM_OF_CLASSIFICATIONS][NUM_OF_COORDINATES];
        //Array holds the num of samples per category so that a mean coordinate values can be calculated on the above array.
        int[] numOfSamplesPerClassification = new int[NUM_OF_CLASSIFICATIONS];

        //For loop accumulates all the coordinates from each matching category and also adds the amount of samples per that category.
        for(int trainSample = 0; trainSample < trainSet.length; trainSample++){
            for(int trainSampleCoordinate = 0; trainSampleCoordinate < NUM_OF_COORDINATES; trainSampleCoordinate++){
                centroidCoordinates[trainSet[trainSample][NUM_OF_COORDINATES]][trainSampleCoordinate] += trainSet[trainSample][trainSampleCoordinate];
            }
            numOfSamplesPerClassification[trainSet[trainSample][NUM_OF_COORDINATES]] += 1;
        }

        //For loop used to transform the summed values in centroidCoordinate array into mean values.
        for(int classification = 0; classification < NUM_OF_CLASSIFICATIONS; classification++){
            for(int coordinate = 0; coordinate < NUM_OF_COORDINATES; coordinate++){
                centroidCoordinates[classification][coordinate] /= numOfSamplesPerClassification[classification];
            }
        }
        //Return the array of calculated centroid coordinates.
        return centroidCoordinates;
    }


    //Method calculates euclidean distance between train sample and centroid coordinates
    private double euclideanDistance(int[] trainSample , double[] centroidCoordinate){
        //Accumulator used to keep track of coordinate deltas.
        double accumulator = 0;
        //Vector size ensure that only the first 64 values are calculated in the euclidean distance.
        int INPUT_SIZE = 64;
        //Iterates through all 64 coordinate pairs and calculates the difference squared.
        for(int coordinatePosition = 0; coordinatePosition < INPUT_SIZE; coordinatePosition++){
            accumulator += Math.pow((trainSample[coordinatePosition] - centroidCoordinate[coordinatePosition]),2);
        }
        //Returns the euclidean distance satisfying the equation sqrt(Δx^2 + Δy^2).
        return Math.sqrt(accumulator);
    }


}
