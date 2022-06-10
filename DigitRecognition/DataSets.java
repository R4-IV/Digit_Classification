package DigitRecognition;

//Stores an object to be returned by the DataLoader containing both sets.
public class DataSets {
    private int[][] setOne;
    private int[][] setTwo;

    //Constructor
    public DataSets(int[][] setOne, int[][] setTwo) {
        this.setOne = setOne;
        this.setTwo = setTwo;
    }

    //Returns set one.
    public int[][] getSetOne() {
        return setOne;
    }

    //Returns set two.
    public int[][] getSetTwo() {
        return setTwo;
    }
}
