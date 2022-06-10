package DigitRecognition;

import java.io.*;

//Class reads in both dataset files and returns them as a DataSets object.
public class DataLoader {
    //initialises a Datasets object to store the train and test set.
    private DataSets sets;

    //Constructor
    public DataLoader(String setOnePath , String setTwoPath , int setSize){
        sets = new DataSets(loadSet(setOnePath,setSize) , loadSet(setTwoPath,setSize));
    }

    //Method returns the train and test set in one object.
    public DataSets returnSets(){
        return sets;
    }

    //Method used to return a dataset from a string file.
    private int[][] loadSet(String setPath , int setSize){
        //32kb constant buffer.
        int BUFFER_SIZE = 32768;
        //Size of the input array, 64 pixel values and 1 classification value.
        int INPUT_VECTOR_SIZE = 65;
        //Stores the entire data set in a 2d array.
        int[][] setMatrix = new int[setSize][INPUT_VECTOR_SIZE];
        //Initialises buffered reader outside the try catch, so it has scope for its call in the finally statement.
        BufferedReader reader = null;

        //Buffered reader may throw exceptions hence a try catch is used.
        try{
            reader = new BufferedReader(new FileReader(setPath), BUFFER_SIZE);
            //Iterates through all the lines of the document.
            //Set size is consistent across the 2 sets hence a constant is used instead of using has next line.
            for(int currentLine = 0; currentLine < setSize; currentLine++){
                //converts the line into a string array.
                String[] line = reader.readLine().split(",");
                //converts the string array into integer array.
                int[] inputVecArr = new int[INPUT_VECTOR_SIZE];
                for(int position = 0; position < INPUT_VECTOR_SIZE; position++){
                    inputVecArr[position] = Integer.parseInt(line[position]);
                }
                setMatrix[currentLine] = inputVecArr;
            }
        }
        //Error catches.
        catch (FileNotFoundException e) {
            System.out.println("File not found Exception");
        } catch (IOException e) {
            System.out.println("IO Exception");
        }
        finally {
            try {
                reader.close();
            } catch (IOException e) {
                System.out.println("Buffered Reader Failed To Close");
            }
        }
        return setMatrix;
    }
}

