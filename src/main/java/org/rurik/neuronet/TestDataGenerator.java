package org.rurik.neuronet;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class TestDataGenerator {
    public static void main(String[] args) throws IOException {
        FileWriter fileWriter = new FileWriter("C:\\Users\\User\\IdeaProjects\\Examples\\SinNeuroNet\\src\\main\\resources\\datasets\\sin_generated_test_data.csv");
        PrintWriter printWriter = new PrintWriter(fileWriter);


        double step = 20d / 1000;
        for (double arg = -10; arg < 10; arg += step) {
            double delta = Math.random() * step;
            double v = arg + delta;
            printWriter.print(v + " , " + Math.sin(v));
            printWriter.println();
        }
        printWriter.close();
    }


}
