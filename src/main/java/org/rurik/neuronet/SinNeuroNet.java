package org.rurik.neuronet;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.io.IOException;

public class SinNeuroNet {


    public static void main(String[] args) throws IOException, InterruptedException {
        int seed = 13;
        double learningRate = 0.01;
        int batchSize = 1000;
        int nEpochs = 600000;

        int numInputs = 1;
        int numOutputs = 1;
        int numHiddenNodes = 10;


        final String filenameTrain = new ClassPathResource("/datasets/sin_generated_test_data.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 1, 1, true);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.05))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.TANH)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(1));


        for (int i = 0; i < nEpochs; i++) {
            trainIter.reset();
            network.fit(trainIter);
        }


        double[] x = new double[2001];
        double[] yExpected = new double[2001];
        double[] yNeuroNet = new double[2001];


        int i = 0;

        for (double v = -10; v < 10; v += 0.01) {

            final INDArray input = Nd4j.create(new double[]{v}, new int[]{1, 1});
            INDArray out = network.output(input, false);
            double realSin = Math.sin(v);
            System.out.println("v=" + String.format("%.3f", v) + " net out=" + out + "sin=" + String.format("%.3f", realSin));


            x[i] = v;
            yExpected[i] = realSin;
            yNeuroNet[i] = out.getDouble(1, 1);
            i++;
        }


        plot(x, yExpected, yNeuroNet);

    }


    //Plot the data
    private static void plot(final double[] x, final double[] yExpected, final double[] yNeuroNet) {
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, x, yExpected, "Sin");
        addSeries(dataSet, x, yNeuroNet, "NeuroNet Sin");


        final JFreeChart chart = ChartFactory.createXYLineChart(
                "NeuroNet Sin",      // chart title
                "X",                        // x axis label
                "sin(X)", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);

        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, final double[] x, final double[] y, final String label) {
        final XYSeries s = new XYSeries(label);
        for (int j = 0; j < x.length; j++) s.add(x[j], y[j]);
        dataSet.addSeries(s);
    }


}
