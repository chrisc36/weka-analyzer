package weka.analyzers;

import weka.analyzers.AnalyzerUtils.GenerateVisualizerWindow;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.gui.Logger;
import weka.gui.visualize.Plot2D;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 * This Analyzer knows how to build a learning curve for some dataset and
 * classifier and present the results in a human readable way.
 */
// TODO Improve the fitting on the shown curve, (maybe segmented?)
public class GenerateLearningCurve extends ClassifierAnalyzer {

    /** For serialization */
    private static final long serialVersionUID = -5728120912169775087L;

    /** Number of points to use when building the curve into a graph */
    protected final int NUM_GRAPH_POINTS = 60;

    /** Percent far ahead of the data points to draw the projected the curve */
    protected final double GRAPH_PROJECTION = 1.2;

    /** Number of instances to increase the training set by each step */
    protected int step = 10;

    /** Number of instances to start in the training set */
    protected int start = 10;

    /** Max of number of instances to use in the training set */
    protected int stop = -1;

    /** Percent of instances to use as the test set */
    protected double holdoutSize = 0.30;

    /** Number of times to build and average the curve */
    protected int trials = 1;

    public String globalInfo() {
        return "Builds a given classifier on increasingly large subsets of the " +
                "data and evaluates its accuracy on a hold out set. " +
                "Graphs the number of training examples used to build the " +
                "model vs. model accuracy. It can be used to " +
                "judge how much additional training examples would help " +
                "increase the accuracy of the model.";
    }

    /**
     * Builds a classifier on increasingly large subsets of the training data and
     * reports the average results over a number of trials using the current
     * configuration.
     *
     * @param data Instances to build the classifier for
     * @param logger Logger to log status updates to
     * @return double[] with the average accuracy of each run, the ith result
     * is the average accuracy of using start + step * i number of instances
     *
     * @throws Exception If there was an error using the classifier
     */
    public double[] testClassifier(Instances data, Logger logger) throws Exception {
        Random r = new Random();
        data.randomize(new Random());
        int testSize = (int) (data.numInstances() * this.holdoutSize);
        int trainSize = data.numInstances() - testSize;
        if(stop > 0) {
            trainSize = Math.min(trainSize, stop);
        }

        // Number of times we will need to build the classifier per trial
        int iterations = (int) (1 + Math.floor((trainSize - start)/((double)step)));

        // holds the accuracy we got for each iteration
        final double[] testResults =  new double[iterations];

        Instances test = new Instances(data, testSize);
        Instances train =  new Instances(data, trainSize);

        // Index to start building the test set for the current trial
        int testStart = 0;

        for(int trial = 1; trial <= trials; trial++) {
            /*
               For each trial, we need to create a new testSet. To help enforce
               disjointed test sets we prefer to use instances that have not been
               in a test set before. Thus we work our way through the instances
               in sequential order, shuffling the dataset and restarting when we
               have too few new instances left to build a completely new test set.
             */

            // TODO we should only shuffle when ALL instances have been used, although
            // we would have to avoid reusing instances somehow.
            // TODO better to use a new test set each time?

            // Shuffle the data if we have too few unused instances left
            if(testStart > data.numInstances()) {
                data.randomize(r);
                testStart = 0;
            }
            test.delete();
            train.delete();
            for(int i = 0; i < data.numInstances(); i++) {
                if(i >= testStart && i < testStart + testSize ||
                        i < -data.numInstances() + testStart + testSize) {
                    test.add(data.instance(i));
                } else {
                    train.add(data.instance(i));
                }
            }
            testStart += testSize;

            // Now we have built the test set we start testing using larger, randomized portions
            // of the training set
            for(int i = 0; i < iterations; i++) {
                int numBuildInstances = start + i * step;
                if(logger != null) {
                    logger.statusMessage("Starting trial: " + trial + " with: " +
                            numBuildInstances+ " instances");
                }
                train.randomize(r);
                Instances buildInstances = new Instances(train, 0, numBuildInstances);
                classifier.buildClassifier(buildInstances);
                double score = 0;
                for(int j = 0; j < test.numInstances(); j++) {
                    Instance cur = test.instance(j);
                    double pred = classifier.classifyInstance(cur);
                    if(data.classAttribute().isNominal()) {
                        if(classifier.classifyInstance(cur) == cur.classValue()) {
                            score++;
                        }
                    } else {
                        score += Math.abs(pred - cur.classValue());
                    }
                }
                testResults[i] += score/(test.numInstances()*trials);
            }
        }
        return testResults;
    }

    @Override
    public AnalyzerOutput analyzeData(Instances data, int idIndex,
                                      Logger logger) throws Exception {
        if(idIndex != -1) {
            data = AnalyzerUtils.removeColumn(data, idIndex);
        }
        final double[] accuracies = testClassifier(data, logger);

        FastVector attInfo = new FastVector();
        attInfo.addElement(new Attribute("log(Number of training examples)"));
        attInfo.addElement(new Attribute("Cross validation score"));

        Instances logTrainDataPoints = new Instances("Log Test", attInfo, accuracies.length);
        for(int d = 0; d < accuracies.length; d++) {
            double[] val = {Math.log(start + d*step),accuracies[d]};
            logTrainDataPoints.add(new Instance(1.0, val));
        }
        logTrainDataPoints.setClassIndex(1);
        SimpleLinearRegression s = new SimpleLinearRegression();
        s.buildClassifier(logTrainDataPoints);
        final double slope = s.getSlope();
        final double intercept = s.getIntercept();

        String scoreStr = data.classAttribute().isNominal() ? "accuracy" : "RMSE";
        StringBuilder report = new StringBuilder();
        report.append("Generating Learning Curve\n");
        report.append(String.format("Best %s: %.3f%n", scoreStr, accuracies[accuracies.length-1]));
        report.append("\nFit logirithm curve to the test " + scoreStr + ":\n");
        report.append(String.format("<%s> = %.3f * ln(<Number of train examples>) + %.3f)", scoreStr, slope, intercept));
        report.append(String.format("%nProjected %s using all the data: %.4f",scoreStr, (intercept + slope*Math.log(data.numInstances()))));
        report.append(String.format("%nProjected derivate at that point: %f", ((double)slope)/data.numInstances()));
        report.append(String.format("%nProjected %s with 10%% more instances: %.4f%n", scoreStr, intercept + slope*Math.log(1.1*data.numInstances())));

        GenerateVisualization[] v = new GenerateVisualization[2];
        v[0] = new GenerateVisualizerWindow("Learning Curve Graph",450,400) {
            protected JComponent generateJComponent() {
                return imposeLogirithmicGraph(slope, intercept,
                        accuracies);
            }
        };
        v[1] = dataPointVisualizer("Test Accuracy Data Points", accuracies, start, step);
        return new AnalyzerOutput(report.toString(),v);
    }

    /**
     * Graphs some data points and a logarithmic curve.
     *
     * @param slope Double of the slope of the logarithmic curve
     * @param intercept intercept of the logarithmic curve
     * @param dataPoints points to graph
     * @return VisualizePanel containing the curve and datapoin
     */
    /*
     * Weka does not really support imposing curves, a hack around this is to
     * sample some datapoints from the curve we want, connect them with a
     * straight lines, and finally plot them alongside the original points.
     */
    public VisualizePanel imposeLogirithmicGraph(double slope, double intercept,
                double[] dataPoints) {
        FastVector attInfo = new FastVector();
        attInfo.addElement(new Attribute("Number of training examples"));
        attInfo.addElement(new Attribute("Accuracy"));
        FastVector testDataAttValues = new FastVector();
        testDataAttValues.addElement("Data Point");
        testDataAttValues.addElement("Curve");
        attInfo.addElement(new Attribute("Type", testDataAttValues));
        int totalPoints = NUM_GRAPH_POINTS + dataPoints.length;
        Instances points = new Instances("Learning Curve",attInfo, totalPoints);

        // Points for the curve
        int stop = start + ((int)(step*(dataPoints.length-1)*GRAPH_PROJECTION));
        for(int i = 1; i <= NUM_GRAPH_POINTS; i++) {
            double x = (stop-start)/((double)NUM_GRAPH_POINTS) * i;
            double y = Math.log(x)*slope + intercept;
            double[] p = {x,y,1};
            points.add(new Instance(1.0, p));
        }

        // The data points we actually have
        for(int d = 0; d < dataPoints.length; d++) {
            double[] testVal = {start + d*step, dataPoints[d], 0};
            points.add(new Instance(1.0, testVal));
        }
        points.setClassIndex(2);
        PlotData2D plot = new PlotData2D(points);
        boolean[] connected = new boolean[totalPoints];
        int[] shapes = new int[totalPoints];
        Arrays.fill(shapes, 0, NUM_GRAPH_POINTS, 1);
        Arrays.fill(shapes, NUM_GRAPH_POINTS, totalPoints, Plot2D.DEFAULT_SHAPE_SIZE);
        Arrays.fill(connected, 0, NUM_GRAPH_POINTS, true);
        VisualizePanel p = new VisualizePanel();
        try {
            plot.setConnectPoints(connected);
            plot.setShapeSize(shapes);
            p.addPlot(plot);
        } catch (Exception e) {
            // This should never happen!
            throw new RuntimeException(e);
        }
        return p;
    }

    /**
     * Builds a GenerateVisualizerWindow that displays a number of data points
     * in csv form.
     *
     * @param name String name of the Visualization
     * @param data y-values to graph
     * @param start value the x-values start at
     * @param step value the x-values increase by for each y value
     * @return Visualizer that displays datapoints in csv form
     */
    private GenerateVisualizerWindow dataPointVisualizer(final String name,
                                                         final double[] data,
                                                         final int start,
                                                         final int step) {
        return new GenerateVisualizerWindow(name, 400, 450) {
            protected JComponent generateJComponent() {
                StringBuilder sb = new StringBuilder();
                for(int i = 0; i < data.length; i++) {
                    sb.append(String.format("%d,%.3f%n", start+step*i, data[i]));
                }
                return AnalyzerUtils.generateTextField(sb.toString());
            }
        };
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option(
                "\tNumber of instances to start with (default: 0) d\n",
                "S", 1, "-S"));
        newVector.addElement(new Option(
                "\tStep size (default: 10) \n",
                "I", 1, "-I"));
        newVector.addElement(new Option(
                "\tNumber of instances to stop with (-1 for maximum possible) \n",
                "M", 1, "-M"));
        newVector.addElement(new Option(
                "\tPercent of instances to put in hold out set (default: .30) \n",
                "H", 1, "-H"));
        newVector.addElement(new Option(
                "\tNumber trials to average across (default: 1) \n",
                "H", 1, "-T"));

        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String startString = Utils.getOption('S', options);
        start = startString.length() != 0 ? Integer.parseInt(startString) : 10;

        String stepString = Utils.getOption('I', options);
        step = stepString.length() != 0 ? Integer.parseInt(stepString) : 10;

        String stopString = Utils.getOption('M', options);
        stop = stopString.length() != 0 ? Integer.parseInt(stopString) : -1;

        String holdoutString = Utils.getOption('H', options);
        holdoutSize = holdoutString.length() != 0 ? Double.parseDouble(holdoutString) : -1;

        String trailsString = Utils.getOption('T', options);
        trials = trailsString.length() != 0 ? Integer.parseInt(trailsString) : 1;

        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        String [] superOptions = super.getOptions();
        String [] options = new String [superOptions.length + 10];

        int current = 0;
        options[current++] = "-S";
        options[current++] = Integer.toString(start);
        options[current++] = "-I";
        options[current++] = Integer.toString(step);
        options[current++] = "-M";
        options[current++] = Integer.toString(stop);
        options[current++] = "-H";
        options[current++] = Double.toString(holdoutSize);
        options[current++] = "-T";
        options[current++] = Integer.toString(trials);

        System.arraycopy(superOptions, 0, options, current,
                 superOptions.length);
        current += superOptions.length;
        while(current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    // Getters, Setters and Tooltips for the configurations

    public int getStart() {
        return start;
    }

    public void setStart(int start) {
        this.start = start;
    }

    public int getStop() {
        return stop;
    }

    public void setStop(int stop) {
        this.stop = stop;
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public double getHoldoutSize() {
        return holdoutSize;
    }

    public void setHoldoutSize(double holdoutSize) {
        this.holdoutSize = holdoutSize;
    }

    public int getTrials() {
        return trials;
    }

    public void setTrials(int i) {
        trials = i;
    }

    public String holdoutSizeTipText() {
        return "Percent of instances to be put into a holdout set " +
                "to be used for testing.";
    }

    public String trialsText() {
        return "Number of times to build the classifier for each subset size. A different holdout " +
                "set is used each trial and the results will be averaged. Doing this " +
                "will reduce the variance of the results making for a " +
                "smoother and more accurate curve, but will be computationally " +
                "expensive.";
    }

    public String startTipText() {
        return "Number of instances to start building the model wih.";
    }

    public String stepTipText() {
        return "Number of additional instances to use when building " +
                "the model for each successive iteration.";
    }

    public String stopTipText() {
        return "Max number of instances to test with. -1 indicates no maximum.";
    }
}
