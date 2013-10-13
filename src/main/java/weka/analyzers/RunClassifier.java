package weka.analyzers;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.gui.Logger;

import java.util.Random;
import java.util.concurrent.Callable;

/**
 * General utility functions and classes for using a classifier to generate 
 * prediction(s) for a dataset.
 */
public final class RunClassifier {

    // Not intended to be instantiated
    private RunClassifier() {}

    /**
     * Runs a classifier using cross validation over numFolds folds
     * and returns its predictions per instance. The data's order
     * is unchanged.
     *
     * @param data that data to train and test the classifier with, should be
     *             randomized
     * @param classifier the classifier to use
     * @param numFolds the number of folds to use
     *
     * @return a double array where arr[i] is the classifiers predictions
     * for data.instance(i).
     * @throws Exception if the classifier could not be built
     */
    public static double[] runClassifier(Instances data, Classifier classifier,
            int numFolds) throws Exception {
        return runClassifier(data, classifier, numFolds, null);
    }

    /**
     * Runs a classifier using cross validation over numFolds folds
     * and returns its predictions per instance. The data's order
     * is unchanged. Progress will be logged of log is non-null.
     *
     * @param data that data to train and test the classifier with, should be
     *             randomized
     * @param classifier the classifier to use
     * @param numFolds the number of folds to use
     * @param log if non null will send status updates to log per fold
     *
     * @return a double array where arr[i] is the classifiers predictions
     * for data.instance(i).
     * @throws Exception if the classifier could not be built
     */
    public static double[] runClassifier(Instances data, Classifier classifier,
            int numFolds, Logger log) throws Exception {
        double[] testResults =  new double[data.numInstances()];
        int offset = 0;
        for(int fold = 0; fold < numFolds; fold++) {
            Instances train = data.trainCV(numFolds, fold);
            Instances test = data.testCV(numFolds, fold);
            if(log != null) {
                log.statusMessage(String.format("Building and Testing classifier for " +
                        "fold %d out of %d", fold + 1, numFolds));
            }
            classifier.buildClassifier(train);
            for(int j = 0; j < test.numInstances(); j++) {
                testResults[offset + j] = classifier.classifyInstance(test.instance(j));
            }
            offset += test.numInstances();
        }
        return testResults;
    }

    /**
     * Runs a classifier using cross validation over numFolds folds and returns
     * its predictions over multiple iterations. If the class is categorical this will
     * be count of the number of time each class was predicted, otherwise it will
     * be a list of all the predictions made. The data's order is unchanged. Progress
     * will be logged if log is non-null.
     *
     * @param data that data to train and test the classifier with
     * @param classifier the classifier to use
     * @param numFolds the number of folds to use
     * @param iterations the number of iterations to use
     * @param log if non null will send status updates to log per fold
     *
     * @return a double array where arr[i][j] is the classifiers predictions
     * for the jth prediction of instance i if the data was numeric and is
     * the number of prediction of class j for instance i if the data was
     * nominal.
     * @throws Exception if the classifier could not be built
     */
    public static double[][] runClassifier(Instances data, Classifier classifier,
            int numFolds, int iterations, Logger log) throws Exception {
        int[] index = new int[data.numInstances()];
        Random rand = new Random();
        for(int i = 0; i < index.length; i++) {
            index[i] = i;
        }
        if(iterations > 0) {
            data = new Instances(data);
        }
        double[][] testResults;
        if(data.classAttribute().isNumeric()) {
            testResults =  new double[data.numInstances()][iterations];
        } else {
            testResults =  new double[data.numInstances()][data.classAttribute().numValues()];
        }
        for(int iteration = 0; iteration < iterations; iteration++) {
            int offset = 0;
            AnalyzerUtils.shuffleInSync(data, index, rand);
            for(int fold = 0; fold < numFolds; fold++) {
                Instances train = data.trainCV(numFolds, fold);
                Instances test = data.testCV(numFolds, fold);
                if(log != null){
                    log.statusMessage(String.format("Building and Testing classifier for " +
                            "fold %d out of %d, iteration %d of %d", fold + 1, numFolds,
                            iteration + 1, iterations));
                }
                classifier.buildClassifier(train);
                for(int j = 0; j < test.numInstances(); j++) {
                    double val = classifier.classifyInstance(test.instance(j));
                    if(data.classAttribute().isNumeric()) {
                        testResults[index[offset + j]][iterations] = val;
                    } else {
                        testResults[index[offset + j]][(int) val]++;
                    }
                }
                offset += test.numInstances();
            }
        }
        return testResults;
    }

    /**
     * Callable implementation of runClassifier(..)
     */
    public static class CallableClassifier implements Callable<double[]> {
        private final Classifier clf;
        private final Instances data;
        private final int folds;

        public CallableClassifier(Instances data, int folds, Classifier clf) {
            this.data = data;
            this.folds = folds;
            this.clf = clf;
        }

        @Override
        public double[] call() throws Exception {
            return runClassifier(data, clf, folds);
        }
    }

    /**
     * Runnable implementation of runClassifier(...). Stores output or exception
     * as fields which can be examined later.
     */
    public static class RunnableClassifier implements Runnable {
        private final int folds;
        private final Instances data;
        public Classifier clf;
        public double[] output;
        public Exception ex;

        public RunnableClassifier(Instances data, int folds, Classifier clf) {
            this.data = data;
            this.folds = folds;
            this.clf = clf;
            output = null;
            ex = null;
        }

        @Override
        public void run() {
            try {
                output = runClassifier(data, clf, folds);
                ex = null;
            } catch (Exception e) {
                output = null;
                ex = e;
            }
        }
    }
}
