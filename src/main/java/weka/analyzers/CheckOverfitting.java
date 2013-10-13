package weka.analyzers;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.gui.Logger;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 * Analyzer to check if a classifier is overfitting. Builds a classifier and
 * evaluates it on the test and train data and reports the results to the user.
 */
// TODO we should be able to do something for numeric classes as well
public class CheckOverfitting extends ClassifierAnalyzer {

    public String globalInfo() {
        return "Builds a classifier and compares its accuracy on the test" +
                " data and train data, if the classifier is getting higher " +
                "score on the training set that indicates it is overfitting.";
    }

    /** For serialization */
    private static final long serialVersionUID = 395430566636168390L;

    /** Number of folds to split the data into */
    private int numFolds = 5;

    /** Number of times to build and evaluate the classifier */
    private int iterations = 1;

    /** Random seed to use for randomization */
    private long seed = 0;

    /**
     * Builds the currently selected classifier and evaluates it on both the
     * test and train dataset over a number of iterations, returns the number
     * correct and total Instances on the test and train datasets.
     *
     * @param data Instances to evaluate, might be reordered
     * @param log Logger to log status updates to
     * @return int[] of the number of correct Instances and total Instances
     * on the test then train set.
     * @throws Exception If there was a problem building the classifier
     */
    public int[] evaluate(Instances data, Logger log) throws Exception {
        int trainCorrect = 0;
        int testCorrect = 0;
        int trainInstances = 0;
        int testInstances = 0;
        Classifier clf = getClassifier();
        Random rand = new Random(seed);
        int fold = 0;
        for(int i = 0; i < iterations; i++) {
            if(fold >= numFolds) {
                data.randomize(rand);
                fold = 0;
            }
            log.statusMessage(String.format("Building and Testing classifier for " +
                    "fold %d out of %d, run %d of %d", fold + 1, numFolds,
                    i + 1, iterations));
            Instances test = data.testCV(numFolds, fold);
            Instances train = data.trainCV(numFolds, fold);
            clf.buildClassifier(train);
            trainCorrect += correctPredictions(train, clf);
            testCorrect += correctPredictions(test, clf);
            trainInstances += train.numInstances();
            testInstances += test.numInstances();
            fold++;
        }
        log.statusMessage("Run complete");
        return new int[] {
                testCorrect,
                testInstances,
                trainCorrect,
                trainInstances
                };
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disable(Capabilities.Capability.NUMERIC_CLASS);
        return capabilities;
    }

    @Override
    public AnalyzerOutput analyzeData(Instances data, int idAtt, Logger logger)
            throws Exception {
        if(idAtt != -1) {
            data = AnalyzerUtils.removeColumn(data, idAtt);
        } else {
            data = new Instances(data); // Preserve order
        }
        int[] results = evaluate(data, logger);
        double testAcc = ((double)results[0] / results[1]);
        double trainAcc = ((double)results[2] / results[3]);
        double difference = trainAcc - testAcc;
        StringBuilder report = new StringBuilder("\n\nOverfitting Report\n");
        report.append("Iterations: " + iterations + "\n");
        report.append("Number of folds: " + numFolds + "\n");
        report.append(String.format("When testing on the held out test data got " +
                " %.4f (%d correct out of %d instances)%n", testAcc,
                results[0], results[1]));
        report.append(String.format("When testing on the data used to build the classifier " +
                "got %.4f (%d correct out of %d instances)%n", trainAcc,
                results[2], results[3]));
        report.append(String.format("%nYour classifier was %.4f percent more accurate on the train data%n",
                difference));
        if(difference >= 0.01) {
            if(difference > 0.05) {
                report.append("This difference is large enough to indicate that your classifier " +
                        " is overfitting.\n");
            } else {
                report.append("This difference is large enough to indicate that your classifier " +
                        " might be overfitting.\n");
            }
            report.append("Your classifier might become more accurate if you increase\n" +
                    "regularization parameter, add more training data, or use feature selection\n" +
                    "to reduce the number of features.\n");
        } else if(difference < 0.01) {
            report.append("This difference is small enough to indicate that your classifier is " +
                    "probably not overfitting");
        }
        return new AnalyzerOutput(report.toString());
    }



    /**
     * Counts the number of correct predictions a Classifier makes on
     * some Instances.
     *
     * @param data Instances to classify
     * @param clf Classifier to use
     * @return number of correct classifications
     * @throws Exception if there was a problem classifying the data
     */
    private int correctPredictions(Instances data, Classifier clf) throws Exception {
        int correct = 0;
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).classValue() == clf.classifyInstance(data.instance(i))) {
                correct++;
            }
        }
        return correct;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String iterString = Utils.getOption('I', options);
        iterations = iterString.length() != 0 ? Integer.parseInt(iterString) : 1;

        String foldString = Utils.getOption('F', options);
        numFolds = foldString.length() != 0 ? Integer.parseInt(foldString) : 5;

        String seedString = Utils.getOption('S', options);
        seed= foldString.length() != 0 ? Long.parseLong(seedString) : 0;


        super.setOptions(options);
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option(
                "\tNumber of times to build/test classifier (default: 5)\n",
                "I", 1, "-I"));
        newVector.addElement(new Option(
                "\tNumber of folds to divide that dataset into\n",
                "F", 1, "-F"));
        newVector.addElement(new Option(
                "\tSeed used to generate test and train subsets\n",
                "S", 1, "-S"));
        return newVector.elements();
    }

    @Override
    public String[] getOptions() {
        String [] superOptions = super.getOptions();
        String [] options = new String [superOptions.length + 6];

        int current = 0;
        options[current++] = "-I";
        options[current++] = Integer.toString(iterations);
        options[current++] = "-F";
        options[current++] = Integer.toString(numFolds);
        options[current++] = "-S";
        options[current++] = Long.toString(seed);

        System.arraycopy(superOptions, 0, options, current,
                 superOptions.length);
        current += superOptions.length;
        while(current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    // Getter, Setters, and ToolTips

    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    public String seedTipText() {
        return "Seed to use for randomization.";
    }

    public String iterationTipText() {
        return "Number of times to build the classifier and evaluate on " +
                " the test and train data. Each iteration will use a disjoint" +
                " test fold, if there are more iterations then fold the data " +
                "will be shuffled and after each fold is used and the " +
                "process repeated.";
    }

    public String numFoldsTipText() {
        return "Number of folds to use when splitting the data into " +
                "test and train sets.";
    }
}
