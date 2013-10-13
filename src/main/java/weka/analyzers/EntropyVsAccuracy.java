package weka.analyzers;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.gui.Logger;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableRowSorter;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 *  Analyzer that computes the accuracy of a number of classifiers on the
 *  data then plots each data point based on the average accuracy of the
 *  classifiers and entropy of the classifiers output.
 *  Can be used to identify noise or hard to classifier data points.
 *  Classifiers can be run in multiple threads.
 *
 *  See:
 *  Patel, K., Drucker, S. M., Fogarty, J., Kapoor, A., & Tan, D. S. (2011, July).
 *  Using multiple models to understand dat
 *
 */
/*
 * We want to be able to time out long running Classifiers, but since Classifiers are
 * not specified to handle interrupts we are forced to resort to thread.stop() to halt
 * threads, (thread.stop is used elsewhere in Weka core to halt Classifiers as well).
 * Sadly we can't use nice ExecutorService classes since we cannot stop threads with them.
 * Thus we have one thread to build the classifier and one to monitor and kill that thread
 * if it takes too long.
 */
// TODO handle regression and binary classification using std instead of entropy
// TODO the classifier list format could be much improved
// TODO preprocess the list to ensure it is formatted correctly
// TODO more efficient if each thread updates one data structure with results
/*
 * TODO we could do this with only extra thread and a blockingQueue to
 * indicate when to restart.
 */
public class EntropyVsAccuracy extends Analyzer {

    public String globalInfo() {
        return "Runs a medley of classifiers on the dataset and uses cross " +
                "validation to create a prediction per classifier per " +
                "instance. Then graphs the average inaccuracy of the " +
                "classifier vs. the entropy of the examples.";
    }

    /** Where the list of classifiers to use is */
    private String classifierFilePath = "";

    /** Number of folds to divide the data set into is */
    private int cvFolds = 4;

    /** Number of threads to build classifiers with */
    private int numThreads = 2;

    /** How long to wait for classifiers to finish */
    private long timeout = 60;

    /**
     * Class to hold statistics about a classifier. Sortable by
     * Classifier score. Immutable.
     */
    public class ClfStats implements Comparable<ClfStats>{

        /** String representation if the classifier */
        public final String clf;

        /** Number of this Classifier */
        public final int clfNum;

        /** Classifier's score */
        public final double score;

        public ClfStats(String clf, int clfNum, double score) {
            this.clf = clf;
            this.clfNum = clfNum;
            this.score = score;
        }

        @Override
        public int compareTo(ClfStats other) {
            return Double.compare(score, other.score);
        }
    }

    /**
     * Class to hold error messages produced when building a Classifier.
     * Sortable by classifier number. Immutable.
     */
    public class ClfErrorMsg implements Comparable<ClfErrorMsg>{

        /** The error message */
        public final String msg;

        /** Classifier that generated the error message */
        public final int clfNum;

        /** High level description of what created the error */
        public final String src;

        public ClfErrorMsg(String msg, String src, int clfNum) {
            this.msg = msg;
            this.clfNum =  clfNum;
            this.src = src;
        }

        @Override
        public int compareTo(ClfErrorMsg other) {
            return Integer.compare(clfNum, other.clfNum);
        }


        @Override
        public String toString() {
            return "(" + clfNum + ")" + src + "/n" + msg;
        }
    }

    /**
     * Builds a table of Classifier scores suitable for a user to view.
     *
     * @param stats Vector of ClassifierStats that mark the score for each
     *              classifier to use
     * @return JScrollPane containing the table
     */
    private static JScrollPane buildStatsTable (
            final Vector<ClfStats> stats) {

        // Define the table layout
        class StatsModel extends AbstractTableModel {
            private static final long serialVersionUID = -632538114790221585L;
            private String[] colnames = {"Classifier","Score"};

            @Override
            public int getColumnCount() {
                return 2;
            }

            @Override
            public int getRowCount() {
                return stats.size();
            }

            @Override
            public String getColumnName(int col) {
                return colnames[col];
            }

            @Override
            public Object getValueAt(int row, int col) {
                switch (col) {
                    case 0: return stats.get(row).clf;
                    case 1: return stats.get(row).score;
                    default: throw new RuntimeException();
                }
            }

            @SuppressWarnings({ "unchecked", "rawtypes" })
            public Class getColumnClass(int col) {
                switch (col) {
                    case 0: return String.class;
                    case 1: return Double.class;
                    default: throw new RuntimeException();
                }
            }
        }

        // Build and wrap the table
        StatsModel model = new StatsModel();
        TableRowSorter<StatsModel> sorter = new TableRowSorter<StatsModel>(model);
        final JTable jt = new JTable(model);
        jt.setRowSorter(sorter);
        jt.setCellSelectionEnabled(true);
        jt.setRowSelectionAllowed(false);
        jt.setCellSelectionEnabled(false);
        jt.setFillsViewportHeight(true);
        JScrollPane js = new JScrollPane(jt);
        js.setName("Classifier Stats");
        return js;
    }

    /**
     *  Class that reads lines from a line reader, builds the specified
     *  Classifier, uses it to acquire classifications for the data killing
     *  the Classifier if it takes too long. Thread safe, we assume several of
     *  this are running at once.
     */
    private class RunClassifiers implements Callable<int[][]> {

        /** Data to build the classifier from */
        private final Instances data;

        /** Source to read Classifier String representation and number */
        private final LineNumberReader lnr;

        /** Vector to append any error message to */
        private final Vector<ClfErrorMsg> errors;

        /** Vector to append classifier statistics to */
        private final Vector<ClfStats> stats;

        /** Logger to log output to */
        private final Logger logger;

        public RunClassifiers(Instances data, LineNumberReader lnr,
                              Vector <ClfErrorMsg> errors, Vector<ClfStats> stats, Logger logger) {
            this.data = new Instances(data); // Ensure ita a private copy
            this.lnr = lnr;
            this.errors = errors;
            this.logger = logger;
            this.stats = stats;
        }

        /** Print a message to the log, if a Logger is in use */
        private void logMsg(String msg) {
            if(logger != null) {
                // Logger might not be thread safe
                synchronized(logger) {
                    logger.statusMessage(msg);
                }
            }
        }

        /** Report an error to both the log and the list of errors */
        private void reportError(String src, String error, int num) {
            ClfErrorMsg errorMsg = new ClfErrorMsg(src, error, num);
            logMsg("ERROR:" + errorMsg.toString());
            errors.add(errorMsg);
        }

        @SuppressWarnings("deprecation") // Need use of Thread.stop()
        @Override
        public int[][] call() {
            int[] indices = new int[data.numInstances()];
            for(int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            int[][] output = new int[data.numInstances()][data.numClasses()];
            RunClassifier.RunnableClassifier rc;
            while(true) {
                if(Thread.currentThread().isInterrupted()) {
                    return output;
                }
                // Grab the next Classifier
                String line;
                int lineNumber;
                try {
                    synchronized(lnr) {
                        line = lnr.readLine();
                        if(line == null) {
                            return output;
                        }
                        lineNumber = lnr.getLineNumber();
                    }
                } catch (IOException e) {
                    reportError("Thread interrupted by IOException!/n" + e.getMessage(),
                            "Load Error", -1);
                    return output;
                }
                try {
                    rc = new RunClassifier.RunnableClassifier(data, cvFolds, loadClassifier(line));
                } catch (Exception e) {
                    reportError("Error loading classifier", line, lineNumber);
                    continue;
                }

                // Run it and collect the result
                AnalyzerUtils.shuffleInSync(data, indices);
                logMsg("Starting classifier " + lineNumber + " : " + line);
                Thread t = new Thread(rc);
                try {
                    t.start();
                    t.join(timeout * 1000);
                    if(t.isAlive()) {
                        reportError("Classifier timed out", line, lineNumber);
                        t.stop();
                    } else if(rc.ex != null) {
                        reportError("Problem building classifier:/n" +
                                rc.ex.getMessage(), line, lineNumber);
                    } else {
                        double correct = 0;
                        for(int i = 0; i < rc.output.length; i++) {
                            double pred = rc.output[i];
                            output[indices[i]][(int)pred]++;
                            if(pred == data.instance(i).classValue()) {
                                correct++;
                            }
                        }
                        stats.add(new ClfStats(line, lineNumber, correct / data.numInstances()));
                    }
                } catch(InterruptedException e) {
                    t.stop();
                    return output;
                }
            }
        }
    }

    /**
     * Loads a Classifier from its string representation.
     *
     * @param clfString String representation of the classifier to load
     * @return Initialized Classifier
     * @throws Exception there was a problem loading the classifier
     */
    public Classifier loadClassifier(String clfString) throws Exception {
        String[] input = clfString.split(" ");
        String classifierString = input[0];

        // Convert to absolute names if we need to
        if(!classifierString.startsWith("weka.classifiers.")) {
            classifierString = "weka.classifiers." + classifierString;
        } else if(!classifierString.startsWith("weka.")) {
            classifierString = "weka." + classifierString;
        }
        String[] options = new String[input.length - 1];
        System.arraycopy(input, 1, options, 0, options.length);
        Classifier clf = (Classifier) Class.forName(classifierString).newInstance();
        clf.setOptions(options);
        return clf;
    }

    /**
     * Reads Classifier's from the currently set file, runs them on an
     * Instances and uses cross validation to acquire a prediction per
     * classifier per Instance, and collects the predictions. Additionally
     * appends any error messages and per classifier stats to the given vectors.
     *
     * @param data Instances to run the classifiers on
     * @param logger Logger to write status updates to
     * @param errors Vector to append error messages to
     * @param stats Vector to per classifier stats to
     * @return number of classifier that predicted each class for each Instance
     * @throws Exception if there was a problem building or running the Classifiers
     */
    public int[][] run(Instances data, Logger logger,
                       Vector<ClfErrorMsg> errors, Vector<ClfStats> stats) throws Exception {
        data = new Instances(data);
        LineNumberReader lnr = new LineNumberReader(new FileReader(classifierFilePath));

        int[][] allPredictions = new int[data.numInstances()][data.numClasses()];

        // Start the worker threads
        ExecutorService threadExecutor = Executors.newFixedThreadPool(numThreads);
        List<Future<int[][]>> futureList = new ArrayList<Future<int[][]>>();
        for(int i = 0; i < numThreads; i++) {
            futureList.add(threadExecutor.submit(new RunClassifiers(data, lnr, errors, stats, logger)));
        }

        // Collect the output
        for(Future<int[][]> partialOutputs : futureList) {
            int[][] partialOutput = partialOutputs.get();
            for(int i = 0; i < data.numInstances(); i++) {
                for(int j = 0; j < data.numClasses(); j++) {
                    allPredictions[i][j] += partialOutput[i][j];
                }
            }
        }
        lnr.close();
        threadExecutor.shutdown();
        return allPredictions;
    }

    @Override
    public AnalyzerOutput analyzeData(Instances data, int idIndex, Logger logger) throws Exception {
        final Vector<ClfErrorMsg> errors = new Vector<ClfErrorMsg>();
        final Vector<ClfStats> stats = new Vector<ClfStats>();
        int[][] allPredictions = run(idIndex == -1 ? data : AnalyzerUtils.removeColumn(data, idIndex),
                logger, errors, stats);
        int totalClassifiers = Utils.sum(allPredictions[0]);
        StringBuilder report = new StringBuilder("Finished! Used " + totalClassifiers + " classifiers/n");
        int correct = 0;
        for(int i = 0; i < data.numInstances(); i++) {
            correct += allPredictions[i][(int) data.instance(i).classValue()];
        }
        report.append(String.format("Average accuracy was %.3f%n",
                ((double) correct) / (totalClassifiers*data.numInstances())));
        Collections.sort(stats, Collections.reverseOrder());
        ClfStats best = stats.get(0);
        report.append(String.format("Best classifier was (%d) %s with %.4f accuracy%n",
                best.clfNum, best.clf, best.score));
        report.append("Classifier ranking saved to classifiers");
        int resultOffset = 0;
        if(errors.size() > 0) {
            resultOffset++;
        }
        GenerateVisualization[] gvs = new GenerateVisualization[resultOffset + 2];
        if(errors.size() > 0) {
            report.append("There were " + errors.size() + " errors, see Errors for details/n");
            StringBuilder errorReport = new StringBuilder("Errors:/n");
            Collections.sort(errors);
            for(ClfErrorMsg msg : errors) {
                errorReport.append(msg.toString() + "/n");
            }
            gvs[0] = new AnalyzerUtils.GenerateTextWindow("Error Report", errorReport.toString(), 500, 600);
        } else {
            report.append("No errors encountered when building or using classifiers/n");
        }
        gvs[resultOffset] = new AnalyzerUtils.GenerateVisualizerWindow("Classifier Scores", 500, 600) {
            protected JComponent generateJComponent() {
                return buildStatsTable(stats);
            }
        };
        final Instances modifiedData = addEntropyAndAccuracy(allPredictions, data);
        modifiedData.setClass(data.classAttribute());
        gvs[resultOffset + 1] = new AnalyzerUtils.GenerateVisualizerWindow("Classifier Scores", 500, 600) {
            protected JComponent generateJComponent() {
                PlotData2D plot = new PlotData2D(modifiedData);
                VisualizePanel p = new VisualizePanel();
                try {
                    p.addPlot(plot);
                    p.setXIndex(modifiedData.numAttributes()- 2);
                    p.setYIndex(modifiedData.numAttributes()- 1);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                return p;
            }
        };
        return new AnalyzerOutput(report.toString(), gvs);
    }

    /**
     * Gets the entropy and percent inaccuracy of per class per
     * Instance predictions.
     *
     * @param predictions Array of per class predictions per Instance
     * @param data Original data ordered in alignment with predictions
     * @return Inaccuray and entropy per Instance
     */
    public double[][] calculateEntropyAndAccuracy(int[][] predictions, Instances data) {
        double[][] values = new double[predictions.length][data.numClasses()];
        double totalGuesses = Utils.sum(predictions[0]);
        for(int i = 0; i < values.length; i++) {
            int trueClass = (int) data.instance(i).classValue();
            values[i][0] = 1- predictions[i][trueClass] / totalGuesses;
            values[i][1] = AnalyzerUtils.entropy(predictions[i]);;
        }
        return values;
    }

    /**
     * Adds an entropy and incorrectness column to Instances. Input similar to
     * getEntropyAndInaccuracy.
     *
     * @param predictions Array of per class predictions per Instance
     * @param data Original data ordered in alignment with predictions
     * @return Copy of Instances with Incorrectness and Entropy columns
     */
    public Instances addEntropyAndAccuracy(int[][] predictions, Instances data) {
        data.insertAttributeAt(new Attribute("Incorrectness"), data.numAttributes());
        data.insertAttributeAt(new Attribute("Entropy"), data.numAttributes());
        double totalGuesses = Utils.sum(predictions[0]);
        for(int i = 0; i < data.numInstances(); i++) {
            int trueClass = (int) data.instance(i).classValue();
            double incorrectness = 1- predictions[i][trueClass] / totalGuesses;
            double entropy = AnalyzerUtils.entropy(predictions[i]);
            data.instance(i).setValue(data.numAttributes()- 2, incorrectness);
            data.instance(i).setValue(data.numAttributes()- 1, entropy);
        }
        return data;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option(
                "/tNumber of cross validation folds/n",
                "C", 1, "-C"));
        newVector.addElement(new Option(
                "/tNumber of Threads/n",
                "T", 1, "-T"));
        newVector.addElement(new Option(
                "/tNumber of Threads/n",
                "I", 1, "-I"));

        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String cvFoldString = Utils.getOption('C', options);
        cvFolds = cvFoldString.length() != 0 ? Integer.parseInt(cvFoldString) : 4;

        String timeoutString = Utils.getOption('L', options);
        timeout = timeoutString.length() != 0 ? Long.parseLong(timeoutString) : 1000 * 60;

        classifierFilePath = Utils.getOption('F', options);
        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        String [] superOptions = super.getOptions();
        String [] options = new String [6 + superOptions.length];

        int current = 0;
        options[current++] = "-C";
        options[current++] = Integer.toString(cvFolds);
        options[current++] = "-L";
        options[current++] = Long.toString(timeout);
        options[current++] = "-F";
        options[current++] = classifierFilePath;

        System.arraycopy(superOptions, 0, options, current,
                superOptions.length);
        current += superOptions.length;
        while(current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    public int getCVFolds() {
        return cvFolds;
    }

    public void setCVFolds(int cvFolds) {
        this.cvFolds = cvFolds;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    public long getTimeout() {
        return timeout;
    }

    public void setTimeout(long timeout) {
        this.timeout = timeout;
    }

    public String getClassifierFilePath() {
        return classifierFilePath;
    }

    public void setClassifierFilePath(String classifierFilePath) {
        this.classifierFilePath=  classifierFilePath;
    }

    public String numThreadsToolTip() {
        return "Number of threads to build Classifiers with";
    }

    public String timeoutToolTip() {
        return "Time in seconds to allow a classifier to run for before terminating it";
    }

    public String cvFoldsTipText() {
        return "Number of cross validation folds to use when generating " +
                "predictions.";
    }

    public String classifierFilePathTipText() {
        return "Path to a file with a list of classifiers to run, each classifier" +
                "being specified with its name followed by its parameters.";
    }
}

