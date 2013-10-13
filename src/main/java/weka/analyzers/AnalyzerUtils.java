package weka.analyzers;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;
import java.awt.*;
import java.util.Collection;
import java.util.Random;

/**
 * General utility class for Analyzers.
 */
public final class AnalyzerUtils {

    // Should not be instantiated
    private AnalyzerUtils() {}

    /**
     * Abstract class that implements GenerateVisualization and takes care of
     * storing and setting the name and dimensions of the JPanel.
     */
    public abstract static class GenerateVisualizerWindow extends GenerateVisualization {

        /** Height of the window to generate. **/
        protected int height;

        /** Width of the window to generate. **/
        protected int width;

        /** Name of the window to generate. **/
        protected String name;

        public GenerateVisualizerWindow(String name, int height, int width) {
            this.name = name;
            this.width = width;
            this.height = height;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public JPanel getVisualizerPanel() {
            JPanel jp = new JPanel();
            jp.setSize(width, height);
            jp.setLayout(new BorderLayout());
            jp.add(generateJComponent(), BorderLayout.CENTER);
            return jp;
        }

        /**
         * Returns a JComponent to be displayed by this window.
         *
         * @return JComponent to display
         */
        protected abstract JComponent generateJComponent();
    }

    /**
     * Implementation of GenerateVisualizerWindow that can be used for
     * displaying text.
     */
    public static class GenerateTextWindow extends GenerateVisualizerWindow {

        /** Text to display */
        private String text;

        public GenerateTextWindow(String name, String text, int height,
                                  int width) {
            super(name, height, width);
            this.text = text;
        }


        @Override
        protected JComponent generateJComponent() {
            return generateTextField(text);
        }
    }


    /**
     * Open a new window displaying a JPanel.
     *
     * @param panel JPanel to display, its name and dimension will
     *              be used when formatting the window.
     */
    public static void openWindow(final JPanel panel) {
        final javax.swing.JFrame jf = new javax.swing.JFrame(panel.getName());
        jf.setSize(panel.getWidth(), panel.getHeight());
        jf.add(panel, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });
        jf.setVisible(true);
    }

    /**
     * Implementation of GenerateWindowVisualizerWindow that can be used to
     * display plots of Instances.
     */
    public static class GeneratePlotWindow extends GenerateVisualizerWindow {

        /** Instances to display */
        private Instances data;

        public GeneratePlotWindow(String name, int height,
                                  int width, Instances data) {
            super(name, height, width);
            this.data = data;
        }

        @Override
        protected JComponent generateJComponent() {
            return buildPlot(data);
        }
    }

    /**
     * Builds a VisualizePanel to display some Instances as a Plot.
     *
     * @param data Instances that can be displayed on a VisualizePanel
     * @return VisualizePanel that displays that data as a Plot
     */
    public static VisualizePanel buildPlot(Instances data) {
        PlotData2D plot = new PlotData2D(data);
        VisualizePanel p = new VisualizePanel();
        try {
            p.addPlot(plot);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return p;
    }

    /**
     * Removes a column form some Instances. Requires making a copy of the Instances.
     *
     * @param instances Instances to remove the column from
     * @param columnIndex index of the column to remove, cannot be the class column
     * @return a copy of the Instances without the column
     * @throws Exception if there was problem removing the column
     */
    public static Instances removeColumn(Instances instances, int columnIndex) throws Exception {
        if(columnIndex == instances.classIndex())
            throw new IllegalArgumentException("Cannot remove class index");
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(new int[] {columnIndex});
        remove.setInputFormat(instances);
        Instances newData = Filter.useFilter(instances, remove);
        int classIndex = instances.classIndex();
        if(classIndex >= 0){
            // Might need shift the class index so it points at the
            // same attribute as before
            if (columnIndex < classIndex)
                classIndex--;
            newData.setClassIndex(classIndex);
        }
        return newData;
    }

    /**
     * Shuffles instances and other in and an identical manner so that
     * an instance and double at the same index before the shuffle
     * will be at the same index after the shuffle.
     *
     * @param instances Instances to shuffle
     * @param other double array to shuffle of the same length
     */
    public static void shuffleInSync(Instances instances, double[] other) {
        shuffleInSync(instances, other, new Random());
    }

    /**
     * Shuffles instances and other in and an identical manner so that
     * an instance and int  at a given index before the shuffle
     * will be at the same index after the shuffle.
     *
     * @param instances Instances to shuffle
     * @param other int array to shuffle of the same length
     */
    public static void shuffleInSync(Instances instances, int[] other) {
        shuffleInSync(instances, other, new Random());
    }

    /**
     * Shuffles instances and other in and an identical manner so that
     * an instance and double at a given index before the shuffle
     * will be at the same index after the shuffle. Uses rnd to do
     * the shuffling.
     *
     * @param instances Instances to shuffle
     * @param other double array to shuffle of the same length
     * @param rnd Random to use when shuffling
     */
    static void shuffleInSync(Instances instances, double[] other, Random rnd) {
        if(instances.numInstances() != other.length) {
            throw new IllegalArgumentException();
        }
        for (int i = other.length - 1; i >= 0; i--) {
            int index = rnd.nextInt(i + 1);
            instances.swap(index, i);
            double a = other[index];
            other[index] = other[i];
            other[i] = a;
        }
    }

    /**
     * Shuffles instances and other in and an identical manner so that
     * an instance and double at a given index before the shuffle
     * will be at the same index after the shuffle. Uses rnd to do
     * the shuffling.
     *
     * @param instances Instances to shuffle
     * @param other double array to shuffle of the same length
     * @param rnd Random to use when shuffling
     */
    static void shuffleInSync(Instances instances, int[] other, Random rnd) {
        if(instances.numInstances() != other.length) {
            throw new IllegalArgumentException();
        }
        for (int i = other.length - 1; i >= 0; i--) {
            int index = rnd.nextInt(i + 1);
            instances.swap(index, i);
            int a = other[index];
            other[index] = other[i];
            other[i] = a;
        }
    }

    /**
     * Gets a subset of an Instances.
     *
     * @param instances Instances to get the subset from
     * @param indices List of indices to include in the subset
     * @return the subset
     */
    public static Instances instancesSubset(Instances instances, Collection<Integer> indices) {
        Instances subset = new Instances(instances, indices.size());
        for(Integer i : indices) {
            subset.add(instances.instance(i));
        }
        return subset;
    }

    /**
     * Generates a scroll panel around some text.
     *
     * @param text text to use
     * @return JScrollPanel that frames the text
     */
    public static JScrollPane generateTextField(String text) {
        JTextArea ta = new JTextArea();
        ta.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        ta.setFont(new Font("Monospaced", Font.PLAIN, 12));
        ta.setEditable(false);
        ta.setText(text);
        ta.setCaretPosition(0);
        JScrollPane js = new JScrollPane(ta);
        return js;
    }

    /**
     * Converts a value from a given attribute into a String.
     *
     * @param att Attribute the value is from
     * @param val the double encoded value
     * @return the String version of that value
     */
    public static String attValStr(Attribute att, double val) {
        if(att.isNominal() || att.isString()) {
            // Look up and return the string value for Nominal and String
            return att.value((int)val);
        } else {
            if(val == (int)val) {
                // Do not print decimal for integers
                return String.format("%,d", (int)val);
            }
            return String.format("%,.4f", val);
        }
    }

    /**
     * Returns the entropy of the numbers in an array
     * after normalizing them.
     *
     * @param arr int array to get the entropy for
     * @return the entropy
     */
    public static double entropy(int[] arr) {
        double sum = Utils.sum(arr);
        double total = 0;
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] != 0) {
                double t = arr[i]/sum;
                total -= t  * Math.log(t);
            }
        }
        return total;
    }

    /**
     * Calculates a confusion matrix based on an Instances and an array of
     * class predictions.
     *
     * @param data Instances the prediction were made for
     * @param predictions predictions for each instances
     * @return double[][] where entry[i][j] is the number of Instances with true
     * class i and prediction j
     */
    public static double[][] confusionMatrix(Instances data, double[] predictions) {
        double[][] confusionMatrix =
                new double[data.classAttribute().numValues()][data.classAttribute().numValues()];
        for(int i = 0; i < data.numInstances(); i++) {
            confusionMatrix[(int) data.instance(i).classValue()][(int)predictions[i]]++;
        }
        return confusionMatrix;
    }

    /**
     * Calculates a confusion matrix based on an Instances and an array of
     * prediction counts for each class for each attribute.
     *
     * @param data Instances the prediction were made for
     * @param predictions predictions for each instances. predictions[i][j] should
     *                    be the number of vote to classify Instance i as class j
     * @return double[][] where entry[i][j] is the number of Instances with true
     * class i and prediction j
     */
    public static double[][] confusionMatrix(Instances data, double[][] predictions) {
        double[][] confusionMatrix =
                new double[data.classAttribute().numValues()][data.classAttribute().numValues()];
        for(int i = 0; i < data.numInstances(); i++) {
            int prediction = Utils.maxIndex(predictions[i]);
            confusionMatrix[(int) data.instance(i).classValue()][prediction]++;
        }
        return confusionMatrix;
    }

    /*
     * Methods to prints matrices as Strings, there is code to do this in weka
     * in weka.classifiers.Evaluation but it is not readily usable
     */
    // TODO it might be better to use weka's style of symbols for row

    /**
     * Takes a square matrix of String values and appends those value to
     * a StringBuilder taking care to ensure all columns in the matrix
     * are alighed.
     *
     * @param sb StringBuilder to appends the result to
     * @param values Square String matrix of the values to use
     * @param min minimum width a column can have
     * @param max maximum width a column can have, longer String are truncated
     * @param pad number of space to put between columns
     */
    public static void writeMatrix(StringBuilder sb, String[][] values,
                                   int min, int max, int pad) {
        StringBuilder formatStrBuilder = new StringBuilder();
        StringBuilder padBuilder = new StringBuilder(pad);
        for (int i = 0; i < pad; i++) {
           padBuilder.append(" ");
        }
        String padStr = padBuilder.toString();
        for(int i = 0; i < values[0].length; i++) {
            int colMax = min;
            for(int j = 0; j < values.length; j++) {
                String val = values[j][i];
                if(val.length() > max) {
                    val = val.substring(0,max - 1) + ".";
                    values[j][i] = val;
                }
                colMax = Math.max(colMax, val.length());
            }
            formatStrBuilder.append("%-" + Integer.toString(colMax) + "s" + padStr);
        }
        formatStrBuilder.append("\n");
        String formatStr = formatStrBuilder.toString();
        for(int i = 0; i < values.length; i++) {
            sb.append(String.format(formatStr, (Object[])values[i]));
        }
    }

    /**
     * Takes a square matrix of values and an array of labels and
     * appends the values written as a square matrix with the labels
     * being used on both the columns and rows.
     *
     * @param sb StringBuilder to appends the result to
     * @param labels Strings to label the rows and columns with
     * @param values Square matrix of the values to use
     */
    public static void writeSquareMatrix(StringBuilder sb, String[] labels, double[][] values) {
        writeMatrix(sb, labels, labels, values, 1, 12, 2);
    }

    /**
     * Writes a matrix of values as a String with columns aligned, uses an
     * array of labels to use as the first column and first row.
     *
     * @param sb StringBuilder to appends the result to
     * @param rowNames Strings to label the rows with
     * @param colNames Strings to label the columns with
     * @param values Square matrix of the values to us
     * @param min min width of the columns
     * @param max max width of the columns, longer columns will be truncated
     * @param pad spaces between columns
     */
    public static void writeMatrix(StringBuilder sb, String[] rowNames, String[] colNames,
            double[][] values, int min, int max, int pad) {
        int rowOffset = rowNames == null ? 0 : 1;
        int colOffset = colNames == null ? 0 : 1;
        String[][] matrixValues = new String[values.length + rowOffset]
                                            [values[0].length + colOffset];
        if(colNames != null) {
            matrixValues[0][0] = "";
            System.arraycopy(colNames, 0, matrixValues[0], 1, colNames.length);
        }
        for(int i = 0; i < values.length; i++) {
            if(rowNames != null) {
                matrixValues[i + rowOffset][0] = rowNames[i];
            }
            for(int j = 0; j < values[i].length; j++) {
                double v = values[i][j];
                if(v == (int) v)
                    matrixValues[i + rowOffset][j + colOffset] =
                        String.format("%d", (int)v);
                else
                    matrixValues[i + rowOffset][j + colOffset] =
                        String.format("%.3f", v);
            }
        }
        writeMatrix(sb, matrixValues, min, max, pad);
    }
}
