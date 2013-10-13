package weka.analyzers;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.gui.Logger;

import javax.swing.*;
import java.util.Enumeration;
import java.util.Vector;

/**
 * Analyzer used to compare various partitions of the DataSet against
 * each other. Allows the user to compare how statistics of different
 * attributes change across different partitions of the dataset.
 */
// In progress!
public class SubsetComparer extends Analyzer {

    /** Double to partition the data around */
    private double splitPoint = 0.0;

    /** Either the name or number of the attribute to split on */
    private String splitAttribute = "1";

    /**
     * Either the name or number of the attribute to of a classifier
     * predictions for the data
     * */
    private String predictionAttribute = "22";

    /**
     * Whether to compare the selected subset with the whole dataset or its
     * negation.
     */
    private boolean compareWithAll = true;

    /**
     * Should the subset include parts larger then or smaller or equal to
     * splitPoint.
     */
    private boolean largerThan = true;

    /**
     * Converts a String input to the attribute index it expresses.
     *
     * @param str String form of the attribute
     * @param defaultInt default int to use if not String is entered
     * @param data Data the attribute index will refer to
     * @param field Name of the field this String was for, used for error
     *              messages
     * @param allowNegOne  if -1 is an allowable input
     * @return attribute index the string represents
     */
    private int toInt(String str, int defaultInt, Instances data, String field, boolean allowNegOne) {
        if(str.length() == 0) {
            return defaultInt;
        } else if(str.equals("first")) {
            return 0;
        } else if(str.equals("last")) {
            return data.numAttributes() - 1;
        } else {
            try {
                int k = Integer.parseInt(str);
                if(k > data.numAttributes() ||
                        k <= -2 ||
                        (!allowNegOne && k == -1)) {
                    // TODO this could be more helpful
                    throw new IllegalArgumentException("Error with the range of field: " + field);
                }
                return k;
            } catch (NumberFormatException ex) {
                for(int i = 0; i < data.numAttributes(); i++) {
                    if(data.attribute(i).name().equals(str)) {
                        return i;
                    }
                }
            }
        }
        throw new IllegalArgumentException(field + " must be <last> or <first> or <attribute name> or" +
                " <attribute index>");
    }


    @Override
    public AnalyzerOutput analyzeData(Instances data, int idAtt, Logger logger) throws Exception {
        int predictionAtt = toInt(this.predictionAttribute, data.numAttributes() - 1, data,
                "Prediction Attribute", true);
        int splitAttribute = toInt(this.splitAttribute, idAtt + 1, data,
                "Split Attribute", true);
        if(idAtt != -1 && (idAtt == predictionAtt || idAtt == splitAttribute)) {
            throw new IllegalArgumentException("id attribute cannot be the split on " +
                    "prediction attribute");
        }
        if(predictionAtt == data.classIndex()) {
            throw new IllegalArgumentException("Prediction attribute cannot be class attribute");
        }
        if(predictionAtt != -1 && (
                data.attribute(predictionAtt).type() != data.classAttribute().type() ||
                data.attribute(predictionAtt).numValues() != data.classAttribute().numValues())) {
            throw new IllegalArgumentException("Prediction attribute (" + data.attribute(predictionAtt).name() +
                        ") and class attribute " + data.classAttribute().name() + " are not compatable");
        }
        if(idAtt != -1) {
            data = AnalyzerUtils.removeColumn(data, idAtt);

            // Adjust indices to account for this missing coluumn
            if(idAtt < predictionAtt) {
                predictionAtt--;
            }
            if(idAtt < splitAttribute) {
                splitAttribute--;
            }
        }
        int s1count = 0;
        for(int i = 0; i < data.numInstances(); i++) {
            double val = data.instance(i).value(splitAttribute);
            if((largerThan && val > splitPoint) ||
                    (!largerThan && val <= splitPoint)) {
                s1count++;
            }
        }
        if(s1count <= 0 || s1count >= data.numInstances()) {
            throw new IllegalArgumentException(String.format("Not enough instances in specified " +
                    "subset (Attribute %s %s %.4f)",
                    data.attribute(splitAttribute).name(),
                    (largerThan ? ">" : "<="),
                    splitPoint
                    ));
        }
        final Instances subset1 = new Instances(data, s1count);
        final Instances subset2;
        if(compareWithAll) {
            subset2 = data;
        } else {
            subset2 = new Instances(data, data.numInstances() - s1count);
        }
        for(int i = 0; i < data.numInstances(); i++) {
            double val = data.instance(i).value(splitAttribute);
            if((largerThan && val > splitPoint) ||
                    (!largerThan && val <= splitPoint)) {
                subset1.add(data.instance(i));
            } else if(!compareWithAll) {
                subset2.add(data.instance(i));
            }
        }
        subset1.setClassIndex(data.classIndex());
        subset2.setClassIndex(data.classIndex());
        GenerateVisualization[] GVs = new GenerateVisualization[1];
        GVs[0] = new GenerateVisualization() {

            @Override
            public String getName() {
                return "Comparison";
            }

            @Override
            public JPanel getVisualizerPanel() {
                try {
                    ComparisonVisualizer cv = new ComparisonVisualizer(subset1, subset2);
                    cv.setSize(800, 500);
                    return cv;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        };
        return new AnalyzerOutput("Comparison save to result list", GVs);
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>();
        newVector.addElement(new Option(
                "\tAttribute to split on\n",
                "A", 1, "-A"));
        newVector.addElement(new Option(
                "\tSplit point\n",
                "S", 1, "-S"));
        newVector.addElement(new Option(
                "\tPrediction Column\n",
                "P", 1, "-P"));
        newVector.addElement(new Option(
                "\tPrediction Column\n",
                "C", 0, "-C"));
        newVector.addElement(new Option(
                "\tLarger Than\n",
                "L", 0, "-L"));
        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        splitAttribute = Utils.getOption('A', options);

        String splitString = Utils.getOption('S', options);
        splitPoint = splitString.length() != 0 ? Double.parseDouble(splitString) : 0;

        predictionAttribute = Utils.getOption('P', options);

        compareWithAll = Utils.getFlag('C', options);
        largerThan = Utils.getFlag('L', options);

        super.setOptions(options);
    }

    @Override
    public String [] getOptions() {
          String[] superOptions = super.getOptions();
          String[] options = new String [8 + superOptions.length];

          int current = 0;
          options[current++] = "-A";
          options[current++] =  splitAttribute;
          options[current++] = "-S";
          options[current++] =  Double.toString(splitPoint);
          options[current++] = "-P";
          options[current++] =  predictionAttribute;

          if (compareWithAll) {
              options[current++] = "-C";
          }
          if (largerThan) {
              options[current++] = "-L";
          }

          System.arraycopy(superOptions, 0, options, current,
                  superOptions.length);
          current += superOptions.length;
          while(current < options.length) {
              options[current++] = "";
          }
          return options;
    }

    public String getSplitAttribute() {
        return splitAttribute;
    }

    public void setSplitAttribute(String splitAttribute) {
        this.splitAttribute = splitAttribute;
    }

    public boolean getCompareWithAll() {
        return compareWithAll;
    }

    public void setCompareWithAll(boolean compareWithAll) {
        this.compareWithAll = compareWithAll;
    }

    public boolean getLargerThan() {
        return largerThan;
    }

    public void setLargerThan(boolean largerThan) {
        this.largerThan = largerThan;
    }

    public String getPredictionAttribute() {
        return predictionAttribute;
    }

    public void setPredictionAttribute(String predictionAtt) {
        this.predictionAttribute = predictionAtt;
    }

    public double getSplitPoint() {
        return splitPoint;
    }

    public void setSplitPoint(double splitPoint) {
        this.splitPoint = splitPoint;
    }

    // TODO Tooltips
}
