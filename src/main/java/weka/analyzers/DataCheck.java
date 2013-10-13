package weka.analyzers;

import weka.analyzers.AnalyzerUtils.GeneratePlotWindow;
import weka.analyzers.AnalyzerUtils.GenerateTextWindow;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.gui.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Vector;

 /**
  * Analyzer that runs a number of prepossessing checks on the data to detect
  * potential problems or significant properties that might be important to make
  * a user aware of. Includes checking for large numbers of unknown values,
  * uniform attributes,and correlations.
 */
public class DataCheck extends Analyzer {

    /** Max attribute we will try to print the correlation coeff for */
    private final int MAX_PRINTABLE_CORR_MATRIX = 200;

    // Thresholds to generate warning, or count as having lots or few unknown
    // values by percent when counting knowns per Instance or Attribute
    private final double UNKNOWNS_WARN_ATT = 0.9;
    private final double UNKNOWNS_LOTS_ATT = 0.8;
    private final double UNKNOWNS_FEW_ATT = 0.1;

    private final double UNKNOWNS_WARN_INST = 0.9;
    private final double UNKNOWNS_LOTS_INST = 0.8;
    private final double UNKNOWNS_FEW_INST = 0.1;

    /** Number of top correlations to print */
    private int numTopCorrelationsToPrint = 10;

    /** Show warning if an attribute this percent uniform */
    private double warnOnVariance = .95;

    /** Id index of the data */
    private int idIndex = -1;

    /** Indices associated with each check */
    private final int NOISE = 0;
    private final int DUPLICATES = 1;
    private final int ATTRIBUTES = 2;
    private final int CORRELATION = 3;
    private final int UNKNOWNS = 4;

    /** String description of each check we can run */
    private final String[] checkNames = {"Check for overt noise",
            "Check for duplicate instances",
            "Check for redundant attributes",
            "Examine the correlation between the attributes",
            "Check for unknowns"};

    /** Check flags */
    private final String[] checkFlags = {"N","D","A","C","U"};

    /** What checks are enabled */
    private final boolean[] doChecks = {true,true,true,true,true};

    public String globalInfo() {
        return "Runs a number of prepossessing checks on the data to detect " +
                " potential problems or significant properties that might be " +
                "import to know of.";
    }

    @Override
    public AnalyzerOutput analyzeData(Instances data, int idIndex,
                                      Logger logger) throws Exception {
        this.idIndex = idIndex;
        StringBuilder report = new StringBuilder();
        List<GenerateVisualization> GVs= new ArrayList<GenerateVisualization>();
        if(doChecks[ATTRIBUTES]) {
            logger.statusMessage("Checking for poor Attributes");
            checkAttributes(data, report);
        }
        if(doChecks[NOISE] || doChecks[DUPLICATES]) {
            if(doChecks[NOISE] && doChecks[DUPLICATES])
                logger.statusMessage("Checking for Noise and Duplicates");
            if(doChecks[NOISE])
                logger.statusMessage("Checking for Noise");
            if(doChecks[DUPLICATES])
                logger.statusMessage("Checking for Duplicates");
            checkNoiseAndDuplicates(data, report, GVs);
        }
        Instances filteredData;
        if(idIndex >= 0) {
            filteredData = AnalyzerUtils.removeColumn(data, idIndex);
        } else {
            filteredData = data;
        }
        if(doChecks[CORRELATION]) {
            logger.statusMessage("Checking Attribute Correlation");
            checkCorrelation(filteredData, report, GVs);
        }
        if(doChecks[UNKNOWNS]) {
            logger.statusMessage("Checking Unknowns");
            checkUnknowns(filteredData, report, GVs);
        }
        logger.statusMessage("DataCheck done");
        return new AnalyzerOutput(report.toString(),
                GVs.toArray(new GenerateVisualization[GVs.size()]));
    }

    /**
     * Creates a report of the number of missing attributes of either attributes or
     * instances.
     *
     * @param missingAttCounts Array of counts of missing values
     * @param attribute whether this report is for attributes or instances
     * @param fewThreshold percent of max before a count is labelled has
     *                     having a few missing values
     * @param lotsThreshold percent of max before a count is labelled has
     *                      having a many missing values
     * @param warnThreshold percent of max before a count generates a warning
     * @param max max number of missing values possible
     * @return Report of the missing values
     */
    private String writeUnknowns(int[] missingAttCounts, boolean attribute, double fewThreshold,
                double lotsThreshold, double warnThreshold, int max) {
        // TODO limit number of warnings
        // TODO cleaner to do this using sorting?
        String type = attribute ? "attributes" : "instances";
        StringBuilder curReport = new StringBuilder();
        int none = 0;  // Count of attributes with no missing values
        int few = 0; // Count of attributes with a small number of missing values
        int lots = 0; // Count of attributes with many missing values
        int fewCutoff = (int) (max * fewThreshold);
        int lotsCutoff = (int) (max * lotsThreshold);
        int warnCutoff = (int) (max * warnThreshold);
        for(int i = 0; i < missingAttCounts.length; i++) {
            int missing = missingAttCounts[i];
            if(missing== 0) {
                none++;
                continue;
            }
            if(missing >= fewCutoff) {
                few++;
            }
            if(missing >= lotsCutoff) {
                lots++;
            }
            if(missing >= warnCutoff) {
                curReport.append(String.format("WARNING %s %d has " +
                        "%.2f percent unknown values%n", type, i,
                        ((double) 100*missing) / max));
            }
        }
        curReport.append(String.format("%d of %d (%.2f%%) %s have values missing%n",
                missingAttCounts.length - none, missingAttCounts.length,
                100 - ((double) none*100) / missingAttCounts.length, type));
        if(few == 0) {
            curReport.append(String.format("No %s is missing over %.0f%% of its values%n",
                    type, fewThreshold*100));
        } else {
            curReport.append(String.format("%d of %d (%.2f%%) %s have greater than %.0f%% " +
                    "of values missing%n",
                    few, missingAttCounts.length, ((double) few*100) / missingAttCounts.length,
                    type, fewThreshold*100));
            if(lots == 0) {
                curReport.append(String.format("No %s is missing over %.0f%% of its values%n",
                        type, lotsThreshold*100));
            } else {
                curReport.append(String.format("%d of %d (%.2f%%) %s have greater than %.0f%% " +
                        "of values missing%n",
                        lots, missingAttCounts.length, ((double) lots*100) / missingAttCounts.length,
                        type, lotsThreshold*100));
            }
        }
        return curReport.toString();
    }

    /**
     * Appends a summary of the unknown values to curReport and adds
     * a more complete break down to curGVs
     *
     * @param data Instances to count unknowns for
     * @param curReport StringBuilder to append the summary to
     * @param curGVs List of GenerateVisualization to append an unknown count
     *               GenerateVisualization to
     */
    private void checkUnknowns(Instances data, StringBuilder curReport,
            List<GenerateVisualization> curGVs) {
        curReport.append("\n**** Checking Unknowns ****\n");
        StringBuilder report = new StringBuilder("Unknown values report\n");
        int[][] unknownCounts = countUnknowns(data);
        int[] attUnknownCount = unknownCounts[0];
        int[] instUnknownCount = unknownCounts[1];
        if(Utils.sum(attUnknownCount) == 0) {
            curReport.append("No unknown values");
            return;
        }

        int numInst = data.numInstances();
        int numAtt = data.numAttributes();

        curReport.append("Attribute Summary:\n");
        curReport.append(writeUnknowns(attUnknownCount, true, UNKNOWNS_FEW_ATT,
                UNKNOWNS_LOTS_ATT, UNKNOWNS_WARN_ATT, numInst));
        curReport.append("Instance Summary:\n");
        curReport.append(writeUnknowns(instUnknownCount, false, UNKNOWNS_FEW_INST,
                UNKNOWNS_LOTS_INST, UNKNOWNS_WARN_INST, numAtt));

        report.append("Unknown in attributes\n");
        displayUnknown(buildHistogram(attUnknownCount, 10, numInst), report, true);
        report.append("Unknown in instances\n");
        displayUnknown(buildHistogram(instUnknownCount, 10, numAtt), report, false);

        curReport.append("Detailed break down saved to Unknown Counts in the result list");
        curGVs.add(new GenerateTextWindow("Unknown Counts", report.toString(),
                700, 550));
    }


    /**
     * Given a histogram of unknown counts as output by buildHistogram
     * appends a summary to sb.
     *
     * @param output the histogram, array of arrays with the first index being
     *               the count and the second being the min value of the bucket
     * @param sb StringBuilder to append the summary to
     * @param attribute true iff histogram is for attributes rather
     *                   then instances
     */
    private void displayUnknown(int[][] output, StringBuilder sb, boolean attribute) {
        String type = attribute ? "attributes" : "instances";
        sb.append(String.format("%d %s with 0 missing values%n",output[0][0], type));
        for(int i = 1; i < output.length; i++) {
            sb.append(String.format("%d %s with between %d and %d unknown values %n",output[i][0], type,
                    output[i-1][1] + 1, output[i][1]));
        }
    }

    /**
     * Converts a list of values into a histogram of the values, with zero
     * always being its own bucket.
     *
     * @param input array of values to put in the histogram
     * @param buckets number of buckets to include in the histogram
     * @param max max value possible in input
     * @return int[][], where the each int[] is a tuple of the min value
     * and the number of value that were larger then that min but smaller
     * then any other min.
     */
    // TODO test this more
    public int[][] buildHistogram(int[] input, int buckets, int max) {
        buckets = Math.min(buckets, max);
        int[][] output = new int[buckets][2];

        // For n buckets we need n - 1 intervals, plus handle 0 seperately
        double divisor = buckets - 1;

        // Assign intervals at the second index, from 0 to
        output[0][1] = 0;
        output[1][1] = 1;
        for(int i = 2; i < buckets; i++) {
            output[i][1] = 1 + (int) Math.floor((i - 1) * ((max - 1) / divisor));
        }

        // Bucket the values
        for(int i = 0; i < input.length; i++) {
            if(input[i] == 0) {
                output[0][0]++;
            } else {
                int bucket = 1 + (int) Math.floor((input[i]) / ((max - 1) / divisor));
                output[bucket][0]++;
            }
        }
        return output;
    }

    /**
     * Counts the unknown values per instance and per attribute.
     *
     * @param data Instances to count unknown values from
     * @return int[][] containing an array of the number of missing values per
     * instance and an array of the number of missing values per attribute
     */
    public int[][] countUnknowns(Instances data) {
        // For instances
        int[] unknownAtt = new int[data.numAttributes()];
        int[] unknownInstances = new int[data.numInstances()];
        for(int i = 0; i < data.numInstances(); i++) {
            Instance cur = data.instance(i);
            for(int j = 0; j < data.numAttributes(); j++) {
                if(cur.isMissing(j)) {
                    unknownAtt[j]++;
                    unknownInstances[i]++;
                }
            }
        }
        return new int[][] {unknownAtt, unknownInstances};
    }

    // Builds up a count of value to ocurrances from an attribute in a Instances
    private Map<Double, Integer> count(Instances data, int attIndex) {
        Map<Double, Integer> countMap = new HashMap<Double, Integer>();
        for(int i = 0; i < data.numInstances(); i++) {
            Double val = data.instance(i).value(attIndex);
            if(countMap.containsKey(val)) {
                countMap.put(val, countMap.get(val) + 1);
            } else {
                countMap.put(val, 1);
            }
        }
        return countMap;
    }

    /**
     * Examines an Instances appends a summary of a sanity check of the
     * attributes to a StringBuilder.
     *
     * @param data Instances to examine
     * @param curReport StringBuilder to append the report to
     */
    public void checkAttributes(Instances data, StringBuilder curReport) {
        curReport.append("\n**** Checking attributes ****\n");
        int commentsAdded = 0;
        for(int i = 0; i < data.numAttributes(); i++) {
            if(i == idIndex)
                continue;
            String attName = data.attribute(i).name();
            Map<Double, Integer> countMap = count(data, i);
            if(countMap.keySet().size() < 2) {
                curReport.append("WARNING For all instances, attribute <" + attName +
                        "> had the value: " +
                        countMap.keySet().iterator().next() + "\n");
                commentsAdded++;
            } else if(data.attribute(i).isNominal() && countMap.keySet().size() > data.numInstances()/2) {
                curReport.append("Nominal Attribute " + attName + " has a very large number of (" +
                        countMap.values().size() + ") values\n");
                commentsAdded++;
            } else {
                int maxVariance = (int)(data.numInstances()*warnOnVariance);
                for(Entry<Double, Integer> entry: countMap.entrySet()) {
                    if(entry.getValue() >= maxVariance) {
                        curReport.append(String.format("Attribute %s" +
                                " has %.2f percent of instances with the value %.3f\n",
                                attName, ((double)entry.getValue())/data.numInstances(),
                                entry.getKey()));
                        commentsAdded++;
                        break;
                    }
                }
            }
        }
        if(commentsAdded == 0) {
            curReport.append("No problems found\n");
        }
    }

    /**
     * Class that represents the correlation between two attributes. Immutable.
     */
    private class Correlation implements Comparable<Correlation> {

        /** The correlation */
        public final double corr;

        /** First attribute index used for finding the correlation */
        public final int att1;

        /** Second attribute index used for finding the correlation */
        public final int att2;

        public Correlation(int att1, int att2, double corr) {
            this.corr = corr;
            this.att1 = att1;
            this.att2 = att2;
        }
        public int compareTo(Correlation other) {
            return Double.compare(Math.abs(corr), Math.abs(other.corr));
        }
    }

    /**
     * Calculates the correlation coefficient matrix between the attributes
     * of some Instances and appends a report and a GenerateVisualization to
     * the given StringBuilder and GenerateVisualization List.
     *
     * @param data Instances to examine
     * @param curReport StringBuilder to append the report to
     * @param curGVs List of GenerateVisualization to append any additional output to
     */
    public void checkCorrelation(Instances data, StringBuilder curReport,
            List<GenerateVisualization> curGVs) {
        Queue<Correlation> q = new PriorityQueue<Correlation>(numTopCorrelationsToPrint);
        int numAtt = data.numAttributes();
        curReport.append("\n**** Checking attribute correlation ****\n");
        int corrPairs = 0;
        double[][] corrMatrix = new double[numAtt][numAtt];

        double[][] avsAndVar = new double[numAtt][2];
        for(int j = 0; j < data.numInstances(); j++) {
            for(int i = 0; i < numAtt; i++) {
                avsAndVar[i][0] += data.instance(j).value(i);
            }
        }
        for(int i = 0; i < numAtt; i++) {
            avsAndVar[i][0] /= (double) data.numInstances();
        }
        for(int j = 0; j < data.numInstances(); j++) {
            for(int i = 0; i < numAtt; i++) {
                avsAndVar[i][1] += Math.pow(data.instance(j).value(i) - avsAndVar[i][0],2);
            }
        }
        for(int i = 0; i < corrMatrix.length; i++) {
            corrMatrix[i][i] = 1.0;
            for(int j = i+1; j < corrMatrix.length; j++) {
                if(avsAndVar[i][1]*avsAndVar[j][1] == 0) {
                    corrMatrix[i][j] = 0;
                } else {
                    double y12 = 0.0;
                    for(int k = 0; k < data.numInstances(); k++) {
                        y12 += (data.instance(k).value(i) - avsAndVar[i][0]) *
                                    (data.instance(k).value(j) - avsAndVar[j][0]);
                    }
                    corrMatrix[i][j] = y12 / Math.sqrt(Math.abs(avsAndVar[i][1] * avsAndVar[j][1]));
                }
                corrMatrix[j][i] = corrMatrix[i][j];
                if(Math.abs(corrMatrix[i][j]) == 1.0) {
                    curReport.append(("WARNING Attributes: " + data.attribute(i).name() + " and " +
                            data.attribute(j).name() + " are perfectly correlated\n"));
                    corrPairs++;
                } else if((1.0 - Math.abs(corrMatrix[i][j])) < 0.01) {
                    curReport.append(String.format("WARNING Attributes: %s and %s are almost " +
                            "perfectly correlated  (correlation coefficient: %3f)\n",
                            data.attribute(i).name(), data.attribute(j).name(), corrMatrix[i][j]));
                    corrPairs++;
                }
                q.add(new Correlation(i,j, corrMatrix[i][j]));
                if(q.size() > numTopCorrelationsToPrint) {
                    q.poll();
                }
            }
        }
        if(corrPairs == 0) {
            curReport.append("No attributes were very closely correlated\n");
        }
        curReport.append("Top " + numTopCorrelationsToPrint + " correlated values and correlation matrix\n" +
                "written to <Correlations> in the result lists\n");
        StringBuilder corrReport = new StringBuilder("The Strongest Correlated Attributes:\n");
        List<Correlation> pairList = new ArrayList<Correlation>();
        while(!q.isEmpty()) {
            pairList.add(q.poll());
        };
        Collections.reverse(pairList);
        for(Correlation c : pairList) {
            corrReport.append(String.format("<%s> and <%s> correlation %.5f%n",
                    data.attribute(c.att1).name(), data.attribute(c.att2).name(), c.corr));
        }
        if(MAX_PRINTABLE_CORR_MATRIX >= numAtt) {
            corrReport.append("\nCorrelation Matrix:\n");
            String[] labels = new String[data.numAttributes()];
            for(int j = 0; j < data.numAttributes(); j++) {
                labels[j] = data.attribute(j).name();
            }
            AnalyzerUtils.writeSquareMatrix(corrReport, labels, corrMatrix);
        } else {
            corrReport.append("\nCorrelation matrix to large to print\n");
        }
        curGVs.add(new GenerateTextWindow("Correlations", corrReport.toString(),
                700, 550));
    }


    /**
     * Examines an Instances for duplicate Instances and Instances that are
     * duplicates excepting class values, appends a summary to a StringBuilder
     * and a List of GenerateVisualizations.
     *
     * @param data Instances to examine
     * @param curReport StringBuilder to append the report to
     * @param curGVs List of GenerateVisualization to append any additional output to
     */
    public void checkNoiseAndDuplicates(Instances data, StringBuilder curReport, List<GenerateVisualization> curGVs) {
        Map<InstanceClassless, List<Integer>> count = new HashMap<InstanceClassless, List<Integer>>();

        StringBuilder duplicateMsg = null;
        StringBuilder noiseMsg = null;

        if(doChecks[DUPLICATES]) {
            duplicateMsg = new StringBuilder("**** Duplicate Instances ****\n" +
                    "(Instances that are identical)\n");
        }
        if(doChecks[NOISE]) {
            noiseMsg = new StringBuilder("**** Overt Noise ****\n" +
                    "(Instances that are identifcal except for class)\n");
        }
        int dupCount = 0;
        int noiseCounts = 0;

        // Build a count of occurances of unique Instances excluding id and class values
        for(int i = 0; i < data.numInstances(); i++) {
            if(i == idIndex)
                continue;
            InstanceClassless next = new InstanceClassless(data.instance(i));
            if(!count.containsKey(next)) {
                ArrayList<Integer> indices = new ArrayList<Integer>();
                indices.add(i);
                count.put(next, indices);
            } else {
                count.get(next).add(i);
            }
        }

        // Split into duplicates with and with out looking at class values
        List<Integer> duplicateInstances = new ArrayList<Integer>();
        List<Integer> noiseyInstances = new ArrayList<Integer>();
        for(Entry<InstanceClassless, List<Integer>> entry : count.entrySet()) {
            List<Integer> l = entry.getValue();
            if(l.size() < 2)
                continue;
            double firstClass = data.instance(l.get(0)).classValue();
            boolean sameClasses = true;
            for(int j = 1; j < l.size(); j++) {
                sameClasses = sameClasses && (data.instance(l.get(j)).classValue() == firstClass);
            }
            if(sameClasses) {
                dupCount += l.size();
                if(doChecks[DUPLICATES]) {
                    duplicateInstances.addAll(l);
                    writeIndices(l, data, duplicateMsg, true, idIndex);
                }
            } else {
                noiseCounts += l.size();
                if(doChecks[NOISE]) {
                    noiseyInstances.addAll(l);
                    writeIndices(l, data, noiseMsg, false, idIndex);
                }
            }
        }

        // Writes the reports
        if(doChecks[DUPLICATES]) {
            curReport.append("\n**** Checking Duplicates ****\n");
            if(dupCount == 0) {
                curReport.append("No Duplicates Found\n");
            } else {
                curReport.append(String.format("Found %d duplicates\n", dupCount));
                curReport.append("Specific duplicate instances saved to result list\n");
                curGVs.add(new GenerateTextWindow("Duplicates Text", duplicateMsg.toString(), 550, 700));
                curGVs.add(new GeneratePlotWindow("Duplicates Plot",
                        550, 700, AnalyzerUtils.instancesSubset(data, duplicateInstances)));
            }
        }
        if(doChecks[NOISE]) {
            curReport.append("\n**** Checking for Overt Noise ****\n");
            if(noiseCounts == 0) {
                curReport.append("No overtly noisey instances found\n");
            } else {
                curReport.append(String.format("WARNING Found %d noisey instances\n", noiseCounts));
                curReport.append("Specific noisey instances saved to result list\n");
                curGVs.add(new GenerateTextWindow("Noise Text", noiseMsg.toString(), 550, 700));
                curGVs.add(new GeneratePlotWindow("Noise Plot",
                        550, 700, AnalyzerUtils.instancesSubset(data, noiseyInstances)));
            }
        }
    }

    /*
     * Appends a message to appen that includes an example Instance
     * and a list of indices (if idIndex < -1) or IDs.
     * includeClass indicates if the message should be for Instances that are
     * duplicates including the class value or not.
     */
    private void writeIndices(List<Integer> indices, Instances data,
            StringBuilder appen, boolean includeClass, int idIndex) {
        if(indices.isEmpty()) {
            return;
        }
        appen.append("Feature Values: ");
        Instance example = data.instance(indices.get(0));
        for(int j = 0; j < example.numAttributes(); j++) {
            if(j != 0) {
                appen.append(",");
            }
            if(!includeClass && (j == data.classIndex())) {
                appen.append("<Class>");
            } else {
                if(data.attribute(j).isNumeric()) {
                    appen.append(example.value(j));
                } else {
                    appen.append(example.stringValue(j));
                }
            }
        }
        appen.append("\n");
        if(idIndex == -1) {
            appen.append("Indices: [" + indices.get(0));
            for(int i = 1; i < indices.size(); i++) {
                appen.append(",");
                appen.append(indices.get(i).toString());
            }
        } else {
            appen.append("IDs: [" + data.instance(indices.get(0)).value(idIndex));
            for(int i = 1; i < indices.size(); i++) {
                appen.append(",");
                appen.append(data.instance(indices.get(i)).value(idIndex));
            }

        }
        appen.append("]\n");
    }

    /**
      * Utility class to allow us to hash and compare Instances ignoring
      * the class and id values.
      */
    private class InstanceClassless {

        /** Instances to be used in the representation */
        public final Instance instance;

        public InstanceClassless(Instance instance) {
            this.instance = instance;
        }

        @Override
        public int hashCode() {
            int h = 17;
            for(int i = 0; i < instance.numValues(); i++) {
                if(i != instance.classIndex() && i != idIndex) {
                    h += instance.value(i)*37;
                }
            }
            return h;
        }

        public boolean equals(Object other) {
            if(other instanceof InstanceClassless) {
                InstanceClassless ignoreClassOther = (InstanceClassless) other;
                for(int i = 0; i < instance.numValues(); i++) {
                    if(i != instance.classIndex() && i != idIndex &&
                            (instance.value(i) != ignoreClassOther.instance.value(i))) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(1);
        for(int i = 0; i < checkNames.length; i++) {
             newVector.addElement(new Option(checkNames[i],checkFlags[i],0,"-" +
                     checkFlags[i]));
        }
        newVector.addElement(new Option("Number for top attributes to print",
                "T",1,"-T"));
        newVector.addElement(new Option("Attribute uniformity to report",
                "V",1,"-V"));
        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        for(int i = 0; i < checkNames.length; i++) {
             doChecks[i] = Utils.getFlag(checkFlags[i], options);
        }
        String topString = Utils.getOption('T', options);
        numTopCorrelationsToPrint = topString.length() != 0 ? Integer.parseInt(topString) : 10;

        String varString = Utils.getOption('V', options);
        warnOnVariance = varString.length() != 0 ? Double.parseDouble(varString) : .95;

    }

    @Override
    public String[] getOptions() {
        String[] options = new String[checkNames.length + 4];
        int current = 0;
        for(int i = 0; i < checkNames.length; i++) {
            if(doChecks[i]) {
                options[current++] = "-" + checkFlags[i];
            }
        }
        options[current++] = "-T";
        options[current++] = Integer.toString(numTopCorrelationsToPrint);
        options[current++] = "-V";
        options[current++] = Double.toString(warnOnVariance);
        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    // Getters, Settersm and Tooltips

    public void setNumTopCorrelationsToPrint(int num) {
        numTopCorrelationsToPrint = num;
    }

    public int getNumTopCorrelationsToPrint() {
        return numTopCorrelationsToPrint;
    }

    public void setReportAttributeVariance(double percent) {
        warnOnVariance = percent;
    }

    public double getReportAttributeVariance() {
        return warnOnVariance;
    }

    public void setCheckNoise(boolean check) {
        doChecks[NOISE] = check;
    }

    public boolean getCheckNoise() {
        return doChecks[NOISE];
    }

    public void setCheckDuplicates(boolean check) {
        doChecks[DUPLICATES] = check;
    }

    public boolean getCheckDuplicates() {
        return doChecks[DUPLICATES];
    }

    public void setCheckAttributes(boolean check) {
        doChecks[ATTRIBUTES] = check;
    }

    public boolean getCheckAttributes() {
        return doChecks[ATTRIBUTES];
    }

    public void setCheckCorrelation(boolean check) {
        doChecks[CORRELATION] = check;
    }

    public boolean getCheckCorrelation() {
        return doChecks[CORRELATION];
    }

    public void setCheckUnknown(boolean check) {
        doChecks[UNKNOWNS] = check;
    }

    public boolean getCheckUnknown() {
        return doChecks[UNKNOWNS];
    }

    public String checkCorrelationtTipText() {
        return "Check for perfectly correlated (and thus redundant) attributes, " +
                "saves the correlation matrix and lists the attributes that have" +
                " the strongest correlation to each other.";
    }

    public String checkNoiseTipText() {
        return "Check for instances that are noise (identical except for class" +
                "value). Noisy instances can indicate label inaccuracy or lack of " +
                "adequate features needed to differentiate the examples.";
    }

    public String checkDuplicateTipText() {
        return "Check for instances that are duplicates of each other. ";
    }

    public String checkUnknownTipText() {
        return "Check for count unknowns in data. ";
    }

    public String checkAttributeTipText() {
        return "Check for attributes that have the same value for all instances " +
                "or mostly the same value for all instances.";
    }

    public String reportAttributeVarianceTipText() {
        return "How many identical values an attribute must have to be reported. Should be " +
                "between .50 and 1.0";
    }

    public String numTopCorrelationsToPrintTipText() {
        return "If the correlation check is run, what number of the most " +
                "correlated attributes should we print?";
    }
}
