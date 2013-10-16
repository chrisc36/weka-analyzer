package weka.analyzers;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableRowSorter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

/**
 * Panel used to compare subset of an Instances against each other.
 *
 */
public class ComparisonVisualizer extends JPanel {

    /** Array of subsets we are comparing */
    private Instances[] subsets;

    /** List of the Statistics we are displaying about the subsets */
    private List<Stats> stats;

    // GUI elements

    /** Model for the statistic table to be shown */
    private ComparisonTableModel tableModel;

    /** Combo box to select statistics */
    private JComboBox<String> statsSelectionCombo = new JComboBox<String>();

    /** Panel to hold the statistic table */
    private JPanel statPanel;

    /** Table of the statistics */
    private JTable comparisonTable;

    public ComparisonVisualizer(Instances data, Instances data2) throws Exception {
        this(data, data2, -1);
    }

    public ComparisonVisualizer(Instances data, Instances data2, int predictionAtt) throws Exception {
        subsets = new Instances[]{data, data2};
        stats = new ArrayList<Stats>();
        statsSelectionCombo = new JComboBox<String>();

        Mean mean = new Mean(subsets);
        addStat(new NominalDist(subsets));
        addStat(mean);
        addStat(new StdDev(subsets));
        addStat(new ASEvaluationStat<GainRatioAttributeEval>("Gain Ratio",
                subsets, new GainRatioAttributeEval()));
//        if(predictionAtt != -1) {
//            addStat(new ConfusionMatrixStats(header, 2, predictionAtt));
//        }
        tableModel = new ComparisonTableModel(stats.get(0));
        TableRowSorter<ComparisonTableModel> sorter = new TableRowSorter<ComparisonTableModel>(tableModel);
        comparisonTable = new JTable(tableModel);
        comparisonTable.setRowSorter(sorter);
        comparisonTable.setCellSelectionEnabled(true);
        comparisonTable.setRowSelectionAllowed(false);
        comparisonTable.setFillsViewportHeight(true);

        statsSelectionCombo.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent arg0) {
                tableModel.stats = stats.get(statsSelectionCombo.getSelectedIndex());
                updateTable();
            }
        });

        statPanel = new JPanel();
        statPanel.setBorder(BorderFactory.createTitledBorder("Display"));
        statPanel.add(statsSelectionCombo);

        JScrollPane js = new JScrollPane(comparisonTable);
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        add(statPanel);
        add(js);
    }

    /**
     * Class for representing Statistics about several subsets of Instances.
     */
    private abstract class Stats {

        /** Name of this Statistic */
        private final String name;

        protected Stats(String name){
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public void setAllInstances(Instances[] instanceses){
            for(int i = 0; i < instanceses.length; i++){
                setInstances(instanceses[i], i);
            }
        }

        public abstract int getNumRows();

        public abstract String getRowName(int row);

        public abstract void setInstances(Instances instances, int subset);

        public abstract double getStat(int subset, int row);
    }

    /**
     * Abstract Class of Statistics that cache statistics in a double array.
     */
    private abstract class CachedStats extends Stats {
        protected final double[][] values;

        public CachedStats(String name, int numSubsets){
            super(name);
            values = new double[numSubsets][];
        }

        @Override
        public int getNumRows() {
            return values[0].length;
        }

        @Override
        public void setInstances(Instances instances, int subset) {
            values[subset] = calcInstancesStats(instances);
        }

        @Override
        public double getStat(int subset, int row) {
            return values[subset][row];
        }

        protected abstract double[] calcInstancesStats(Instances instances);
    }

    /**
     * Class of statics that generate a value for each attribute.
     */
    private abstract class PerAttributeStat extends CachedStats {

        private final int[] attsToUse;
        private final String attPrfix;
        private final Instances header;

        protected PerAttributeStat(String name, String attPrfix,
                                   Instances[] instanceses){
            super(name, instanceses.length);
            this.header = new Instances(instanceses[0], 0);
            this.attPrfix = attPrfix;
            List<Integer> attList = new ArrayList<Integer>();
            for(int i = 0; i < header.numAttributes(); i++){
                if(useAttribute(header.attribute(i))){
                    attList.add(i);
                }
            }
            attsToUse = new int[attList.size()];
            for(int i = 0; i < attsToUse.length; i++){
                attsToUse[i] = attList.get(i);
            }
            setAllInstances(instanceses);
        }

        public String getRowName(int row){
            return attPrfix + ": " + header.attribute(attsToUse[row]).name();
        }

        protected double[] calcInstancesStats(Instances instances){
            double[] stats = new double[attsToUse.length];
            for(int i = 0; i < attsToUse.length; i++){
                stats[i] = calcStat(instances, attsToUse[i]);
            }
            return  stats;
        }

        protected abstract boolean useAttribute(Attribute att);

        abstract double calcStat(Instances instances, int attIndex);
    }

    /**
     * Mean of each numeric attribute.
     */
    public class Mean extends PerAttributeStat {

        public Mean(Instances[] instanceses){
            super("Mean", "Mean: ", instanceses);
        }

        @Override
        protected boolean useAttribute(Attribute att) {
            return att.isNumeric();
        }

        @Override
        double calcStat(Instances instances, int attIndex) {
            double sum = 0.0;
            for(int i = 0; i < instances.numInstances(); i++){
                sum += instances.instance(i).value(attIndex);
            }
            return sum / instances.numInstances();
        }
    }

    /**
     * Standard deviation of each attribute.
     */
    public class StdDev extends PerAttributeStat {

        public StdDev(Instances[] instanceses){
            super("Standard Deviation", "std: ", instanceses);
        }

        @Override
        protected boolean useAttribute(Attribute att) {
            return att.isNumeric();
        }

        @Override
        double calcStat(Instances instances, int attIndex) {
            // TODO avoid the extra copy by calculating by hand
            return Math.sqrt(Utils.variance(instances.attributeToDoubleArray(attIndex)));
        }
    }

    /**
     * Use ASEvaluation & AttributeEvaluator class to build statistics for
     * each attribute.
     *
     * @param <T> Type of ASEvaluation & AttributeEvaluator to use.
     */
    public class ASEvaluationStat<T extends ASEvaluation & AttributeEvaluator> extends CachedStats {

        private final T evaluator;
        private final Instances header;

        public ASEvaluationStat(String name, Instances[] instanceses, T evaluator) {
            super(name, instanceses.length);
            this.evaluator = evaluator;
            header = new Instances(instanceses[0], 0);
            setAllInstances(instanceses);
        }

        @Override
        protected double[] calcInstancesStats(Instances instances) {
            double[] stats = new double[instances.numAttributes()];
            try {
                evaluator.buildEvaluator(instances);
                for(int i = 0; i < stats.length; i++){
                    stats[i] = evaluator.evaluateAttribute(i);
                }
            } catch (Exception e) {
                // TODO handle this
                throw new RuntimeException();
            }
            return stats;
        }

        @Override
        public String getRowName(int row) {
            return header.attribute(row).name();
        }
    }

    /**
     * Distribution of nominal values for nominal attributes.
     */
    public class NominalDist extends CachedStats {

        private final String[] rowNames;

        public NominalDist(Instances instanceses[]) {
            super("Nominal Distributions", instanceses.length);
            List<String> names = new ArrayList<String>();
            for(int i = 0; i < instanceses[0].numAttributes(); i++){
                Attribute att = instanceses[0].attribute(i);
                if(att.isNominal()){
                    for(int j = 0; j < att.numValues(); j++){
                      names.add(att.name() + "=" + att.value(j));
                    }

                }
            }
            rowNames = new String[names.size()];
            names.toArray(rowNames);
        }

        @Override
        protected double[] calcInstancesStats(Instances instances) {
            double[] stats = new double[rowNames.length];
            for(int i = 0; i < instances.numAttributes(); i++){
                Attribute att = instances.attribute(i);
                int offset = 0;
                if(att.isNominal()){
                    for(int j = 0; j < instances.numInstances(); j++){
                        stats[offset + (int) instances.instance(j).value(i)]++;
                    }
                    for(int j = 0; j < att.numValues(); j++) {
                        stats[offset + j] /= instances.numInstances();
                    }
                    offset += att.numValues();
                }
            }
            return stats;
        }

        @Override
        public String getRowName(int row) {
            return rowNames[row];
        }
    }


 /*
    public class ConfusionMatrixStats extends Stats {

        private Instances header;
        private int predAtt;

        public ConfusionMatrixStats(Instances header, int subsets, int predAtt) throws Exception {
            super("Confusion Matrix", subsets);
            this.header = header;
            for(int i = 0; i < subsets; i++) {
                stats[i] = new InstancesConfusionMatrix(predAtt);
            }
            this.predAtt = predAtt;
            numRows = header.numClasses();
            numRows *= numRows;
        }

        @Override
        protected String getRowName(int row) {
            int[] cols = getEntry(row, numRows);
            return header.classAttribute().value(cols[0]) +
                    " (Pred " + header.attribute(predAtt).value(cols[1]) + ")";
        }
    }

    private int[] getEntry(int row, int numRows) {
        numRows = (int) Math.sqrt(numRows);
        int[] out = new int[2];
        out[0] = row % numRows; // Actual
        out[1] = row / numRows; // Prediction
        return out;
    }
        */

    /**
     * Class for displaying a particular statistic in a table.
     */
    private class ComparisonTableModel extends AbstractTableModel {

        Stats stats;

        public ComparisonTableModel(Stats stats) {
            this.stats = stats;
        }

        public int getColumnCount() {
            return 4;
        }

        public int getRowCount() {
            return stats.getNumRows();
        }

        public String getColumnName(int col) {
            switch (col) {
            case 0: return stats.name;
            case 1: return subsets[0].relationName();
            case 2: return subsets[1].relationName();
            case 3: return "Difference";
            default: throw new RuntimeException();
            }
        }

        public Object getValueAt(int row, int col) {
            try{
                switch (col) {
                case 0: return stats.getRowName(row);
                case 1: return stats.getStat(0, row);
                case 2: return stats.getStat(1, row);
                case 3: return stats.getStat(0, row) - stats.getStat(1, row);
                default: throw new RuntimeException();
                }
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        @SuppressWarnings({ "unchecked", "rawtypes" })
        public Class getColumnClass(int col) {
            switch (col) {
            case 0: return String.class;
            case 1: return Double.class;
            case 2: return Double.class;
            case 3: return Double.class;
            default: throw new RuntimeException();
            }
        }
    }

    private void updateTable() {
        comparisonTable.updateUI();
    }

    private void addStat(final Stats s) {
        stats.add(s);
        try {
            s.setAllInstances(subsets);
        } catch (Exception e) {
            e.printStackTrace();
        }
        statsSelectionCombo.addItem(s.name);
    }

    /**
     * Test out the visualizer.
     *
     * @param args Path to the data to test on
     * @throws Exception if there was a problem building the panel
     */
    public static void main(String[] args) throws Exception {
        final javax.swing.JFrame jf =
                new javax.swing.JFrame("Analyze");
        java.io.Reader r = new java.io.BufferedReader(
                new java.io.FileReader(args[0]));
        Instances i = new Instances(r);
        i = AnalyzerUtils.removeColumn(i, 0);
        i.setClass(i.attribute(i.numAttributes()- 1));
        jf.getContentPane().setLayout(new BorderLayout());
        final ComparisonVisualizer cv = new ComparisonVisualizer(i, i, -1);
        jf.getContentPane().add(cv, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
                System.exit(0);
            }
        });
        jf.pack();
        jf.setSize(800, 600);
        jf.setVisible(true);
    }
}

