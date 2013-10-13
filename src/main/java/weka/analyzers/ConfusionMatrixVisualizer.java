package weka.analyzers;

import weka.analyzers.AnalyzerUtils.GenerateVisualizerWindow;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

/**
 * GenerateVisualizer for a confusion matrix where the user
 * can right-click cells to draw up a list of IDs or a plot of
 * Instances in that cell.
 */
public class ConfusionMatrixVisualizer extends GenerateVisualizerWindow {

    /** Number of IDs to print per a line */
    private final int ID_LINE_WIDTH = 110;

    /** Width of the Visualization to display */
    private final int VISUALIZATION_WIDTH = 700;

    /** Height of the Visualization to display */
    private final int VISUALIZATION_HEIGHT = 700;

    /** Instances the will display */
    private final Instances data;

    /** Predictions for each Instance */
    private final int[] classPredictions;

    /** Attribute to use an ID, -1 if there is no such attribute */
    private final int idIndex;

    /**
     * Constructs a new ConfusionMatrixVisualizer.
     *
     * @param name name the visualization should have
     * @param height height the visualization should have
     * @param width width height the visualization should have
     * @param data Instances to display
     * @param predictions array of arrays of votes per class per Instance
     * @param idIndex  Attribute to use an ID, -1 if there is no such attribute
     */
    public ConfusionMatrixVisualizer(String name, int height, int width,
            Instances data, double[][] predictions, int idIndex) {
        super(name, height, width);
        this.data = data;
        this.idIndex = idIndex;
        classPredictions = new int[predictions.length];
        for(int i = 0; i < predictions.length; i++){
            classPredictions[i] = Utils.maxIndex(predictions[i]);
        }
    }

    @Override
    protected JComponent generateJComponent() {
        final int numClasses = data.classAttribute().numValues();
        final String[] columnNames = new String[numClasses + 1];

        // Holds a list of indices for every class,class pair
        // Implicity, pair ((true class, predicted class) ->
        //    trueC cass * num classes + predicted class)
        final List<List<Integer>> indices =
                new ArrayList<List<Integer>>(numClasses * numClasses);
        final int[][] counts = new int[numClasses][numClasses];

        // Build a list of the columns names
        columnNames[0] = "True Class";
        for(int i = 0; i < numClasses; i++) {
            columnNames[i + 1] = "Prediction: " + data.classAttribute().value(i);
        }

        // Fill the indices list
        for(int j = 0; j < numClasses * numClasses; j++) {
            indices.add(new ArrayList<Integer>());
        }
        for(int i = 0; i < data.numInstances(); i++) {
            int trueClass = (int) data.instance(i).classValue();
            int predClass = classPredictions[i];
            indices.get(trueClass * numClasses + predClass).add(i);
            counts[trueClass][predClass]++;
        }

        // Table model for the table to build
        class MyTableModel extends AbstractTableModel {

            public int getColumnCount() {
                return columnNames.length;
            }

            public int getRowCount() {
                return columnNames.length - 1;
            }

            public String getColumnName(int col) {
                return columnNames[col];
            }

            public Object getValueAt(int row, int col) {
                if(col == 0) {
                    return data.classAttribute().value(row);
                }
                return counts[row][col - 1];
            }

            @SuppressWarnings({ "unchecked", "rawtypes" })
            public Class getColumnClass(int c) {
                return getValueAt(0, c).getClass();
            }
        }

        final JTable jt = new JTable(new MyTableModel());
        // On mouse click display the plot of the Instances in the cell or, if
        // there is an idAttribute give the option of ids or plot.
        jt.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent mouseEvent) {
                if(!SwingUtilities.isRightMouseButton(mouseEvent)) {
                    return;
                }
                int selectedColumn = jt.convertColumnIndexToModel(jt.getSelectedColumn());
                if(selectedColumn == 0) {
                    return;
                }
                int selectedRow = jt.convertRowIndexToModel(jt.getSelectedRow());
                Instances selectedData = AnalyzerUtils.instancesSubset(data,
                        indices.get(selectedColumn - 1 + selectedRow * numClasses));
                PlotData2D plot = new PlotData2D(selectedData);
                final VisualizePanel p = new VisualizePanel();
                try {
                    p.addPlot(plot);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                p.setSize(VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT);
                if(idIndex == -1) {
                    AnalyzerUtils.openWindow(p);
                } else {
                    JPopupMenu menu = new JPopupMenu();
                    JMenuItem plotItem = new JMenuItem("Plot");
                    plotItem.addActionListener(new ActionListener() {
                           public void actionPerformed(ActionEvent e) {
                               AnalyzerUtils.openWindow(p);
                           }
                       });
                    JMenuItem idsItem = new JMenuItem("IDs");
                    final StringBuilder sb = new StringBuilder("IDs\n");
                    int curLineWidth = 0;
                    for(int i = 0; i < selectedData.numInstances(); i++) {
                        String idStr = AnalyzerUtils.attValStr(selectedData.attribute(idIndex),
                                selectedData.instance(i).value(idIndex));
                        curLineWidth += idStr.length();
                        if(curLineWidth > ID_LINE_WIDTH) {
                            sb.append("\n");
                            sb.append(idStr);
                            curLineWidth = 0;
                        } else {
                            sb.append(" ");
                            sb.append(idStr);
                        }
                    }
                    idsItem.addActionListener(new ActionListener() {
                           public void actionPerformed(ActionEvent e) {
                               JPanel panel = new JPanel();
                               panel.setName("Instance Ids");
                               panel.setSize(VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT);
                               panel.add(AnalyzerUtils.generateTextField(sb.toString()));
                               AnalyzerUtils.openWindow(panel);
                           }
                       });
                    menu.add(plotItem);
                    menu.add(idsItem);
                    menu.show(jt, mouseEvent.getPoint().x, mouseEvent.getPoint().y);
                }
            }
        });

        // Format the table and a label
        jt.setCellSelectionEnabled(true);
        jt.setRowSelectionAllowed(false);
        jt.setCellSelectionEnabled(false);
        jt.setFillsViewportHeight(true);
        JScrollPane js = new JScrollPane(jt);

        final JLabel label = new JLabel("(Right click to view specific instances)");
        label.setAlignmentX(JLabel.CENTER_ALIGNMENT);
        label.setAlignmentY(JLabel.CENTER_ALIGNMENT);
        label.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        JPanel output = new JPanel();
        output.setLayout(new BorderLayout());
        output.add(label, BorderLayout.NORTH);
        output.add(js, BorderLayout.CENTER);
        return output;
    }
}
