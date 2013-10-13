package weka.analyzers;

import javax.swing.*;

/**
 * Abstract class for classes that can produce some visual output for
 * the user to utilize.
 */
public abstract class GenerateVisualization {

    /**
     * @return name of this visualization
     */
    public abstract String getName();

    /**
     * @return JPanel, with the size set, that displays the visualization
     * with the same name as getName()
     */
    public abstract JPanel getVisualizerPanel();

    /**
     * @return JPanel that displayed the visualizations with the
     * name and size already set
     */
    protected JPanel getVisualizerWindow() {
        JPanel jp = getVisualizerPanel();
        jp.setName(getName());
        return jp;
    }
}
