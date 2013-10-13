package weka.analyzers;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.gui.GenericObjectEditor;
import weka.gui.Logger;
import weka.gui.PropertyPanel;
import weka.gui.ResultHistoryPanel;
import weka.gui.SysErrLog;
import weka.gui.TaskLogger;
import weka.gui.explorer.Explorer;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;
import weka.gui.explorer.ExplorerDefaults;
import weka.gui.explorer.Messages;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Panel that allows the user to use a number of heterogeneous tools that aim
 * to provide information that can be of use in a learning task. Formatted
 * and used similar to the {@link weka.gui.explorer.ClassifierPanel}, results
 * are displayed in text in a window while any additional graphics (and the
 * original output text) can be found by right clicking the run in a history panel.
 */
/*
 * Code in part adapted from weka.gui.explorer.ClassifierPanel
 */
// TODO ensure Capabilities for Analyzers is working correctly
public class AnalyzePanel
        extends JPanel
        implements ExplorerPanel, LogHandler {

    /** The parent frame */
    protected Explorer explorer = null;

    /** The destination for log/status messages */
    protected Logger logger = new SysErrLog();

    /** Sends notifications when the set of working instances gets changed */
    protected PropertyChangeSupport changeSupport =
            new PropertyChangeSupport(this);

    /** Lets the user select the class column */
    protected JComboBox<String> classCombo = new JComboBox<String>();

    /** Lets the user select the ID column */
    protected JComboBox<String> IDCombo = new JComboBox<String>();

    /** Button to start running the classifier */
    protected JButton startBut = new JButton("Start");

    /** Button to stop running the classifier */
    protected JButton stopBut = new JButton("Stop");

    /** Instances to analyze */
    protected Instances instances = null;

    /** The output area for text results */
    protected JTextArea outText = new JTextArea(20, 40);

    /** A panel controlling results viewing */
    protected ResultHistoryPanel history = new ResultHistoryPanel(outText);

    /** Lets the user configure the classifier */
    protected GenericObjectEditor analyzerEditor =
            new GenericObjectEditor();

    /** The panel showing the current classifier selection */
    protected PropertyPanel analyzerPanel = new PropertyPanel(analyzerEditor);

    /** A thread that analyzer runs in */
    protected Thread runThread;

    /** Sets the Explorer to use as parent frame */
    public void setExplorer(Explorer parent) {
        explorer = parent;
    }

    /**
     * Construct a new a new AnalyzerPanel, the JPanel should not be used
     * until setInstances has been called.
     */
    public AnalyzePanel() {

        // Creating a text panel for the analyzer textual output
        outText.setEditable(false);
        outText.setFont(new Font("Monospaced", Font.PLAIN, 12));
        outText.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        outText.setLineWrap(true);
        outText.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent e) {
                if ((e.getModifiers() & InputEvent.BUTTON1_MASK)
                        != InputEvent.BUTTON1_MASK) {
                    outText.selectAll();
                }
            }
        });
        JPanel p3 = new JPanel();
        p3.setBorder(BorderFactory.createTitledBorder("Analyzer output"));
        p3.setLayout(new BorderLayout());
        final JScrollPane js = new JScrollPane(outText);
        p3.add(js, BorderLayout.CENTER);
        js.getViewport().addChangeListener(new ChangeListener() {
          private int lastHeight;
          public void stateChanged(ChangeEvent e) {
        JViewport vp = (JViewport)e.getSource();
        int h = vp.getViewSize().height;
        if (h != lastHeight) { // i.e. an addition not just a user scrolling
          lastHeight = h;
          int x = h - vp.getExtentSize().height;
          vp.setViewPosition(new Point(0, x));
        }
          }
        });

        // Setup the start, stop, class, and id boxes
        startBut.setToolTipText("Start the selected analyzer");
        stopBut.setToolTipText("Stop the running analyzer");
        classCombo.setToolTipText("Select the attribute to use as the class");
        IDCombo.setToolTipText("Select the attribute (or none) use as " +
                "an identifier for the class");

        startBut.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                startAnalyzer();
            }
        });
          stopBut.addActionListener(new ActionListener() {
              public void actionPerformed(ActionEvent e) {
                  stopAnalyzer();
              }
          });

        classCombo.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                updateCapabilitiesFilter(analyzerEditor.getCapabilitiesFilter());
            }
        });

        classCombo.setBorder(BorderFactory.createTitledBorder("Class"));
        IDCombo.setBorder(BorderFactory.createTitledBorder("ID"));

        // Create a panel to hold the settings components and add everything
        JPanel ssButs = new JPanel();
        ssButs.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        ssButs.setLayout(new GridLayout(1, 2, 5, 5));
        ssButs.add(startBut);
        ssButs.add(stopBut);

        JPanel settings = new JPanel();
        settings.setLayout(new GridLayout(3, 1, 3, 5));
        settings.add(IDCombo);
        settings.add(classCombo);
        settings.add(ssButs);

        // Panel to hold the results
        history.setBorder(BorderFactory.createTitledBorder(
                "Results list (right-click for options)"));
        history.setHandleRightClicks(false);
        history.getList().addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent e) {
                if (((e.getModifiers() & InputEvent.BUTTON1_MASK)
                        != InputEvent.BUTTON1_MASK) || e.isAltDown()) {
                    int index = history.getList().locationToIndex(e.getPoint());
                    if (index != -1) {
                        String name = history.getNameAtIndex(index);
                        buildHistoryOptionMenu(name, e.getPoint().x, e.getPoint().y);
                    }
                }
            }
        });

        // Creating the panel to select the analyzer
        analyzerEditor.setClassType(Analyzer.class);
        analyzerEditor.setValue(new MineMisclassifications());
        analyzerEditor.addPropertyChangeListener(new PropertyChangeListener() {
            public void propertyChange(PropertyChangeEvent e) {
                startBut.setEnabled(true);
                Capabilities currentFilter = analyzerEditor.getCapabilitiesFilter();
                Analyzer analyzer = (Analyzer) analyzerEditor.getValue();
                if (analyzer != null && currentFilter != null &&
                        (analyzer instanceof CapabilitiesHandler)) {
                    Capabilities currentSchemeCapabilities =
                            ((CapabilitiesHandler) analyzer).getCapabilities();

                    if (!currentSchemeCapabilities.supportsMaybe(currentFilter) &&
                            !currentSchemeCapabilities.supports(currentFilter)) {
                        startBut.setEnabled(false);
                    }
                }
                repaint();
            }
        });

        // Add everything to a center panel
        JPanel center = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        center.setLayout(gbL);

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 0;     gbC.gridx = 1;     gbC.gridheight = 2;
        gbC.weightx = 100; gbC.weighty = 100;
        gbL.setConstraints(p3, gbC);
        center.add(p3);

        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;     gbC.gridx = 0;
        gbL.setConstraints(settings, gbC);
        center.add(settings);

        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 1;     gbC.gridx = 0; gbC.weightx = 0;
        gbL.setConstraints(history, gbC);
        center.add(history);

        // Create the top label panel and add it and the center panel to this
        JPanel topPanel= new JPanel();
        topPanel.setBorder(BorderFactory.createCompoundBorder(
             BorderFactory.createTitledBorder("Analzyer"),
             BorderFactory.createEmptyBorder(0, 5, 5, 5)
             ));

        topPanel.setLayout(new BorderLayout());
        topPanel.add(analyzerPanel, BorderLayout.NORTH);

        setLayout(new BorderLayout());
        add(topPanel, BorderLayout.NORTH);
        add(center, BorderLayout.CENTER);
    }

    /**
     * Builds and shows a menu that is a list of the output the run
     * with a given name returned.
     *
     * @param name name of a analyzer that has run
     * @param x the x coordinate to show the panel
     * @param y the y coordinate to show the panel
     */
    protected void buildHistoryOptionMenu(String name, int x, int y) {
        final String selectedName = name;
        final AnalyzerOutput output = (AnalyzerOutput) history.getNamedObject(name);
        if(output.visualizers == null){
            return;
        }

        JPopupMenu resultListMenu = new JPopupMenu();
        JMenuItem analyzerText = new JMenuItem("Analyzer Text Output");
        analyzerText.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent e) {
              history.openFrame(selectedName);
          }
        });
        resultListMenu.add(analyzerText);
           for(final GenerateVisualization gv : output.visualizers) {
               JMenuItem panelOption = new JMenuItem(gv.getName());
               panelOption.addActionListener(new ActionListener() {
                   public void actionPerformed(ActionEvent e) {
                    AnalyzerUtils.openWindow(gv.getVisualizerWindow());
                   }
               });
               resultListMenu.add(panelOption);
        }
        resultListMenu.show(history.getList(), x, y);
    }

    /**
     * Stops the currently running analyzer (if any).
     */
    @SuppressWarnings("deprecation")
    protected void stopAnalyzer() {
        if (runThread != null) {
            runThread.interrupt();
        }

        /*
         To be consistent with ClassifierPanel and because Analyzers cannot
         be trusted to catch interrupts we need to use stop.
          */
        runThread.stop();
    }

    /**
     * Start running the currently selected Analyzer in a new thread.
     */
    protected void startAnalyzer() {
        if (runThread == null) {
            synchronized (this) {
                startBut.setEnabled(false);
                stopBut.setEnabled(true);
            }
            runThread = new Thread() {
          public void run() {

              int classIndex = classCombo.getSelectedIndex();
              instances.setClassIndex(classIndex);
              StringBuffer outBuff = new StringBuffer();
              Analyzer analyzer = (Analyzer) analyzerEditor.getValue();
              String name = (new SimpleDateFormat("HH:mm:ss - "))
                      .format(new Date());
            String cname = analyzer.getClass().getName();
            if (cname.startsWith("weka.analyzers.")) {
              name += cname.substring("weka.analyzers.".length());
            } else {
              name += cname;
            }
            String cmd = analyzer.getClass().getName();
              try {
          if (analyzerEditor.getValue() instanceof OptionHandler)
              cmd += " " + Utils.joinOptions((
                      (OptionHandler) analyzerEditor.getValue()).getOptions());
          logger.logMessage(cmd);
          if (logger instanceof TaskLogger) {
              ((TaskLogger) logger).taskStarted();
          }
          outBuff.append("=== Analyzer Out ===/n/n");
          outBuff.append("Analyzer: " + cname + "/n");
          outBuff.append("Full command: " + cmd + "/n");
          outBuff.append("Relation: " + instances.relationName() + "/n");
          outBuff.append("Instances: " + instances.numInstances() + "/n");
          outBuff.append("Attributes: " + instances.numAttributes() + "/n");
          outBuff.append("Class: " + instances.attribute(instances.classIndex()).name() + "/n");
          
          int idIndex = IDCombo.getSelectedIndex();
          if(idIndex == instances.numAttributes())
              idIndex = -1;
          if(idIndex == classIndex)
              throw new IllegalArgumentException("Class index and ID index cannot be equal!");
          AnalyzerOutput out = analyzer.analyzeData(instances, idIndex, logger);

          outBuff.append(out.textOutput);
          history.addObject(name, out);
          history.addResult(name, outBuff);
          history.setSingle(name);
          logger.logMessage("Done running " + cmd);
          
          } catch (Exception ex) {
              ex.printStackTrace(System.err);
            logger.logMessage("Error with: " + cmd);
            logger.statusMessage("Error");
            JOptionPane.showMessageDialog(
                    AnalyzePanel.this,
                    "Error with Analyzer: " + ex.getMessage(),
                    "Analyze Error",
                    JOptionPane.ERROR_MESSAGE);
          } finally {
          
          if(isInterrupted()) {
              logger.logMessage("Interrupted: " + cmd);
              logger.statusMessage("Interrupted");
          } 
          
          synchronized (this) {
              startBut.setEnabled(true);
              stopBut.setEnabled(false);
              runThread = null;
          }
          if (logger instanceof TaskLogger) {
              ((TaskLogger) logger).taskFinished();
          }
          }
          }
            };
            runThread.setPriority(Thread.MIN_PRIORITY);
            runThread.start();
        }
    }

    /**
     * Updates the capabilities filter of the Analyzer currently selected.
     *
     * @param filter    the new filter to use
     */
    protected void updateCapabilitiesFilter(Capabilities filter) {
        Instances tempInst;
        Capabilities filterClass;

        if (filter == null) {
            analyzerEditor.setCapabilitiesFilter(new Capabilities(null));
            return;
        }

        if (!ExplorerDefaults.getInitGenericObjectEditorFilter())
            tempInst = new Instances(instances, 0);
        else
            tempInst = new Instances(instances);
        tempInst.setClassIndex(classCombo.getSelectedIndex());

        try {
            filterClass = Capabilities.forInstances(tempInst);
        }
        catch (Exception e) {
            filterClass = new Capabilities(null);
        }

        // set new filter
        analyzerEditor.setCapabilitiesFilter(filterClass);

        startBut.setEnabled(true);
        // Check capabilities
        Capabilities currentFilter = analyzerEditor.getCapabilitiesFilter();
        Analyzer analyzer = (Analyzer) analyzerEditor.getValue();
        Capabilities currentSchemeCapabilities =  null;
        if (analyzer != null && currentFilter != null &&
                (analyzer instanceof CapabilitiesHandler)) {
            currentSchemeCapabilities = ((CapabilitiesHandler)analyzer).getCapabilities();

            if (!currentSchemeCapabilities.supportsMaybe(currentFilter) &&
                    !currentSchemeCapabilities.supports(currentFilter)) {
                startBut.setEnabled(false);
            }
        }
    }

    /** Returns the parent Explorer frame */
    public Explorer getExplorer() {
        return explorer;
    }

    /** Returns the title for the tab in the Explorer */
    public String getTabTitle() {
        return "Analyze";  // what's displayed as tab-title, e.g., Classify
    }

    /** Returns the tooltip for the tab in the Explorer */
    public String getTabTitleToolTip() {
        return "Utilities methods for analyzing your data";
    }

    /**
     * Set the Instances to use in this.
     *
     * @param inst a set of Instances
     */
    public void setInstances(Instances inst) {
         instances = inst;

         String[] attribNames = new String[instances.numAttributes()];
         for (int i = 0; i < attribNames.length; i++) {
             String type = "";
             switch (instances.attribute(i).type()) {
             case Attribute.NOMINAL:
                 type = Messages.getInstance().getString(
                         "ClassifierPanel_SetInstances_Type_AttributeNOMINAL_Text");
                 break;
             case Attribute.NUMERIC:
                 type = Messages.getInstance().getString(
                         "ClassifierPanel_SetInstances_Type_AttributeNUMERIC_Text");
                 break;
             case Attribute.STRING:
                 type = Messages.getInstance().getString(
                         "ClassifierPanel_SetInstances_Type_AttributeSTRING_Text");
                 break;
             case Attribute.DATE:
                 type = Messages.getInstance().getString(
                         "ClassifierPanel_SetInstances_Type_AttributeDATE_Text");
                 break;
             case Attribute.RELATIONAL:
                 type = Messages.getInstance().getString(
                         "ClassifierPanel_SetInstances_Type_AttributeRELATIONAL_Text");
                 break;
             default:
                 type = Messages.getInstance().getString(
                         "ClassifierPanel_SetInstances_Type_AttributeDEFAULT_Text");
             }
             attribNames[i] = type + instances.attribute(i).name();
         }
        String[] attribNamesAndNone = new String[attribNames.length + 1];
        System.arraycopy(attribNames, 0, attribNamesAndNone, 0, attribNames.length);
        attribNamesAndNone[attribNames.length] = "None";
        classCombo.setModel(new DefaultComboBoxModel<String>(attribNames));
        IDCombo.setModel(new DefaultComboBoxModel<String>(attribNamesAndNone));
        // Try to automatically detect and set to an id attribute
        // TODO be more selective about when to use this
        IDCombo.setSelectedIndex(instances.numAttributes());
        for(int i = 0; i < instances.numAttributes(); i++) {
            if(instances.attributeStats(i).distinctCount == instances.numInstances()) {
                IDCombo.setSelectedIndex(i);
            }
        }
        IDCombo.setEnabled(true);
        if (attribNames.length > 0) {
            if (inst.classIndex() == -1)
                classCombo.setSelectedIndex(attribNames.length - 1);
            else
                classCombo.setSelectedIndex(inst.classIndex());
            classCombo.setEnabled(true);
            startBut.setEnabled(runThread == null);
            stopBut.setEnabled(runThread != null);
        } else {
            startBut.setEnabled(false);
            stopBut.setEnabled(false);
        }
    }

    /** PropertyChangeListener who will be notified of value changes. */
    public void addPropertyChangeListener(PropertyChangeListener l) {
        changeSupport.addPropertyChangeListener(l);
    }

    /** Removes a PropertyChangeListener. */
    public void removePropertyChangeListener(PropertyChangeListener l) {
        changeSupport.removePropertyChangeListener(l);
    }

    /**
     * Sets the Logger this should use.
     *
     * @param newLog the Logger that will now get info messages
     */
    public void setLog(Logger newLog) {
        logger = newLog;
    }

    /**
     * Tests out the classifier panel from the command line.
     *
     * @param args contains the path of a dataset to load optionally
     *             followed by the class index
     */
    public static void main(String [] args) {
        if(args.length < 1) {
            System.out.println("Include the name of a dataset to test with " +
                    "and optionally the class index.");
            return;
        }
        try {
                final javax.swing.JFrame jf =
                    new javax.swing.JFrame("Analyze");
            jf.getContentPane().setLayout(new BorderLayout());
            final AnalyzePanel sp = new AnalyzePanel();
            jf.getContentPane().add(sp, BorderLayout.CENTER);
            weka.gui.LogPanel lp = new weka.gui.LogPanel();
            sp.setLog(lp);
            jf.getContentPane().add(lp, BorderLayout.SOUTH);
            jf.addWindowListener(new java.awt.event.WindowAdapter() {
                public void windowClosing(java.awt.event.WindowEvent e) {
                    jf.dispose();
                    System.exit(0);
                }
            });
            jf.pack();
            jf.setSize(800, 600);
            jf.setVisible(true);

            java.io.Reader r = new java.io.BufferedReader(
                    new java.io.FileReader(args[0]));
            Instances i = new Instances(r);
            i.setClassIndex(args.length == 1 ? 0 : Integer.parseInt(args[1]));
            sp.setInstances(i);
        } catch (Exception ex) {
            ex.printStackTrace();
            System.err.println(ex.getMessage());
        }
    }
}
