package weka.analyzers;

import weka.gui.Logger;

/**
 * Logger that prints everything to the console.
 */
public class ConsoleLogger implements Logger {

    @Override
    public void logMessage(String message) {
        System.out.println(message);
    }

    @Override
    public void statusMessage(String message) {
        System.out.println(message);
    }
}
