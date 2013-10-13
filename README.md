Weka Analyzer
=========
An extension to Weka 3.6 that adds an additional tab to Weka's explorer GUI called 'Analyze'. This tab provides a set of heterogeneous tools that analyze data and output statistics that can help the user complete machine learning tasks. The general aim of these tools is to help the user avoid common mistakes that can occur when applying machine learning and to provide insights into the data that can help users chose what classifier to use and what adjustments in their feature construction and data gathering processes might be helpful. This project was built as part of my honor's thesis at the University of Washington supervised by Oren Etzioni. I have since made a reasonable effort to ensure the code is readable and bug free, but the system has not undergone rigorous testing or extensive use so it is still liable to have bugs. My thesis paper, saved here as weka-analyzer-paper.pdf, gives a detailed overview of the motivations and implementation of this project. 

#### Building

Build the project with

```
mvn package
```

#### Integrating with Weka

To integrate the new tab with the explorer in a current installation of Weka the classes in weka-analyzer-0.0.2-classes-only.jar need to be put on the classpath. Then two properties files need to be edited, in GenericPropertiesCreator.props add the line:

```
weka.analyzers.Analyzer=\
 weka.analyzers
```

In Explorer.props add

```
weka.analyzers.AnalyzePanel
```

to the list of tabs in for the Tabs variable. Default configuration files with these modifications can be found in src/resources. 

#### Stand Alone Jar

A complete jar containing both Weka and the Analyzer's classes, in addition to updated config files shaded over Weka's original config files, is also built during the package phase using the maven shading plugin. This jar can run weka with the Analyze tab included out of the box with

```
java -jar weka-analyzer-0.0.2.jar
```